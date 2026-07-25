#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>

struct Metrics {
    double sharpe;
    double max_dd;
    double ann_return;
    double total_pnl;
};

Metrics run_backtest(const std::vector<double>& closes, double annual_factor) {
    double capital = 1000000.0;
    double initial_capital = capital;
    double position = 0.0; // Net shares
    
    std::vector<double> portfolio_values;
    portfolio_values.push_back(capital);
    
    for (size_t t = 200; t < closes.size(); ++t) {
        // --- COMPONENT INJECTION ---
        double roc = (t >= 8) ? (closes[t] - closes[t-8]) / closes[t-8] : 0.0;
double signal = (roc > 0.005) ? 1.0 : ((roc < -0.005) ? -1.0 : 0.0);
        double position_size = (capital * 0.001) / closes[t];
        double exec_price = closes[t] - (signal * closes[t] * 0.0002);
bool execute = (signal != 0.0);
        // ---------------------------
        
        // Execute trade if requested and not crossing some basic constraints
        if (execute && signal != 0.0) {
            double target_position = signal * position_size;
            double trade_qty = target_position - position;
            
            // Assume 0.01% slippage & fee
            double cost = trade_qty * exec_price;
            double fee = std::abs(cost) * 0.0001;
            
            capital -= (cost + fee);
            position = target_position;
        }
        
        // M2M Portfolio
        double port_val = capital + (position * closes[t]);
        portfolio_values.push_back(port_val);
    }
    
    // Calculate Returns & Metrics
    std::vector<double> returns;
    for(size_t i = 1; i < portfolio_values.size(); ++i) {
        returns.push_back((portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]);
    }
    
    double mean_ret = 0, sum_sq = 0;
    for(double r : returns) mean_ret += r;
    if (returns.size() > 0) mean_ret /= returns.size();
    
    for(double r : returns) sum_sq += (r - mean_ret) * (r - mean_ret);
    double std_ret = returns.size() > 1 ? std::sqrt(sum_sq / (returns.size() - 1)) : 1.0;
    if (std_ret == 0) std_ret = 1e-6;
    
    // Sharpe resolution
    double sharpe = (mean_ret / std_ret) * annual_factor;
    
    double max_dd = 0;
    double peak = portfolio_values[0];
    for(double val : portfolio_values) {
        if(val > peak) peak = val;
        double dd = (peak - val) / peak;
        if(dd > max_dd) max_dd = dd;
    }
    
    // Annualized return (assuming 3 months = 0.25 years of data)
    double total_ret = (portfolio_values.back() - initial_capital) / initial_capital;
    double ann_return = (std::pow(1.0 + total_ret, 4.0) - 1.0); // 3 months -> power of 4
    
    return {sharpe, max_dd * 100.0, ann_return * 100.0, portfolio_values.back() - initial_capital};
}

int main(int argc, char* argv[]) {
    // Dynamic timeframe injections
    std::string tf = "1h";
    int n_bars = 375;
    double annual_factor = std::sqrt(252.0 * 6.25);
    
    std::vector<std::string> regime_names = {
        "Steady_Bull_Trend", "Aggressive_Bear_Market", "Mean_Reverting_Sideways",
        "High_Volatility_Chop", "Low_Volatility_Squeeze", "Jump_Diffusion_Flash_Crash",
        "Momentum_Breakout_Spike", "Vol_Clustered_GARCH", "Trending_Channel_Oscillator",
        "Liquidity_Shock_Gap"
    };
    
    // Noise scaling logic
    double noise_scale = 10.0;
    if (tf == "5m") noise_scale = 22.0;
    else if (tf == "15m") noise_scale = 38.0;
    else if (tf == "1h") noise_scale = 75.0;
    else if (tf == "1d") noise_scale = 200.0;
    
    std::vector<Metrics> results(10);
    
    for (int r = 0; r < 10; ++r) {
        std::vector<double> closes(n_bars);
        closes[0] = 20000.0;
        srand(r * 1234 + 7); // Set stable seed for reproducible regimes
        
        if (r == 0) {
            // 1. Steady Bull Trend: Smooth upward drift, low volatility
            for(int i = 1; i < n_bars; ++i) {
                double noise = ((rand() % 1000) / 1000.0 - 0.4) * (noise_scale * 0.4);
                closes[i] = closes[i-1] + noise;
            }
        }
        else if (r == 1) {
            // 2. Aggressive Bear Market: Sharp downward drift, high volatility
            for(int i = 1; i < n_bars; ++i) {
                double noise = ((rand() % 1000) / 1000.0 - 0.6) * (noise_scale * 1.5);
                closes[i] = closes[i-1] + noise;
            }
        }
        else if (r == 2) {
            // 3. Mean Reverting Sideways: Oscillates around pivot
            double pivot = 20000.0;
            for(int i = 1; i < n_bars; ++i) {
                double noise = ((rand() % 1000) / 1000.0 - 0.5) * noise_scale;
                double pull = (pivot - closes[i-1]) * 0.05;
                closes[i] = closes[i-1] + noise + pull;
            }
        }
        else if (r == 3) {
            // 4. High Volatility Chop: Violent non-directional moves
            for(int i = 1; i < n_bars; ++i) {
                double noise = ((rand() % 1000) / 1000.0 - 0.5) * (noise_scale * 4.0);
                closes[i] = closes[i-1] + noise;
            }
        }
        else if (r == 4) {
            // 5. Low Volatility Squeeze: Very narrow channel
            double pivot = 20000.0;
            for(int i = 1; i < n_bars; ++i) {
                double noise = ((rand() % 1000) / 1000.0 - 0.5) * (noise_scale * 0.15);
                double pull = (pivot - closes[i-1]) * 0.02;
                closes[i] = closes[i-1] + noise + pull;
            }
        }
        else if (r == 5) {
            // 6. Jump Diffusion Flash Crash: Steady trend with sudden spikes
            for(int i = 1; i < n_bars; ++i) {
                double noise = ((rand() % 1000) / 1000.0 - 0.48) * (noise_scale * 0.5);
                double jump = 0;
                if (rand() % 1000 < 5) jump = -(noise_scale * 25.0); // flash crash gap down
                closes[i] = closes[i-1] + noise + jump;
            }
        }
        else if (r == 6) {
            // 7. Momentum Breakout Spike: Flat then parabolic spike
            for(int i = 1; i < n_bars; ++i) {
                double noise = ((rand() % 1000) / 1000.0 - 0.5) * (noise_scale * 0.2);
                double breakout = 0;
                if (i > n_bars * 0.6) breakout = (noise_scale * 1.5);
                closes[i] = closes[i-1] + noise + breakout;
            }
        }
        else if (r == 7) {
            // 8. Volatility Clustered GARCH: Alternating calm/wild periods
            double vol = noise_scale * 0.5;
            for(int i = 1; i < n_bars; ++i) {
                vol = 0.95 * vol + 0.05 * std::abs(((rand() % 1000) / 1000.0 - 0.5) * (noise_scale * 3.0));
                double noise = ((rand() % 1000) / 1000.0 - 0.5) * vol;
                closes[i] = closes[i-1] + noise;
            }
        }
        else if (r == 8) {
            // 9. Trending Channel Oscillator: Trend + oscillations
            for(int i = 1; i < n_bars; ++i) {
                double noise = ((rand() % 1000) / 1000.0 - 0.5) * (noise_scale * 0.8);
                double trend = (noise_scale * 0.2);
                double osc = std::sin(i / 50.0) * (noise_scale * 1.0);
                closes[i] = closes[i-1] + noise + trend + osc;
            }
        }
        else {
            // 10. Liquidity Shock Gap: Frequent gaps
            for(int i = 1; i < n_bars; ++i) {
                double noise = ((rand() % 1000) / 1000.0 - 0.5) * (noise_scale * 0.3);
                double gap = 0;
                if (rand() % 100 < 3) gap = ((rand() % 1000) / 1000.0 - 0.5) * (noise_scale * 8.0);
                closes[i] = closes[i-1] + noise + gap;
            }
        }
        
        results[r] = run_backtest(closes, annual_factor);
    }
    
    // Output strictly JSON mapping each regime to its metrics
    std::cout << "{" << std::endl;
    for(int r = 0; r < 10; ++r) {
        std::cout << "  \"" << regime_names[r] << "\": {"
                  << "\"sharpe\": " << (std::isnan(results[r].sharpe) ? 0.0 : results[r].sharpe)
                  << ", \"max_dd\": " << (std::isnan(results[r].max_dd) ? 0.0 : results[r].max_dd)
                  << ", \"ann_return\": " << (std::isnan(results[r].ann_return) ? 0.0 : results[r].ann_return)
                  << ", \"total_pnl\": " << (std::isnan(results[r].total_pnl) ? 0.0 : results[r].total_pnl)
                  << "}";
        if (r < 9) std::cout << ",";
        std::cout << std::endl;
    }
    std::cout << "}" << std::endl;
              
    return 0;
}
