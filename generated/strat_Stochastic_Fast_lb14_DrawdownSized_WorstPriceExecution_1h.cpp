#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>

// ----------------------------------------------------------------------------
// Base Strategy Template for AIN Combinatorial Backtester
// Compatible with: SIMD OFI, Hawkes Process Intensity, and Avellaneda Skews
// ----------------------------------------------------------------------------

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
    
    double prev_signal = 0.0;
    double prev_position_size = 0.0;
    double prev_slippage_factor = 0.0005;
    bool prev_execute = false;
    
    double peak = capital;
    
    for (size_t t = 26; t < closes.size(); ++t) {
        // 1. Enforce strict 1-bar execution lag: Execute previous signal at current close
        if (prev_execute) {
            double target_position = prev_signal * prev_position_size;
            if (target_position != position) {
                double trade_qty = target_position - position;
                
                // Enforce strict order book slippage: Buy at Ask, Sell at Bid
                double slip = (trade_qty > 0) ? (1.0 + prev_slippage_factor) : (1.0 - prev_slippage_factor);
                double exec_price_with_friction = closes[t] * slip;
                
                double cost = trade_qty * exec_price_with_friction;
                double fee = std::abs(cost) * 0.0001; // 1 bp commission fee
                
                capital -= (cost + fee);
                position = target_position;
            }
        }
        
        // M2M Portfolio
        double port_val = capital + (position * closes[t]);
        portfolio_values.push_back(port_val);
        
        // Running peak and drawdown track
        if (port_val > peak) peak = port_val;
        double current_dd = (peak - port_val) / peak;
        
        // Hard Margin Stop: Instantly liquidate if drawdown >= 15%
        if (current_dd >= 0.15) {
            return { -99.0, 15.0, -100.0, port_val - initial_capital };
        }
        
        // 2. NOW generate new signal at the end of bar t (using history strictly up to closes[t])
        // --- COMPONENT INJECTION ---
        double high14 = closes[t], low14 = closes[t];
for(int i=1; i<14 && t-i>=0; i++) {
    if(closes[t-i] > high14) high14 = closes[t-i];
    if(closes[t-i] < low14) low14 = closes[t-i];
}
double k_fast = (high14 == low14) ? 50.0 : 100.0 * (closes[t] - low14) / (high14 - low14);
double signal = (k_fast < 20.0) ? 1.0 : ((k_fast > 80.0) ? -1.0 : 0.0);
        double peak_val = initial_capital;
for(double val : portfolio_values) { if(val > peak_val) peak_val = val; }
double dd = (peak_val - portfolio_values.back()) / peak_val;
double size_multiplier = std::max(0.1, 1.0 - (dd * 10.0));
double position_size = std::min(3.0 * capital / closes[t], 150.0 * size_multiplier);
        double slippage_factor = 0.0010;
bool execute = true;
        // ---------------------------
        
        prev_signal = signal;
        prev_position_size = position_size;
        prev_execute = execute;
        prev_slippage_factor = slippage_factor;
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
    double dd_peak = portfolio_values[0];
    for(double val : portfolio_values) {
        if(val > dd_peak) dd_peak = val;
        double dd = (dd_peak - val) / dd_peak;
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
            double drift = 3000.0 / n_bars;
            double vol = 1200.0 / std::sqrt(n_bars);
            for(int i = 1; i < n_bars; ++i) {
                double noise = ((rand() % 1000) / 1000.0 - 0.5) * 2.0 * vol;
                closes[i] = closes[i-1] + drift + noise;
            }
        }
        else if (r == 1) {
            // 2. Aggressive Bear Market: Sharp downward drift, high volatility
            double drift = -4000.0 / n_bars;
            double vol = 2500.0 / std::sqrt(n_bars);
            for(int i = 1; i < n_bars; ++i) {
                double noise = ((rand() % 1000) / 1000.0 - 0.5) * 2.0 * vol;
                closes[i] = closes[i-1] + drift + noise;
            }
        }
        else if (r == 2) {
            // 3. Mean Reverting Sideways: Oscillates around pivot
            double vol = 1500.0 / std::sqrt(n_bars);
            double pivot = 20000.0;
            for(int i = 1; i < n_bars; ++i) {
                double noise = ((rand() % 1000) / 1000.0 - 0.5) * 2.0 * vol;
                double pull = (pivot - closes[i-1]) * (30.0 / n_bars);
                closes[i] = closes[i-1] + noise + pull;
            }
        }
        else if (r == 3) {
            // 4. High Volatility Chop: Violent non-directional moves
            double vol = 5000.0 / std::sqrt(n_bars);
            for(int i = 1; i < n_bars; ++i) {
                double noise = ((rand() % 1000) / 1000.0 - 0.5) * 2.0 * vol;
                closes[i] = closes[i-1] + noise;
            }
        }
        else if (r == 4) {
            // 5. Low Volatility Squeeze: Very narrow channel
            double vol = 500.0 / std::sqrt(n_bars);
            double pivot = 20000.0;
            for(int i = 1; i < n_bars; ++i) {
                double noise = ((rand() % 1000) / 1000.0 - 0.5) * 2.0 * vol;
                double pull = (pivot - closes[i-1]) * (20.0 / n_bars);
                closes[i] = closes[i-1] + noise + pull;
            }
        }
        else if (r == 5) {
            // 6. Jump Diffusion Flash Crash: Steady trend with a sudden flash crash jump at midpoint
            double vol = 800.0 / std::sqrt(n_bars);
            double drift = 600.0 / n_bars;
            int jump_bar = (int)(n_bars * 0.5);
            for(int i = 1; i < n_bars; ++i) {
                double noise = ((rand() % 1000) / 1000.0 - 0.5) * 2.0 * vol;
                double jump = 0.0;
                if (i == jump_bar) jump = -3000.0; // Flash crash gap down of 15%
                closes[i] = closes[i-1] + drift + noise + jump;
            }
        }
        else if (r == 6) {
            // 7. Momentum Breakout Spike: Flat then parabolic spike
            double vol = 800.0 / std::sqrt(n_bars);
            for(int i = 1; i < n_bars; ++i) {
                double noise = ((rand() % 1000) / 1000.0 - 0.5) * 2.0 * vol;
                double breakout = 0.0;
                if (i > n_bars * 0.6) breakout = 4000.0 / (0.4 * n_bars);
                closes[i] = closes[i-1] + noise + breakout;
            }
        }
        else if (r == 7) {
            // 8. Volatility Clustered GARCH: Alternating calm/wild periods
            double base_vol = 1200.0 / std::sqrt(n_bars);
            double current_vol = base_vol;
            for(int i = 1; i < n_bars; ++i) {
                current_vol = 0.98 * current_vol + 0.02 * (base_vol * (1.0 + 3.0 * (rand() % 100 < 5)));
                double noise = ((rand() % 1000) / 1000.0 - 0.5) * 2.0 * current_vol;
                closes[i] = closes[i-1] + noise;
            }
        }
        else if (r == 8) {
            // 9. Trending Channel Oscillator: Trend + oscillations
            double drift = 2000.0 / n_bars;
            double vol = 1000.0 / std::sqrt(n_bars);
            for(int i = 1; i < n_bars; ++i) {
                double noise = ((rand() % 1000) / 1000.0 - 0.5) * 2.0 * vol;
                double prev_osc = std::sin(2.0 * 3.141592653589793 * (i-1) * 3.0 / n_bars) * 1000.0;
                double curr_osc = std::sin(2.0 * 3.141592653589793 *  i    * 3.0 / n_bars) * 1000.0;
                closes[i] = closes[i-1] + drift + noise + (curr_osc - prev_osc);
            }
        }
        else {
            // 10. Liquidity Shock Gap: Frequent gaps
            double vol = 800.0 / std::sqrt(n_bars);
            for(int i = 1; i < n_bars; ++i) {
                double noise = ((rand() % 1000) / 1000.0 - 0.5) * 2.0 * vol;
                double gap = 0.0;
                if (rand() % 100000 < (500000.0 / n_bars)) {
                    gap = ((rand() % 1000) / 1000.0 - 0.5) * 3000.0;
                }
                closes[i] = closes[i-1] + noise + gap;
            }
        }
        results[r] = run_backtest(closes, annual_factor);
    }
    
    // Helper function to safely output doubles without NaN/Inf breaking JSON parsing
    auto clean_val = [](double v) -> double {
        if (std::isnan(v) || std::isinf(v)) return 0.0;
        if (v > 1e12) return 1e12;
        if (v < -1e12) return -1e12;
        return v;
    };
    
    // Output strictly JSON mapping each regime to its metrics
    std::cout << "{" << std::endl;
    for(int r = 0; r < 10; ++r) {
        std::cout << "  \"" << regime_names[r] << "\": {"
                  << "\"sharpe\": " << clean_val(results[r].sharpe)
                  << ", \"max_dd\": " << clean_val(results[r].max_dd)
                  << ", \"ann_return\": " << clean_val(results[r].ann_return)
                  << ", \"total_pnl\": " << clean_val(results[r].total_pnl)
                  << "}";
        if (r < 9) std::cout << ",";
        std::cout << std::endl;
    }
    std::cout << "}" << std::endl;
              
    return 0;
}
