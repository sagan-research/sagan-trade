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
    std::vector<double> portfolio_values;
};

Metrics run_backtest(const std::vector<double>& closes, double annual_factor) {
    double capital = 1000000.0;
    double initial_capital = capital;
    double position = 0.0; // Net shares
    
    std::vector<double> portfolio_values;
    portfolio_values.push_back(capital);
    
    for (size_t t = 200; t < closes.size(); ++t) {
        // --- COMPONENT INJECTION ---
        double roc = (t >= 20) ? (closes[t] - closes[t-20]) / closes[t-20] : 0.0;
double signal = (roc > 0.005) ? 1.0 : ((roc < -0.005) ? -1.0 : 0.0);
        double variance = 0;
for(int i=0; i<10 && t-i>=0; i++) variance += pow(closes[t-i] - closes[t], 2);
variance /= 10.0;
double vol = sqrt(variance) + 1e-6;
double position_size = std::min(1000.0, 500.0 / vol);
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
    
    // Assuming dynamic timeframe resolution
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
    
    return {sharpe, max_dd * 100.0, ann_return * 100.0, portfolio_values.back() - initial_capital, portfolio_values};
}

int main(int argc, char* argv[]) {
    // Dynamic timeframe injections
    std::string tf = "1m";
    int n_bars = 22500;
    double annual_factor = std::sqrt(252.0 * 375.0);
    
    std::vector<double> closes(n_bars);
    closes[0] = 20000.0;
    
    // Generate price data matching the timeframe noise profile
    // 1d has larger price movements than 1m bars
    double noise_scale = 10.0;
    if (tf == "5m") noise_scale = 22.0;
    else if (tf == "15m") noise_scale = 38.0;
    else if (tf == "1h") noise_scale = 75.0;
    else if (tf == "1d") noise_scale = 200.0;
    
    for(int i = 1; i < n_bars; ++i) {
        double noise = ((rand() % 1000) / 1000.0 - 0.5) * noise_scale;
        double trend = std::sin(i / 100.0) * (noise_scale * 0.5);
        closes[i] = closes[i-1] + noise + trend;
    }
    
    Metrics m = run_backtest(closes, annual_factor);
    
    // Output strictly JSON so Python can parse it
    std::cout << "{\"sharpe\": " << m.sharpe 
              << ", \"max_dd\": " << m.max_dd 
              << ", \"ann_return\": " << m.ann_return 
              << ", \"total_pnl\": " << m.total_pnl
              << ", \"portfolio_values\": [";
    
    for(size_t i = 0; i < m.portfolio_values.size(); ++i) {
        std::cout << m.portfolio_values[i];
        if (i < m.portfolio_values.size() - 1) std::cout << ", ";
    }
    std::cout << "]}" << std::endl;
              
    return 0;
}
