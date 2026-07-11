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

Metrics run_backtest(const std::vector<double>& closes) {
    double capital = 1000000.0;
    double initial_capital = capital;
    double position = 0.0; // Net shares
    
    std::vector<double> portfolio_values;
    portfolio_values.push_back(capital);
    
    for (size_t t = 20; t < closes.size(); ++t) {
        // --- COMPONENT INJECTION ---
        double high20 = closes[t], low20 = closes[t];
for(int i=1; i<20 && t-i>=0; i++) {
    if(closes[t-i] > high20) high20 = closes[t-i];
    if(closes[t-i] < low20) low20 = closes[t-i];
}
double mid = (high20 + low20) / 2.0;
double signal = (closes[t] > mid) ? 1.0 : -1.0;
        double mean_val = 0; for(int i=0; i<10 && t-i>=0; i++) mean_val += closes[t-i]; mean_val /= 10.0;
double dev = 0; for(int i=0; i<10 && t-i>=0; i++) dev += pow(closes[t-i] - mean_val, 2);
double stdev = sqrt(dev / 10.0) + 1e-6;
double position_size = std::min(800.0, (capital * 0.01) / stdev);
        double exec_price = closes[t];
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
    
    // Assuming roughly 5000 ticks per day, 252 days a year
    double sharpe = (mean_ret / std_ret) * std::sqrt(252.0 * 5000.0);
    
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
    // We generate dummy mean-reverting data for the 3-month NIFTY500 backtest.
    // 3 months = ~60 trading days. At 1 min resolution = 60 * 375 = 22,500 bars.
    int n_bars = 22500;
    std::vector<double> closes(n_bars);
    closes[0] = 20000.0;
    
    // Generate some auto-correlated random walk with a sine wave overlay
    for(int i = 1; i < n_bars; ++i) {
        double noise = ((rand() % 1000) / 1000.0 - 0.5) * 10.0;
        double trend = std::sin(i / 100.0) * 5.0;
        closes[i] = closes[i-1] + noise + trend;
    }
    
    Metrics m = run_backtest(closes);
    
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
