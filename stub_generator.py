import os

def generate_institutional_stubs(output_path: str):
    header_content = """#pragma once
#include <vector>
#include <string>
#include <unordered_map>

// ----------------------------------------------------------------------------
// Data Structures
// ----------------------------------------------------------------------------
struct TickData {
    long long timestamp;
    double bid_price;
    double ask_price;
    double bid_size;
    double ask_size;
    double last_price;
    double volume;
};

struct OrderUpdate {
    std::string order_id;
    std::string status;
    double filled_price;
    double filled_qty;
    long long timestamp;
};

// ----------------------------------------------------------------------------
// Interfaces for an Institutional Trading Strategy
// ----------------------------------------------------------------------------
class IInstitutionalStrategy {
public:
    virtual ~IInstitutionalStrategy() = default;

    // 1. Initialization and Lifecycle
    virtual void on_init() = 0;
    virtual void on_start() = 0;
    virtual void on_stop() = 0;
    virtual void on_timer(long long current_timestamp) = 0;

    // 2. Market Data Handlers (LOB & Trades)
    virtual void on_tick(const TickData& tick) = 0;
    virtual void on_orderbook_snapshot(const std::vector<TickData>& depth) = 0;

    // 3. Alpha / Signal Generation
    // Returns a raw alpha score [-1.0 to 1.0] representing directional conviction
    virtual double calculate_alpha() = 0;
    
    // Updates internal volatility, momentum, and mean-reversion indicators
    virtual void update_indicators(const TickData& tick) = 0;

    // 4. Portfolio Construction & Sizing
    // Translates alpha scores into a target position size
    virtual double generate_target_position(double alpha_score) = 0;

    // 5. Risk Management & Compliance
    // Returns true if the proposed trade passes all risk limits (Max DD, Inventory limits, VaR)
    virtual bool check_risk_limits(double proposed_trade_qty) = 0;

    // 6. Execution & Order Management
    virtual void execute_trade(double target_position) = 0;
    virtual void on_order_update(const OrderUpdate& update) = 0;
    virtual void cancel_all_orders() = 0;

    // 7. Accounting & Performance
    virtual void update_pnl() = 0;
    virtual double get_current_sharpe() = 0;
    virtual double get_drawdown() = 0;
};
"""
    with open(output_path, "w") as f:
        f.write(header_content)
    print(f"[*] Successfully generated institutional trading stubs at: {output_path}")

if __name__ == "__main__":
    output_file = os.path.join(os.path.dirname(__file__), "institutional_strategy_interface.h")
    generate_institutional_stubs(output_file)
