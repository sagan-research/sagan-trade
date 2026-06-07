// State variables
let globalData = null;
let activeTicker = "RELIANCE";

// Chart instances for recycling
let equityChartInstance = null;
let gatingChartInstance = null;
let spreadChartInstance = null;

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", () => {
    fetchData();
});

// Fetch data.json compiled by run.py
async function fetchData() {
    try {
        const response = await fetch("data.json");
        if (!response.ok) {
            throw new Error("data.json file not found. Ensure run.py was executed successfully.");
        }
        globalData = await response.json();
        
        // Populate sidebar stock navigation
        populateStockList();
        
        // Render initial active stock
        renderStockDashboard(activeTicker);
    } catch (error) {
        console.error("Error loading dashboard data:", error);
        document.body.innerHTML += `
            <div class="glass-panel" style="position:fixed; top:50%; left:50%; transform:translate(-50%,-50%); padding:2rem; text-align:center; z-index:9999; border-color:var(--neon-magenta);">
                <h2 style="color:var(--neon-magenta); margin-bottom:1rem;">⚠️ Data Load Error</h2>
                <p style="color:var(--text-secondary); max-width:450px;">${error.message}</p>
                <button onclick="location.reload()" style="margin-top:1.5rem; padding:0.6rem 1.2rem; background:rgba(255,255,255,0.05); border:1px solid var(--border-glass); border-radius:6px; color:#fff; cursor:pointer;">Reload</button>
            </div>
        `;
    }
}

// Populate the Sidebar stocks listing
function populateStockList() {
    const listElement = document.getElementById("stockList");
    listElement.innerHTML = "";
    
    const tickers = Object.keys(globalData);
    
    tickers.forEach(ticker => {
        const isHigh = ["RELIANCE", "HDFCBANK", "INFY"].includes(ticker);
        const badgeClass = isHigh ? "high" : "low";
        const badgeLabel = isHigh ? "High Liq" : "Low Liq";
        
        const li = document.createElement("li");
        li.className = ticker === activeTicker ? "active" : "";
        li.dataset.ticker = ticker;
        li.innerHTML = `
            <span>${ticker}</span>
            <span class="liq-badge ${badgeClass}">${badgeLabel}</span>
        `;
        
        li.addEventListener("click", () => {
            document.querySelectorAll("#stockList li").forEach(el => el.classList.remove("active"));
            li.classList.add("active");
            activeTicker = ticker;
            renderStockDashboard(ticker);
        });
        
        listElement.appendChild(li);
    });
}

// Main render controller
function renderStockDashboard(ticker) {
    const data = globalData[ticker];
    if (!data) return;
    
    const m = data.metrics;
    
    // Update labels
    document.getElementById("activeStockName").innerText = data.name.toUpperCase();
    document.getElementById("activeStockDesc").innerText = `Out-of-sample HF spread-arbitrage and market-making backtest outcomes`;
    
    // Format numeric stats
    document.getElementById("metricReturn").innerText = `${m.total_return_pct >= 0 ? "+" : ""}${m.total_return_pct}%`;
    document.getElementById("metricSharpe").innerText = m.sharpe_ratio;
    document.getElementById("metricDD").innerText = `${m.max_drawdown}%`;
    
    // Net fee formatting
    const netFees = m.net_fees;
    const netFeesElement = document.getElementById("metricFees");
    if (netFees <= 0) {
        // Earned a net rebate!
        netFeesElement.className = "metric-value text-green";
        netFeesElement.innerText = `Rebate: ₹${Math.abs(netFees).toLocaleString('en-IN', { maximumFractionDigits: 2 })}`;
    } else {
        // Paid a net fee
        netFeesElement.className = "metric-value text-red";
        netFeesElement.innerText = `Fee: ₹${netFees.toLocaleString('en-IN', { maximumFractionDigits: 2 })}`;
    }
    
    // Set Symbolic formula text and description
    document.getElementById("symbolicEquation").innerHTML = `g(X<sub>resid</sub>) = ${formatEquationLatex(data.symbolic_equation)}`;
    document.getElementById("symbolicId").innerText = data.symbolic_formula;
    document.getElementById("symbolicParams").innerText = `Params: [${data.symbolic_params.map(p => p.toFixed(4)).join(", ")}]`;
    
    // Update trade counts
    document.getElementById("makerTradeCount").innerText = m.total_maker_trades;
    document.getElementById("takerTradeCount").innerText = m.total_taker_trades;
    document.getElementById("slippageCostVal").innerText = `₹${m.slippage_losses.toLocaleString('en-IN', { maximumFractionDigits: 2 })}`;
    
    // Create charts
    createEquityChart(data.equity_curve);
    createGatingChart(data.gating_weights);
    createSpreadChart(data.actual_spreads, data.predicted_spreads);
    
    // Create trade log
    renderTradeLog(data.trade_logs);
}

// Custom parser to format raw LaTeX math expressions into HTML friendly output
function formatEquationLatex(eq) {
    return eq
        .replace(/\\cdot/g, " · ")
        .replace(/\\sigma_t/g, "σ<sub>t</sub>")
        .replace(/\\lambda_t/g, "λ<sub>t</sub>")
        .replace(/\\sin/g, "sin")
        .replace(/\\cos/g, "cos")
        .replace(/\\Delta_t/g, "Δ<sub>t</sub>")
        .replace(/\\text\{OFI\}_t/g, "OFI<sub>t</sub>")
        .replace(/c_1/g, "c<sub>1</sub>")
        .replace(/c_2/g, "c<sub>2</sub>");
}

// Equity Chart builder
function createEquityChart(equityCurve) {
    const ctx = document.getElementById("equityChart").getContext("2d");
    if (equityChartInstance) {
        equityChartInstance.destroy();
    }
    
    const labels = Array.from({ length: equityCurve.length }, (_, i) => i);
    
    // Draw gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, "rgba(195, 0, 255, 0.25)");
    gradient.addColorStop(1, "rgba(195, 0, 255, 0.0)");
    
    equityChartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [{
                label: "Portfolio Capital (₹)",
                data: equityCurve,
                borderColor: "#c300ff",
                borderWidth: 2.5,
                pointRadius: 0,
                fill: true,
                backgroundColor: gradient,
                tension: 0.15
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { color: "rgba(255,255,255,0.02)" },
                    ticks: { color: "#8b9bb4", font: { size: 10 } }
                },
                y: {
                    grid: { color: "rgba(255,255,255,0.05)" },
                    ticks: { 
                        color: "#8b9bb4",
                        font: { size: 10 },
                        callback: function(value) {
                            return "₹" + (value / 10000000).toFixed(2) + "Cr";
                        }
                    }
                }
            }
        }
    });
}

// MoE routing weights chart builder (Area)
function createGatingChart(weights) {
    const ctx = document.getElementById("gatingChart").getContext("2d");
    if (gatingChartInstance) {
        gatingChartInstance.destroy();
    }
    
    // Sample weights to avoid plotting too many dense ticks on chart (e.g. plot every 5th tick)
    const sampledWeights = [];
    const step = 5;
    for (let i = 0; i < weights.length; i += step) {
        sampledWeights.push(weights[i]);
    }
    
    const labels = Array.from({ length: sampledWeights.length }, (_, i) => i * step);
    
    const exp1 = sampledWeights.map(w => w[0]);
    const exp2 = sampledWeights.map(w => w[1]);
    const exp3 = sampledWeights.map(w => w[2]);
    
    gatingChartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Exp 1 (Calm/Fast)",
                    data: exp1,
                    borderColor: "#00f0ff",
                    backgroundColor: "rgba(0, 240, 255, 0.1)",
                    fill: "origin",
                    pointRadius: 0,
                    tension: 0.2,
                    borderWidth: 1.5
                },
                {
                    label: "Exp 2 (Volatile)",
                    data: exp2,
                    borderColor: "#ff007b",
                    backgroundColor: "rgba(255, 0, 123, 0.1)",
                    fill: "-1",
                    pointRadius: 0,
                    tension: 0.2,
                    borderWidth: 1.5
                },
                {
                    label: "Exp 3 (Microstructure/OFI)",
                    data: exp3,
                    borderColor: "#c300ff",
                    backgroundColor: "rgba(195, 0, 255, 0.1)",
                    fill: "-1",
                    pointRadius: 0,
                    tension: 0.2,
                    borderWidth: 1.5
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "top",
                    labels: { color: "#8b9bb4", boxWidth: 12, font: { size: 9, family: "Outfit" } }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: "#8b9bb4", font: { size: 9 } }
                },
                y: {
                    stacked: true,
                    grid: { color: "rgba(255,255,255,0.05)" },
                    max: 1.0,
                    ticks: { color: "#8b9bb4", font: { size: 9 } }
                }
            }
        }
    });
}

// Spread actual vs predicted chart builder
function createSpreadChart(actual, predicted) {
    const ctx = document.getElementById("spreadChart").getContext("2d");
    if (spreadChartInstance) {
        spreadChartInstance.destroy();
    }
    
    // Zoom in on the last 120 ticks to show tracking precision
    const sliceCount = 120;
    const actualSlice = actual.slice(-sliceCount);
    const predictedSlice = predicted.slice(-sliceCount);
    
    const labels = Array.from({ length: actualSlice.length }, (_, i) => i);
    
    spreadChartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Actual LOB Spread",
                    data: actualSlice,
                    borderColor: "#ff007b",
                    borderWidth: 2,
                    pointRadius: 0.5,
                    fill: false,
                    tension: 0.1
                },
                {
                    label: "Sagan Pred Spread",
                    data: predictedSlice,
                    borderColor: "#00f0ff",
                    borderWidth: 1.5,
                    borderDash: [4, 4],
                    pointRadius: 0,
                    fill: false,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "top",
                    labels: { color: "#8b9bb4", font: { size: 9 } }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { display: false }
                },
                y: {
                    grid: { color: "rgba(255,255,255,0.05)" },
                    ticks: { color: "#8b9bb4", font: { size: 9 } }
                }
            }
        }
    });
}

// Trade execution logs list renderer
function renderTradeLog(logs) {
    const container = document.getElementById("tradeLogContainer");
    container.innerHTML = "";
    
    if (logs.length === 0) {
        container.innerHTML = `<div style="padding:2rem; text-align:center; color:var(--text-secondary); width:100%;">No trades executed during this out-of-sample sequence.</div>`;
        return;
    }
    
    // Show newest trades first
    const reversedLogs = [...logs].reverse();
    
    reversedLogs.forEach(log => {
        const row = document.createElement("div");
        row.className = "log-row";
        
        const isMaker = log.type.includes("MAKER");
        const typeBadge = isMaker 
            ? `<span class="maker-badge">Maker</span>` 
            : `<span class="taker-badge">Taker</span>`;
            
        const actionType = log.type.includes("BUY") 
            ? `<span style="color:#00ff88; font-weight:600;">BUY</span>` 
            : `<span style="color:#ff0055; font-weight:600;">SELL</span>`;
            
        const rebateText = log.rebate > 0 
            ? `<span class="text-green">+₹${log.rebate.toFixed(2)}</span>` 
            : `<span style="color:rgba(255,255,255,0.15)">-</span>`;
            
        const feeText = log.fee > 0 
            ? `<span class="text-red">-₹${log.fee.toFixed(2)}</span>` 
            : `<span style="color:rgba(255,255,255,0.15)">-</span>`;
            
        const slippageText = log.slippage > 0 
            ? `<span class="text-magenta">-₹${log.slippage.toFixed(2)}</span>` 
            : `<span style="color:rgba(255,255,255,0.15)">-</span>`;
            
        row.innerHTML = `
            <span style="color:var(--text-secondary);">${log.tick}</span>
            <span>${typeBadge} ${actionType}</span>
            <span style="color:#fff; font-weight:500;">₹${log.price.toFixed(2)}</span>
            <span>${log.size}</span>
            <span>${rebateText}</span>
            <span>${feeText}</span>
            <span>${slippageText}</span>
        `;
        
        container.appendChild(row);
    });
}
