import json
from pathlib import Path
from datetime import datetime
from sagan.config import config

COMPLIANCE_DIR = config.home_dir / "compliance"

def generate_compliance_report(res: dict, algo_id: str | None = None) -> tuple[Path, Path]:
    """Generate .md and .json regulatory reports for a prediction result."""
    COMPLIANCE_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    algo_id = algo_id or f"ALGO_{timestamp_str}"
    
    md_path = COMPLIANCE_DIR / f"{algo_id}.md"
    json_path = COMPLIANCE_DIR / f"{algo_id}.json"
    
    # JSON Report
    report_data = {
        "algo_id": algo_id,
        "prediction": res,
        "regulatory_metadata": {
            "sebi_compliance_framework": "v2026.1",
            "model_type": "PINN-TFT Hybrid Ensemble",
            "audit_trail_active": True
        }
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)
        
    # Markdown Report
    xai = res["xai_justification"]
    conflict_status = "⚠️ CONFLICT DETECTED" if xai.get("conflict") else "✅ SIGNALS ALIGNED"
    
    md_content = f"""# SEBI Compliance Report - {algo_id}
**Timestamp:** {res['timestamp']}
**Model ID:** {res['model_id']}
**Tickers:** {", ".join(res['tickers'])}

## 1. Executive Summary
- **Final Action:** {res['signal']}
- **Confidence:** {res['confidence']:.2%}
- **Compliance Status:** {conflict_status}

## 2. Technical Justification
{xai['reason']}

### Indicator Snapshot
"""
    for ticker, tech in xai.get("technical_indicators", {}).items():
        md_content += f"- **{ticker}**: RSI={tech['rsi']:.1f}, BB_Middle={tech['bb_middle']:.2f}, Price={tech['price']:.2f}\n"

    md_content += f"""
### Model-Generated Thresholds
"""
    for ticker, thresh in xai.get("thresholds", {}).items():
        md_content += f"- **{ticker}**: RSI_Buy={thresh['rsi_buy']}, RSI_Sell={thresh['rsi_sell']}\n"

    md_content += f"""
## 3. Regulatory Disclosure
This algorithmic trade was generated using a Physics-Informed Neural Network (PINN) combined with a Temporal Fusion Transformer (TFT). 
The model-generated thresholds are derived from ticker-specific historical volatility and mean-reversion characteristics.

**Algo ID:** {algo_id}
**Audit Trail:** Logged in local SQLite database.
"""
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    return md_path, json_path
