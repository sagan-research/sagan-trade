import ollama
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger("sagan.llm")

class FunctionGemmaBridge:
    """
    Bridge to FunctionGemma (via Ollama) for symbolic strategy discovery.
    """
    
    def __init__(self, model: str = "functiongemma", host: str = "http://localhost:11434"):
        self.model = model
        self.client = ollama.Client(host=host)

    def suggest_composite_function(self, target_variable: str, input_variables: List[str]) -> str:
        """
        Asks FunctionGemma to suggest a mathematical expression to predict 
        the target_variable using a combination of input_variables.
        """
        candidates = self.suggest_candidates(target_variable, input_variables, count=1)
        return candidates[0] if candidates else " + ".join(input_variables)

    def suggest_candidates(self, target_variable: str, input_variables: List[str], count: int = 5) -> List[str]:
        """
        Asks FunctionGemma to suggest multiple candidate mathematical expressions.
        """
        prompt = f"""
        [INST] <<SYS>>
        You are a symbolic regression engine. Your output MUST ONLY be a list of {count} valid Python/NumPy mathematical expressions, one per line. 
        Focus on nonlinear interactions like multiplication, division, log, and exp.
        Do not provide explanations. Do not provide markdown.
        <</SYS>>

        Task: Return {count} candidate mathematical formulas to predict {target_variable} using: {', '.join(input_variables)}
        
        Requirements:
        1. Mix variables nonlinearly: (A * B), (A / B), np.log(A) * B, etc.
        2. Output MUST be exactly {count} lines.
        3. Use ONLY standard math and NumPy (np.exp, np.log, np.sin, np.cos).

        Candidates for {target_variable}: [/INST]"""
        
        try:
            response = self.client.generate(model=self.model, prompt=prompt)
            raw = response['response'].strip()
            
            lines = [line.strip() for line in raw.split("\n") if line.strip()]
            
            refusal_phrases = ["cannot assist", "i am sorry", "legal", "financial advice", "disclaimer", "as an ai"]
            
            cleaned = []
            for line in lines:
                c = line.replace("```python", "").replace("```", "").strip()
                # Filter out refusals
                if any(phrase in c.lower() for phrase in refusal_phrases):
                    continue
                # Ensure it looks like a formula (contains at least one variable)
                if not any(v in c for v in input_variables):
                    continue
                    
                # Remove leading numbers/bullets
                if ". " in c[:4]: c = c.split(". ", 1)[1]
                if "- " in c[:3]: c = c.split("- ", 1)[1]
                cleaned.append(c)
                
            return cleaned[:count]
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return [" + ".join(input_variables)]

    def optimize_discovered_function(self, formula: str, data: Dict[str, Any]) -> str:
        """
        Refines the formula based on statistical feedback.
        """
        return formula

    def suggest_relevant_signals(self, ticker: str) -> List[str]:
        """
        Asks FunctionGemma to suggest relevant yfinance-compatible signals for a ticker.
        """
        prompt = f"""
        [INST] <<SYS>>
        You are a quantitative data architect. Your output MUST ONLY be a JSON list of strings.
        Each string must be a yfinance-compatible ticker or indicator name.
        Do not provide explanations. Do not provide markdown.
        <</SYS>>

        Task: Return a list of 5-8 relevant signals to predict the price of {ticker}.
        Include:
        1. Macro indicators (e.g., ^VIX, ^TNX, DX-Y.NYB)
        2. Related sector tickers (e.g., SOXX for NVDA, XLK for MSFT)
        3. Technical indicator names (RSI, SMA_20, MACD)
        
        Signals for {ticker}: [/INST]"""
        
        try:
            response = self.client.generate(model=self.model, prompt=prompt)
            raw = response['response'].strip().replace("```json", "").replace("```", "").strip()
            
            # Remove any potential text before/after the list
            if "[" in raw and "]" in raw:
                raw = raw[raw.find("["):raw.rfind("]")+1]
            
            try:
                signals = json.loads(raw)
                if isinstance(signals, list):
                    return [str(s) for s in signals if s]
            except:
                pass

            # Fallback parsing for non-json
            lines = [l.strip().strip('"').strip("'").strip("-").strip("*").strip() 
                     for l in raw.split("\n") if l.strip()]
            
            # Filter out likely conversational lines
            signals = [l for l in lines if len(l) < 20 and not any(kw in l.lower() for kw in ["here", "list", "json", "predict"])]
            
            if not signals:
                return ["Adj Close", "Volume", "RSI", "SMA_20", "^VIX"]
            return signals[:8]
        except Exception as e:
            logger.error(f"Signal discovery failed: {e}")
            return ["Adj Close", "Volume", "RSI", "SMA_20", "^VIX"]

    def parse_intent(self, text: str) -> Dict[str, Any]:
        """
        Parses natural language input into a structured task for the Sagan engine.
        """
        prompt = f"""
        [INST] <<SYS>>
        You are the Sagan Copilot Intent Parser. Your output MUST ONLY be a JSON object.
        Supported Tasks: 'research', 'train', 'rebalance', 'predict', 'list'.
        Do not provide explanations. Do not provide markdown.
        <</SYS>>

        Task: Parse this user request: "{text}"
        
        JSON Schema:
        {{
            "task": "research" | "train" | "rebalance" | "predict" | "list",
            "tickers": ["AAPL", ...],
            "params": {{ "period": "2y", "formula": "..." }},
            "holdings": {{ "AAPL": 1000.0, ... }} // if mentioned
        }}

        Intent JSON: [/INST]"""
        
        try:
            response = self.client.generate(model=self.model, prompt=prompt)
            raw = response['response'].strip().replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except Exception as e:
            logger.error(f"Intent parsing failed: {e}")
            return {"task": "unknown", "text": text}

# Define tools for FunctionGemma (Schema)
TOOLS = [
    {
        "name": "polynomial_fit",
        "description": "Fits a polynomial of degree N to the data.",
        "parameters": {
            "type": "object",
            "properties": {
                "degree": {"type": "integer", "description": "Degree of polynomial"}
            }
        }
    },
    {
        "name": "fourier_fit",
        "description": "Fits a fourier series with N harmonics.",
        "parameters": {
            "type": "object",
            "properties": {
                "harmonics": {"type": "integer", "description": "Number of harmonics"}
            }
        }
    }
]
