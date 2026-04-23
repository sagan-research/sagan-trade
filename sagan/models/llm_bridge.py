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
        prompt = f"""
        [INST] <<SYS>>
        You are a symbolic regression engine. Your output MUST ONLY be a single line containing a valid Python/NumPy mathematical expression. 
        Do not provide explanations. Do not provide disclaimers. Do not provide advice.
        <</SYS>>

        Task: Return a mathematical formula for {target_variable} using these variables: {', '.join(input_variables)}
        
        Requirements:
        1. Use arithmetic (+, -, *, /) and NumPy functions (np.exp, np.log, np.sin).
        2. Output MUST be a single line.
        3. No text or markdown around the formula.
        
        Examples: 
        (Adj_Close * 0.5) + (Volume / 1e6)
        np.exp(Adj_Close / 100) * np.sin(Volume)

        Formula for {target_variable}: [/INST]"""
        
        try:
            response = self.client.generate(model=self.model, prompt=prompt)
            raw = response['response'].strip()
            
            # Basic cleanup in case the model ignored instructions
            lines = [line.strip() for line in raw.split("\n") if line.strip() and ("(" in line or "np." in line or any(v in line for v in input_variables))]
            if not lines:
                return " + ".join(input_variables) # Minimal fallback
            
            # Take the longest line that looks like a formula
            formula = max(lines, key=len)
            return formula.replace("```python", "").replace("```", "").strip()
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            # Fallback to a simple linear combination if AI fails
            return " + ".join(input_variables)

    def optimize_discovered_function(self, formula: str, data: Dict[str, Any]) -> str:
        """
        Refines the formula based on statistical feedback (placeholder for active learning).
        """
        return formula

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
