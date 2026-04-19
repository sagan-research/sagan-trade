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
        You are a mathematical symbolic regression expert. 
        Task: Suggest a mathematical formula to predict {target_variable} using the following variables:
        {', '.join(input_variables)}
        
        Rules:
        1. Use common operators (+, -, *, /, exp, log, sin, cos).
        2. The output must be a valid Python/NumPy expression using variable names from the list.
        3. Keep it relatively simple but effective for trend capturing.
        
        Example Output:
        (Close * 0.5) + (RSI / 100) * exp(Volume / 1e6)
        
        Suggest the formula for {target_variable}:
        """
        
        try:
            response = self.client.generate(model=self.model, prompt=prompt)
            # The model might return extra text, we try to extract the expression
            # For a production 'functioncalling' model, we'd use its native tool format.
            return response['response'].strip()
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            # Fallback to a simple linear combination if AI fails
            return " + ".join(input_variables)

    def optimize_discovered_function(self, formula: str, data: Dict[str, Any]) -> str:
        """
        Refines the formula based on statistical feedback (placeholder for active learning).
        """
        # This could be another call to Gemma to refine parameters if R2 is low
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
