import logging
from typing import Dict, Any
from sagan.models.llm_bridge import FunctionGemmaBridge

logger = logging.getLogger("sagan.nlp")

class SaganInterpreter:
    """
    Translates natural language inputs into structured Sagan tasks.
    """
    def __init__(self, bridge: FunctionGemmaBridge = None):
        self.llm = bridge or FunctionGemmaBridge()

    def interpret(self, query: str) -> Dict[str, Any]:
        """
        Parses the query and returns a standardized task object.
        """
        logger.info(f"Interpreting query: {query}")
        intent = self.llm.parse_intent(query)
        
        # Add defaults if missing
        if "task" not in intent:
            intent["task"] = "unknown"
        if "tickers" not in intent:
            intent["tickers"] = []
            
        return intent

class CopilotOrchestrator:
    """
    Executes tasks based on interpreted intent.
    """
    def __init__(self):
        from sagan.autonomous import AutonomousResearcher
        self.researcher = AutonomousResearcher()
        self.interpreter = SaganInterpreter()

    def execute_query(self, query: str) -> Dict[str, Any]:
        task = self.interpreter.interpret(query)
        
        if task["task"] == "research" or task["task"] == "train":
            if not task["tickers"]:
                return {"status": "error", "message": "No tickers specified in request."}
            
            ticker = task["tickers"][0]
            # If multiple, we might loop, but for now let's handle one
            results = self.researcher.run_full_pipeline(ticker)
            return results
        
        elif task["task"] == "rebalance":
            # This would trigger the rebalancer logic
            from sagan.portfolio.rebalancer import PortfolioRebalancer
            rebalancer = PortfolioRebalancer()
            holdings = task.get("holdings", {})
            if not holdings:
                return {"status": "error", "message": "Please specify your current holdings for rebalancing."}
            
            plan = rebalancer.generate_rebalance_plan(holdings)
            return {"task": "rebalance", "plan": plan, "status": "success"}

        elif task["task"] == "list":
            import sagan
            models = sagan.list_models()
            return {"task": "list", "models": models.to_dict(orient="records"), "status": "success"}

        return {"status": "error", "message": f"Task '{task['task']}' is not yet supported via NLP."}
