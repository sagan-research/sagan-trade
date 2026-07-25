import logging


class MockLangGraphNode:
    """Base class for mocking LangGraph Nodes."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(self.name)


class SentimentAnalyzerNode(MockLangGraphNode):
    """Analyzes text to extract bullish/bearish market sentiment."""

    def __init__(self):
        super().__init__("SentimentAnalyzer")

    def analyze(self, news_text: str) -> float:
        self.logger.info(f"Analyzing sentiment for news: '{news_text[:30]}...'")
        # In a real LangGraph setup, this would invoke an LLM.
        # Here we mock sentiment from -1.0 (Bearish) to 1.0 (Bullish).
        words = news_text.lower().split()
        bull_words = ["surge", "growth", "up", "bull", "profit", "expansion"]
        bear_words = ["crash", "drop", "down", "bear", "loss", "inflation"]

        score = 0.0
        for w in words:
            if w in bull_words:
                score += 0.2
            if w in bear_words:
                score -= 0.2

        return max(min(score, 1.0), -1.0)


class MacroEconomistNode(MockLangGraphNode):
    """Evaluates systemic risk and macro economic conditions."""

    def __init__(self):
        super().__init__("MacroEconomist")

    def evaluate_volatility(self, macro_data: dict) -> float:
        self.logger.info("Evaluating Macro Volatility Modifiers...")
        vix = macro_data.get("VIX", 15.0)
        # Yields a volatility multiplier (e.g. 1.0 = Normal, 2.0 = High Vol)
        return max(1.0, vix / 15.0)


class RiskManagerNode(MockLangGraphNode):
    """Final node that synthesizes sentiment and volatility into actionable modifiers."""

    def __init__(self):
        super().__init__("RiskManager")

    def synthesize(self, sentiment: float, vol_modifier: float):
        self.logger.info(
            f"Synthesizing risk state (Sentiment: {sentiment:.2f}, Vol: {vol_modifier:.2f})"
        )
        # Scale positions based on conviction (sentiment) and risk (vol_modifier)
        position_sizing_multiplier = abs(sentiment) / vol_modifier
        direction = "LONG" if sentiment > 0 else "SHORT" if sentiment < 0 else "NEUTRAL"

        return {
            "direction": direction,
            "sizing_multiplier": position_sizing_multiplier,
            "systemic_volatility": vol_modifier,
        }


class AgenticEnsemble:
    """
    Mock representation of a LangGraph multi-agent execution pipeline.
    """

    def __init__(self):
        self.sentiment_agent = SentimentAnalyzerNode()
        self.macro_agent = MacroEconomistNode()
        self.risk_agent = RiskManagerNode()
        logging.basicConfig(level=logging.INFO)

    def process_market_state(self, news_text: str, macro_data: dict):
        """Executes the pipeline."""
        sentiment_score = self.sentiment_agent.analyze(news_text)
        vol_modifier = self.macro_agent.evaluate_volatility(macro_data)

        final_decision = self.risk_agent.synthesize(sentiment_score, vol_modifier)
        return final_decision
