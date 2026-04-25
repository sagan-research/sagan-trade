# Whitepaper: The Sagan Framework for Post-Prediction Explainability in Finance

## 1. Introduction: The Transparency Crisis in Quantitative Finance
Modern quantitative finance is dominated by "Black Box" models—Deep Neural Networks and Gradient Boosted Trees that offer high predictive accuracy but zero structural transparency. In institutional settings, this lack of explainability leads to "Model Risk," regulatory friction, and difficulty in diagnosing failure modes.

The **Sagan Framework** introduces a paradigm shift: **Post-Prediction Explainability (XAI)** through Symbolic Foundation Models.

## 2. Core Principle: Post-Prediction vs. Real-Time Explainability

### 2.1 Real-Time Explainability (The "Glass Box" Trap)
Traditional "Glass Box" models (like Linear Regression) are explainable in real-time but often too simple to capture financial nonlinearities. Engineers are forced to choose between **Accuracy** and **Explainability**.

### 2.2 Post-Prediction Explainability (The Sagan Approach)
Sagan operates on the principle that the model should first optimize for the most robust mathematical fit (the Alpha), and *then* provide a human-readable justification. By using **Symbolic Regression** (Schmidt & Lipson, 2009), the "model" is not a weight matrix, but a **mathematical formula** (e.g., $Close \times \log(Volume)$). 

## 3. Technical Workings of the Sagan Engine

### 3.1 Step 1: Symbolic Basis Discovery
The `MathematicalEngine` searches for combinations of **Basis Functions** using principles of genetic programming (Koza, 1992):
- **Polynomial Kernels**: For local trend captures.
- **Fourier Kernels**: For cyclicalities and seasonality.
- **Transcendental Kernels**: For exponential growth and logarithmic decay.

### 3.2 Step 2: Alpha Desk Coordination
Independent formulas are coordinated via the `AlphaDesk` which applies rolling Z-score thresholds and portfolio-level risk management (Market Neutrality, Long-Only).

### 3.3 Step 3: XAI Justification (The LLM Bridge)
Justification is provided by **FunctionGemma**, a proprietary orchestration of Google’s open-weights **Gemma** models. The LLM take the symbolic formula and current market data to generate a narrative audit trail for every signal.

## 4. Academic Foundations
Sagan is built upon decades of research in symbolic discovery and evolutionary systems:
- **Koza, J. R. (1992)**. *Genetic Programming*.
- **Schmidt, M., & Lipson, H. (2009)**. "Distilling Free-Form Natural Laws from Experimental Data". *Science*.

## 5. Conclusion
The Sagan Framework proves that transparency does not require a sacrifice in performance. By combining symbolic foundations with LLM-driven post-facto explanation, we create a system that is mathematically rigorous and humanly intuitive.

---
*Author: Antigravity AI*
*Framework: sagan-trade v0.3.5*
