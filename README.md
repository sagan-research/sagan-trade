# Sagan Trade

High Frequency Trading Engine.

- Update 0.1.0: Initial commit and package creation
- Update 0.1.1: Add type hints to core components
- Update 0.1.2: Improve documentation in models
- Update 0.1.3: Refactor simulator execution flow
- Update 0.1.4: Add detailed logging mechanisms
- Update 0.1.5: Finalise README and public API
- Update 0.2.0: Integrated Hawkes process for trade arrivals and Bates Jump-Diffusion (Heston + Merton Jumps) for mid-price dynamics simulation.
- **Update 0.8.2 (Latest):** Version bump to synchronize with wider AIN framework capabilities.

## Features
- **Hawkes Process**: Self-exciting order flow dynamics with microstructural tick generation.
- **Bates Jump-Diffusion**: Incorporates stochastic volatility (Heston) and jump diffusions (Merton) to correctly capture heavy tails and volatility clustering in tick-by-tick simulation.

## Quickstart
```python
from simulator import HawkesLOBSimulator
sim = HawkesLOBSimulator()
df = sim.simulate_ticks("RELIANCE", num_ticks=100)
print(df.head())
```