# GLBS-NEURAL_RL

This project implements a discrete-time, risk-sensitive option pricing and hedging framework based on QLBS and reinforcement learning methods.  
It focuses on optimal hedging under residual risk and supports both Black–Scholes and Heston market models.

The codebase is designed for research and experimentation on risk-sensitive pricing, hedging strategies, and numerical analysis.

---

## Folder Overview

- **analysis/**  
  Scripts for numerical experiments and result analysis.

- **configs/**  
  Configuration files for environments, markets, actors, critics, and training settings.

- **src/**  
  Core implementation, including market models, QLBS formulation, trading environment, and training logic.

- **utils/**  
  Utility functions for pricing benchmarks, P&L analysis, implied volatility, logging, and model saving.

- **models/**  
  Saved or pre-trained models.
  
- **md/**  
  Documentation and notes related to the project.

---

## Entry Points

- **main.py**  
  Main entry point for running training or experiments.

- **init_train_models.py**  
  Initialize models and training setup.

- **add_new_critic_models.py**  
  Utilities for extending or adding critic models.
