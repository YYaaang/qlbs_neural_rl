# GLBS-RL

This project implements a discrete-time, risk-sensitive option pricing and hedging framework based on QLBS and reinforcement learning methods.  
It focuses on optimal hedging under residual risk and supports both Black–Scholes and Heston market models.

The codebase is designed for research and experimentation on risk-sensitive pricing, hedging strategies, and numerical analysis.

---

## Project Structure

├── analysis
├── configs
│   ├── base
│   │   ├── __init__.py
│   │   ├── env_cfg.py
│   │   ├── policy_cfg.py
│   │   └── runtime_cfg.py
│   ├── specifications
│   │   ├── __init__.py
│   │   ├── actor_cfg.py
│   │   ├── critic_cfg.py
│   │   ├── market_bs_cfg.py
│   │   └── market_heston_cfg.py
│   ├── __init__.py
│   ├── full_bs_config.py
│   └── full_heston_config.py
├── md
├── models
├── src
│   ├── bs_model.py
│   ├── buffer.py
│   ├── data_buffer_processing.py
│   ├── data_processing.py
│   ├── heston_model.py
│   ├── options.py
│   ├── qlbs.py
│   ├── rl_models.py
│   ├── trading_env.py
│   ├── train_state.py
│   └── trainer.py
├── test
│   └── states.py
├── utils
│   ├── analysis
│   │   ├── __init__.py
│   │   ├── delta_pnl_analysis.py
│   │   ├── evaluation.py
│   │   ├── implied_vol.py
│   │   ├── pnl_analysis.py
│   │   ├── smile_plots.py
│   │   └── transaction_cost.py
│   ├── pricing
│   │   ├── __init__.py
│   │   ├── fft_heston_model.py
│   │   ├── mc_heston_model.py
│   │   ├── pvv_bs_model.py
│   │   └── quantlib_heston_model.py
│   ├── __init__.py
│   ├── log_print.py
│   ├── model_saver.py
│   └── train_one_step.py
├── add_new_critic_models.py
├── init_train_models.py
├── main.py

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

- **test/**  
  Simple tests and state validation.

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

---

## Notes

This project is intended for academic and research purposes.  
It emphasizes clarity, modularity, and reproducibility rather than production deployment.
