# Regime-Based ETF Allocation

This project analyzes whether AI-managed ETFs provide diversification
benefits during different market regimes (calm, stressed, crisis).

## Key Features
- Regime detection using macro indicators (VIX, yield curve, credit stress)
- Comparison of AI vs traditional ETFs
- Regime-based dynamic allocation strategy
- Backtesting vs static benchmark

## Data
- Daily ETF returns (AI and traditional)
- Market indicators from Yahoo Finance

## Methodology
- K-Means clustering for regime detection
- Sortino ratio and CVaR for risk evaluation
- No-lookahead backtesting

## How to Run
```bash
python regime_allocation.py

