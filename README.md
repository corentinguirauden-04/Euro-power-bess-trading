# euro-power-bess-trading

A European power market simulator (15-minute resolution) coupled with a physical Battery Energy Storage System (BESS) model to benchmark manual/heuristic arbitrage against quantitative Mixed-Integer Linear Programming (MILP) optimization.

## Project Architecture

```text
euro-power-bess-trading/
├── src/
│   ├── __init__.py
│   ├── market_sim.py      # 15-min European Spot price simulator (Merit Order, Duck Curve, OU wind, shocks)
│   ├── asset.py           # Physical BESS model (power limits, SoC bounds, efficiency, degradation)
│   ├── trading_env.py     # Discrete-time trading engine (order execution, cashflow & PnL ledger)
│   └── quant_models.py    # [Stage 2] MILP perfect-foresight & rolling-horizon MPC optimizers
├── scripts/
│   └── demo_stage1.py     # Stage 1 validation, naive quantile benchmark & 3-panel visualization
├── app.py                 # [Stage 3] Interactive Streamlit trading dashboard
├── requirements.txt
└── README.md
```

## Stage 1 — Market Fundamentals & BESS Physics (Implemented)

### 1. European Power Market Simulator (`src/market_sim.py`)
`PowerMarketSimulator` generates synthetic 15-minute market fundamentals (`dt = 0.25` h, 96 intervals/day):
- **Demand Profile**: Double-peaked Continental European load curve (morning & evening peaks, weekday/weekend modulation, AR(1) forecast noise).
- **Solar PV (Duck Curve)**: Diurnal clear-sky profile with day-to-day irradiance variability and AR(1) cloud transients.
- **Wind Generation**: Mean-reverting Ornstein-Uhlenbeck (OU) process bounded within physical capacity.
- **Merit Order Stack & Price Regimes**: Piecewise convex supply curve mapping residual net load (+ occasional unplanned outage shocks) through must-run/nuclear baseload, mid-merit thermal, marginal CCGT gas, OCGT peakers, and scarcity pricing—producing both **negative prices** during renewable oversupply and **price spikes** during tight evening peaks.

### 2. Physical BESS Asset (`src/asset.py`)
`BatteryAsset` models a grid-scale battery (default: 10 MWh / 5 MW, 92% one-way charge/discharge efficiency, 10%–90% SoC operating window, 2.0 €/MWh cycling degradation):
- Consistent grid-side sign convention: `+MW` = discharge (sell to grid), `-MW` = charge (buy from grid).
- `step(power_mw, dt)` enforces inverter clipping, hard SoC constraints, and asymmetric efficiency losses, returning `(energy_exchanged_mwh, new_soc)`.

### 3. Trading Simulation Engine (`src/trading_env.py`)
`TradingEnv` steps chronologically through the market horizon, dispatches orders to the battery, settles interval cashflows (including revenues from charging during negative-price intervals), deducts cycling degradation costs, and tracks cumulative PnL and operational KPIs.

## Quick Start

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python scripts/demo_stage1.py
```

To run headlessly and save the diagnostic chart without opening a GUI window:

```bash
python scripts/demo_stage1.py --days 3 --seed 42 --no-show
```

## Roadmap

- [x] **Stage 1**: 15-minute Merit Order market simulator, BESS physical asset, trading engine, and naive quantile arbitrage demo.
- [ ] **Stage 2**: Quantitative dispatch optimization (`src/quant_models.py`) — MILP perfect-foresight benchmark and rolling-horizon MPC.
- [ ] **Stage 3**: Interactive Streamlit application (`app.py`) for manual vs. quantitative strategy comparison.
