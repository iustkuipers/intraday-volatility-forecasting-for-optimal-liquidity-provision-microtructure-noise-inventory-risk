# Intraday Volatility Forecasting for Optimal Liquidity Provision
### Under Microstructure Noise and Inventory Risk

An event-driven market-making simulator for SPY built on TAQ (Trade and Quote) data. The system replays a full day of consolidated quotes and trade prints tick-by-tick, applies a pluggable quoting strategy with inventory skew and realized-volatility adaptation, and produces a detailed performance record including PnL, fill rate, adverse selection, and drawdown metrics.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Data Pipeline](#data-pipeline)
- [Simulator Components](#simulator-components)
- [Strategies](#strategies)
- [Metrics](#metrics)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Test Suite](#test-suite)
- [Results](#results)

---

## Overview

The project implements an **intraday market-making simulation** for a single equity (SPY) with the following research objectives:

1. **Optimal spread quoting** under microstructure noise — separating genuine price discovery from transient quote flickering.
2. **Inventory risk management** — skewing quotes away from the current position to mean-revert inventory without sacrificing fill rate.
3. **Volatility-adaptive quoting** — widening spreads when realized volatility rises to protect against adverse selection.

The baseline strategy (`ConstantSpreadStrategy`) implements the Avellaneda–Stoikov-inspired reservation-price framework:

$$r = m - \gamma \cdot \sigma \cdot m \cdot q$$

$$s^{bid} = r - \delta, \quad s^{ask} = r + \delta$$

where $m$ is the mid price, $\gamma$ is risk aversion, $\sigma$ is realized volatility, $q$ is inventory, and $\delta$ is the half-spread.

---

## Architecture

```
Events (quotes + trades)
        │
        ▼
  MarketState          ← one snapshot per event row
        │
        ▼
  Strategy             ← computes two-sided Quote (bid/ask prices + sizes)
        │
        ▼
  ExecutionEngine      ← holds resting orders, enforces hard position cap
        │
        ▼
  FillModel            ← determines if a resting order was hit by a trade
        │
        ▼
  Accounting           ← updates inventory, cash, realized PnL (FIFO avg-cost)
        │
        ▼
  Snapshot row         → results DataFrame
```

The `Simulator` is a pure orchestrator — it wires the components together and iterates over the event stream. All domain logic lives in the individual components, making each one independently testable and swappable.

---

## Project Structure

```
.
├── main.py                         # Entry point — configure, run, report
│
├── data/
│   ├── raw/                        # Original TAQ CSV exports
│   ├── clean/                      # Cleaned quotes.parquet & trades.parquet
│   └── merged/                     # Unified events.parquet (quotes + trades)
│
├── data_preparation/
│   ├── data_load_bid_ask.py        # Clean consolidated quotes → parquet
│   ├── data_load_trades.py         # Clean TAQ trades → parquet
│   ├── data_merged.py              # Merge quotes+trades into event stream
│   └── data_analysis.py           # Exploratory statistics
│
├── simulator/
│   ├── simulator.py                # Orchestrator (event loop)
│   ├── market_state.py             # Parses event row → attribute snapshot
│   ├── strategy_base.py            # BaseStrategy interface
│   ├── strategy.py                 # ConstantSpreadStrategy, VolatilityAdaptiveStrategy
│   ├── execution_engine.py         # Manages resting orders + position cap
│   ├── fill_model.py               # Fill logic (DeterministicFillModel)
│   ├── accounting.py               # Inventory, cash, PnL ledger (FIFO)
│   ├── order.py                    # Quote / Fill data types
│   ├── metrics.py                  # Post-run performance statistics
│   └── output.py                   # Saves results + metrics to timestamped folder
│
├── volatility/                     # (in development) volatility forecasting models
│
├── tests/
│   ├── test_accounting.py
│   ├── test_execution_engine.py
│   ├── test_fill_model.py
│   ├── test_market_state.py
│   ├── test_metrics.py
│   ├── test_order.py
│   ├── test_simulator.py
│   └── test_strategy.py
│
└── results/                        # Auto-saved simulation outputs (git-ignored)
    └── <timestamp>_<label>/
        ├── results.csv
        └── metrics.json
```

---

## Data Pipeline

Run the preparation scripts **once** in order before the first simulation:

```bash
# 1. Clean raw quote data
python data_preparation/data_load_bid_ask.py

# 2. Clean raw trade data
python data_preparation/data_load_trades.py

# 3. Merge into unified event stream
python data_preparation/data_merged.py
```

### Quote cleaning (`data_load_bid_ask.py`)
- Filters to regular trading hours (09:30 – 16:00)
- Removes crossed or zero-width quotes
- Deduplicates same-timestamp quotes
- Drops stale exchange snapshots with spread > 0.2% of mid

### Trade cleaning (`data_load_trades.py`)
- RTH filter
- Removes zero-price / zero-size prints
- Applies Lee–Ready direction inference

### Event stream (`data_merged.py`)
- Enriches each trade with its prevailing quote state via backward `merge_asof`
- Computes rolling realized volatility (20-event window)
- Computes order-flow imbalance
- Heals residual NBBO spikes (spread > 10× rolling median) by forward-filling
- Saves `events.parquet` (~22M rows for a full day) and `events_5min.parquet`

---

## Simulator Components

| Component | Responsibility |
|---|---|
| `MarketState` | Converts a DataFrame row into `.mid`, `.bid`, `.ask`, `.last_trade_price`, `.realized_vol`, etc. |
| `BaseStrategy` | Interface: `compute_quote(state, inventory) → Quote` |
| `ExecutionEngine` | Holds the current resting `Quote`; enforces hard position cap via `max_position_value` |
| `BaseFillModel` | Interface: `evaluate(engine, state) → list[Fill]` |
| `DeterministicFillModel` | Fills whenever a trade crosses the resting quote price |
| `Accounting` | FIFO avg-cost ledger; tracks `inventory`, `cash`, `realized_pnl`, `portfolio_value` |
| `Metrics` | Computes Sharpe, max drawdown, fill rate, spread capture, adverse selection |
| `OutputManager` | Saves `results.csv` and `metrics.json` under `results/<timestamp>_<label>/` |

---

## Strategies

### `ConstantSpreadStrategy`

Baseline strategy. Parameters are **dimensionless** and scale automatically with price and volatility:

| Parameter | Description | SPY example |
|---|---|---|
| `spread_frac` | Half-spread as fraction of mid | `0.000011` → ~$0.005 at $472 (NBBO touch) |
| `risk_aversion` | Inventory skew coefficient γ | `0.00005` → ~45% of half-spread at 100 shares |
| `max_position_value` | Hard inventory cap in dollars | `500_000` → ~1059 shares at $472 |
| `vol_cap` | Ceiling on realized vol input | `0.005` → prevents bad ticks from blowing out skew |
| `order_size` | Shares per quote | `100` |

### `VolatilityAdaptiveStrategy`

Extends `ConstantSpreadStrategy` by adding a volatility-proportional spread premium:

$$\delta = \delta_0 + \text{vol\_spread\_coef} \times \sigma$$

| Additional Parameter | Description |
|---|---|
| `vol_spread_coef` | Extra spread per unit of realized vol |

---

## Metrics

Computed by `Metrics.compute(results_df)` after each run:

| Metric | Description |
|---|---|
| `total_pnl` | Final portfolio value (unrealized + realized) |
| `realized_pnl` | Profit locked in by closed round-trips |
| `sharpe` | Mean(ΔPortfolio) / Std(ΔPortfolio) |
| `max_drawdown` | Largest peak-to-trough decline |
| `inventory_std` | Standard deviation of inventory |
| `max_inventory` | Peak absolute inventory |
| `fill_rate` | Fraction of events where a fill occurred |
| `n_fills` | Total fills |
| `spread_capture` | `realized_pnl / n_round_trips` — avg profit per round trip |
| `n_round_trips` | Number of times inventory returned to zero |

The main script also reports **adverse selection** (1-second mid-price move after each fill) and **quote sanity** (fraction of events where `bid_quote > mid` or `ask_quote < mid`, which must always be 0%).

---

## Setup

**Requirements:** Python 3.11+

```bash
pip install pandas numpy pyarrow
```

No external market-data subscriptions are needed — the raw TAQ CSV files are already included under `data/raw/`.

---

## Usage

```bash
# Full pipeline (first time only)
python data_preparation/data_load_bid_ask.py
python data_preparation/data_load_trades.py
python data_preparation/data_merged.py

# Run simulation
python main.py
```

Output is printed to the terminal and saved automatically:

```
Loading events ...
Window : 2026-03-13 09:30:00  →  2026-03-13 09:35:00  (12,345 rows)

=======================================================
  Events          : 12,345
  Total fills     : 87  (bid 44 / ask 43)
  Fill rate       : 0.7049%
  Final inventory : 100
  Realized PnL    : 12.3400
  Portfolio value : 59.2100
=======================================================

Metrics:
  total_pnl            59.21
  realized_pnl         12.34
  sharpe               1.83
  ...

Saved → results/20260313_224818_SPY_NBBOtouch/
```

---

## Configuration

Edit the `CONFIG` dict at the top of `main.py`:

```python
CONFIG = dict(
    spread_frac        = 0.000011,   # NBBO-touch: ~1 bp at SPY
    order_size         = 100,        # shares per quote
    risk_aversion      = 0.00005,    # inventory skew strength
    max_position_value = 500_000,    # hard cap: ~1059 shares at $472
    vol_cap            = 0.005,      # clamp realized vol input
)
```

To run over the full trading day instead of the first 5 minutes, replace the windowing slice:

```python
# current (5-minute window)
t_end   = t_start + pd.Timedelta(minutes=5)
events  = events_full[events_full["timestamp"] < t_end].copy()

# full day
events  = events_full.copy()
```

---

## Test Suite

```bash
python -m pytest tests/ -v
```

Each simulator component has a dedicated unit-test module covering edge cases (zero inventory, position cap breach, crossed quotes, etc.).

---

## Results

Saved runs are stored under `results/` as:

```
results/
└── <YYYYMMDD_HHMMSS>_<label>/
    ├── results.csv     ← full snapshot time series
    └── metrics.json    ← summary statistics
```

The `results/troubleshooting/` sub-folder contains earlier calibration runs used during parameter tuning.
