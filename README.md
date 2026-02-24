# Intraday Volatility Forecasting for Optimal Liquidity Provision

Research project on volatility forecasting and market-making strategy optimization under microstructure noise and inventory risk, using TAQ tick data from WRDS.

## Project Overview

The pipeline implements:
- **Data Loading & Cleaning**: TAQ quote ingestion with professional microstructure noise filtering
- **Realized Volatility**: Tick-level RV estimation (Andersen & Bollerslev)
- **EWMA Forecasting**: One-step-ahead variance forecasts (λ=0.94, RiskMetrics)
- **Market-Making Engine**: Simulation of bid/ask quoting with fill, inventory, and P&L tracking
- **Strategy Comparison**: Constant-spread baseline vs. volatility-adaptive quoting

## Directory Structure

```
├── config.yaml                     # Configuration parameters
├── main.py                         # Pipeline entry point
├── requirements.txt                # Python dependencies
│
├── data/                           # Raw TAQ data (excluded from git)
│   └── t822bpd5q8g1deky.csv       # SPY quotes, 2024-01-03 (~24M raw rows)
│
├── src/
│   ├── data_loader.py             # TAQ loading, cleaning, RV computation
│   ├── volatility/
│   │   ├── ewma.py                # EWMA variance/vol forecasting
│   │   ├── har.py                 # HAR model (placeholder)
│   │   └── realized_vol.py        # Realized variance & rolling RV
│   ├── market_making/
│   │   ├── quoting.py             # Vol-adaptive spread computation
│   │   ├── inventory.py           # Inventory management (placeholder)
│   │   └── pnl.py                 # P&L utilities (placeholder)
│   ├── simulator/
│   │   ├── engine.py              # MarketMakerEngine — tick simulation
│   │   └── fill_model.py          # Fill model (placeholder)
│   └── evaluation/
│       ├── metrics.py             # Performance metrics (placeholder)
│       └── plots.py               # Visualization utilities (placeholder)
│
├── test/
│   ├── test_ewma.py               # EWMA unit tests
│   ├── test_realized_vol.py       # Realized vol unit tests
│   ├── test_engine.py             # Engine unit tests
│   ├── test_quoting.py            # Quoting unit tests
│   └── pipelines/
│       └── test_volatility_pipeline.py  # Integration test
│
├── notebooks/
│   └── exploratory.ipynb          # EDA
│
└── results/                        # Output plots and metrics
```

## Setup

```bash
pip install -r requirements.txt
```

Data file is excluded from the repository (`.gitignore`). Obtain TAQ quote data from WRDS and place it at `data/t822bpd5q8g1deky.csv`.

## Running the Pipeline

```bash
python main.py
```

This executes the full pipeline:

1. **Load & clean** TAQ tick data (~24M rows → ~6M after RTH filter + microstructure filters)
2. **Compute realized volatility** per 1-minute bar
3. **Fit EWMA** variance forecast (λ=0.94)
4. **Simulate baseline strategy** — constant half-spread δ=0.03
5. **Simulate vol-adaptive strategy** — δ = K0 + K1 × σ̂ (K0=0.01, K1=1.0)
6. **Print comparison**

### Sample Output (SPY, 2024-01-03)

```
[21:51:27] ✅ Loaded 5,987,412 tick rows | 09:30 → 12:34 (RTH)
[21:51:29] ✅ 185 1-min bars
[21:51:31] ✅ Mean RVol: 0.0153 | EWMA vol mean: 0.0183, max: 0.0476

======================================================================
METRIC                     BASELINE    VOL-ADAPTIVE
======================================================================
total_pnl                   -7.075       -4.285  (+39.4%)
mean_pnl_per_bar            -0.0385      -0.0233 (+39.4%)
std_pnl_per_bar              0.7391       0.4816 (-34.8%)
sharpe_ratio                -0.052       -0.048  (+6.9%)
inventory_variance           7.69         5.97   (-22.4%)
max_abs_inventory            12           10
n_trades                     144          147
======================================================================
```

**Key finding**: Volatility-adaptive quoting reduces losses by 39%, cuts PnL volatility by 35%, and lowers inventory variance by 22% versus the constant-spread baseline. The EWMA volatility signal is economically meaningful even on a single trading day.

## Running Tests

```bash
python test/test_ewma.py
python test/test_realized_vol.py
python test/test_engine.py
python test/test_quoting.py
python test/pipelines/test_volatility_pipeline.py
```

All 20 tests pass. Each file prints ✅/❌ results directly when run.

## Key Components

### Data Cleaning (`src/data_loader.py`)

`load_data(filepath)` applies the following filters in order:

| Filter | Purpose |
|---|---|
| RTH window `[09:30, 16:00)` | Exclude pre/post-market and closing auction |
| `ask > bid` | Remove crossed/locked markets |
| Spread ≤ 1% of mid | Remove outlier quotes and stale streams |
| Quote-stuffing removal | Drop consecutive identical bid/ask pairs |
| \|log return\| ≤ 1% | Remove outlier price jumps |

Outputs a `DatetimeIndex` DataFrame with: `bid, ask, bidsiz, asksiz, mid, log_return`.

### EWMA Forecasting (`src/volatility/ewma.py`)

$$\hat{\sigma}^2_{t+1} = \lambda \hat{\sigma}^2_t + (1 - \lambda) \, \text{RV}_t$$

`ewma_variance_forecast(realized_var, lam=0.94)` — produces one-step-ahead variance forecasts from 1-minute realized variance bars.

### Market-Making Engine (`src/simulator/engine.py`)

`MarketMakerEngine.run(df, delta)`:
- `delta` can be a scalar (constant spread) or `pd.Series` (time-varying)
- Fill logic: next mid ≤ bid → buyer hits, next mid ≥ ask → seller lifts
- Tracks: `inventory`, `cash`, `portfolio_value`, `trade_count` per bar

### Vol-Adaptive Quoting (`src/market_making/quoting.py`)

```python
delta = compute_spread(sigma_hat, k0=0.01, k1=1.0)  # δ = k0 + k1·σ̂
```

Wider spreads in high-volatility regimes reduce adverse selection and inventory exposure.

## Configuration

Key parameters in `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `DELTA` | 0.03 | Constant half-spread for baseline |
| `EWMA_LAM` | 0.94 | EWMA decay factor (RiskMetrics) |
| `K0` | 0.01 | Vol-adaptive spread intercept |
| `K1` | 1.0 | Vol-adaptive spread sensitivity |

## Progress

- ✅ TAQ data loading with professional microstructure cleaning
- ✅ Realized volatility estimation from tick returns
- ✅ EWMA variance forecasting
- ✅ Market-making simulation engine (scalar + series delta)
- ✅ Vol-adaptive quoting strategy
- ✅ Baseline vs. vol-adaptive comparison
- ✅ Full unit test suite (20 tests)
- 🔄 Inventory skew (quote adjustment proportional to position)
- 🔄 HAR model
- 🔄 Adverse selection filter
- 🔄 Multi-day analysis



## License

Private research project.
