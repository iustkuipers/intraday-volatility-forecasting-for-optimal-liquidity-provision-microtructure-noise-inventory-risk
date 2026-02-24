# Intraday Volatility Forecasting for Optimal Liquidity Provision

A quantitative research project for volatility forecasting and market-making optimization under microstructure noise and inventory risk.

## Project Overview

This project implements:
- **Data Loading & Cleaning**: Robust TAQ quote data ingestion with microstructure noise filtering
- **Volatility Models**: EWMA, HAR, and Realized Volatility estimators
- **Market-Making Engine**: Inventory-aware quoting with optimal spread calculation
- **Simulation & Backtesting**: Full order fill model and P&L evaluation

## Directory Structure

```
├── config.yaml                 # Configuration parameters
├── main.py                    # Entry point
├── requirements.txt           # Python dependencies
│
├── data/                      # Raw TAQ data
│   └── t822bpd5q8g1deky.csv  # Sample quote data (SPY, 2024-01-03)
│
├── src/                       # Core source code
│   ├── data_loader.py        # Data loading and cleaning
│   ├── volatility.py/        # Volatility models
│   │   ├── ewma.py          # Exponential Weighted Moving Average
│   │   ├── har.py           # Heterogeneous Autoregressive model
│   │   └── realized_vol.py  # Realized volatility
│   ├── market_making/        # Market-making logic
│   │   ├── quoting.py       # Optimal quoting strategy
│   │   ├── inventory.py     # Inventory management
│   │   └── pnl.py           # P&L calculation
│   ├── simulator/            # Backtesting engine
│   │   ├── engine.py        # Simulation loop
│   │   └── fill_model.py    # Order fill probability model
│   └── evaluation/           # Performance metrics
│       ├── metrics.py       # Risk/return metrics
│       └── plots.py         # Visualization utilities
│
├── test/                      # Test suite
│   └── test_data_loader.py   # Data loading tests with progress tracking
│
├── notebooks/                 # Jupyter notebooks
│   └── exploratory.ipynb     # EDA and analysis
│
└── results/                   # Output and plots
```

## Setup & Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Data

The sample dataset includes:
- **Symbol**: SPY
- **Date**: 2024-01-03
- **Data**: Full TAQ quotes (23.8M raw rows)
- **After cleaning**: ~1.5–2M quotes (RTH + noise filtering)

## Running Tests

### Test Data Loader (With Progress Tracking)

The test suite validates data loading, timestamp parsing, and resampling:

```bash
python test/test_data_loader.py
```

**What it tests:**
- ✅ Raw tick data loading (24M → 1.5M rows after RTH filtering)
- ✅ Timestamp parsing robustness
- ✅ Mid-price calculation
- ✅ Bid-ask spread validity
- ✅ Log return computation
- ✅ 1-minute resampling
- ✅ Index frequency validation

**Expected output** (with minute-by-minute progress updates for long operations):
```
======================================================================
TESTING load_data()
======================================================================
[14:32:45 | 0m 2s] Starting test_load_data_basic...
[14:32:45 | 0m 2s] Loading data from data/t822bpd5q8g1deky.csv
[14:33:15 | 0m 32s] ✅ Loaded 1,847,293 rows
...
[14:35:22 | 2m 39s] ✅ test_resample_reduces_rows PASSED
```

## Key Components

### Data Loader (`src/data_loader.py`)

**`load_data(filepath)`** — Load and clean TAQ data:
- Parses timestamps (handles WRDS nanosecond format)
- Filters to regular trading hours (09:30–16:00)
- Removes bad quotes (bid > ask, bid/ask ≤ 0)
- Filters extreme spreads (> 1% of mid) to reduce microstructure noise
- Computes mid-price and log returns

**Input**: Raw CSV with columns: `date, time_m, bid, bidsiz, ask, asksiz`

**Output**: DataFrame with:
```
Index: timestamp (DatetimeIndex)
Columns: bid, ask, bidsiz, asksiz, mid, log_return
```

**`resample_to_1min(df)`** — Aggregate tick data:
- Resamples to 1-minute frequency (last quote)
- Drops empty minutes
- Recomputes log returns post-resampling

### Volatility Models (`src/volatility.py/`)

- **EWMA**: Fast, responsive to recent shocks
- **HAR**: Captures multi-scale volatility (daily, weekly, monthly)
- **Realized Volatility**: True volatility from intraday returns

### Market-Making (`src/market_making/`)

- **Quoting Strategy**: Inventory-aware optimal bid-ask spreads
- **Inventory Management**: Track position and implement controls
- **P&L Calculation**: Realized and unrealized P&L tracking

### Simulator (`src/simulator/`)

- **Simulation Engine**: Full discrete-event tick-by-tick simulation
- **Fill Model**: Probability-based order execution
- **Performance Metrics**: Sharpe ratio, max drawdown, statistics

## Configuration

Edit `config.yaml` for:
- Data file path
- Volatility model parameters
- Market-making strategy settings
- Simulation parameters

## Troubleshooting

### Tests Hang or Run Slowly

The data loading test processes 24M raw rows → 1.5M cleaned rows. This takes time:
- First run with all tests: ~5–10 minutes
- Subsequent runs (if data cached): ~2–3 minutes

Monitor progress with timestamped output every minute.

### Data Format Issues

Ensure input CSV has columns:
- `date`: YYYYMMDD format
- `time_m`: HH:MM:SS or HH:MM:SS.nanoseconds
- `bid, ask, bidsiz, asksiz`: Numeric

### Memory Issues

If dataset is too large:
1. Reduce trading hours window
2. Filter by symbol or time range in `load_data()`
3. Process data in chunks

## Next Steps

1. ✅ **Data cleaning** — Complete
2. 🔄 **Volatility estimation** — Implement HAR, EWMA, RV models
3. 🔄 **Market-making strategy** — Implement quoting engine
4. 🔄 **Backtesting** — Run full simulation loop
5. 🔄 **Performance analysis** — Generate plots and metrics

## References

- **TAQ Data**: Trades and Quotes from WRDS
- **Volatility Forecasting**: Corsi (2009) HAR model
- **Market-Making**: Avellaneda & Stoikov (2008)
- **Microstructure**: Hasbrouck (2007)

## License

Private research project.
