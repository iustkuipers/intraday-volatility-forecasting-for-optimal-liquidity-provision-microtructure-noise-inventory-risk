import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import load_data, resample_to_1min


class ProgressTracker:
    """Track and print progress with timestamps every minute."""
    
    def __init__(self):
        self.start_time = time.time()
        self.last_print = self.start_time
    
    def log(self, message: str):
        """Print message with elapsed time."""
        current_time = time.time()
        elapsed = current_time - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp} | {minutes}m {seconds}s] {message}")
        self.last_print = current_time
    
    def check_and_print(self, test_name: str):
        """Print every minute of elapsed time."""
        current_time = time.time()
        if current_time - self.last_print >= 60:
            elapsed = current_time - self.start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp} | {minutes}m {seconds}s] ⏳ Still running: {test_name}...")
            self.last_print = current_time


progress = ProgressTracker()


class TestLoadData:
    """Test suite for load_data function."""
    
    def test_load_data_basic(self):
        """Test basic loading and structure of raw tick data."""
        progress.log("Starting test_load_data_basic...")
        filepath = "data/t822bpd5q8g1deky.csv"
        
        progress.log(f"Loading data from {filepath}")
        df = load_data(filepath)
        progress.log(f"✅ Loaded {len(df)} rows")
        
        # Check it's a DataFrame
        assert isinstance(df, pd.DataFrame)
        
        # Check rows exist
        assert len(df) > 0, "DataFrame is empty"
        
        # Check required columns
        required_cols = ["bid", "ask", "bidsiz", "asksiz", "mid", "log_return"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"
        
        print(df.head())
        progress.log(f"✅ test_load_data_basic PASSED")
    
    def test_load_data_timestamp_index(self):
        """Test timestamp index is properly set."""
        progress.log("Starting test_load_data_timestamp_index...")
        filepath = "data/t822bpd5q8g1deky.csv"
        df = load_data(filepath)
        
        # Check index is datetime
        assert isinstance(df.index, pd.DatetimeIndex), "Index is not DatetimeIndex"
        
        # Check index is sorted
        assert df.index.is_monotonic_increasing, "Index is not sorted"
        
        progress.log(f"✅ Timestamp index valid: {df.index.min()} to {df.index.max()}")
    
    def test_load_data_mid_price_valid(self):
        """Test mid price calculation is reasonable."""
        progress.log("Starting test_load_data_mid_price_valid...")
        filepath = "data/t822bpd5q8g1deky.csv"
        df = load_data(filepath)
        
        # Check mid = (bid + ask) / 2 (approximately)
        computed_mid = (df["bid"] + df["ask"]) / 2
        assert np.allclose(df["mid"], computed_mid), "Mid price not correctly computed"
        
        # Check no NaN in mid (except maybe first row for log_return)
        assert df["mid"].notna().sum() > 0, "No valid mid prices"
        
        progress.log(f"✅ Mid price valid. Range: {df['mid'].min():.4f} to {df['mid'].max():.4f}")
    
    def test_load_data_spread_positive(self):
        """Test bid-ask spread is always >= 0."""
        progress.log("Starting test_load_data_spread_positive...")
        filepath = "data/t822bpd5q8g1deky.csv"
        df = load_data(filepath)
        
        spread = df["ask"] - df["bid"]
        assert (spread >= 0).all(), "Found negative spreads (ask < bid)"
        
        progress.log(f"✅ All spreads valid. Mean spread: {spread.mean():.6f}")
    
    def test_load_data_log_returns(self):
        """Test log returns are computed correctly."""
        progress.log("Starting test_load_data_log_returns...")
        filepath = "data/t822bpd5q8g1deky.csv"
        df = load_data(filepath)
        
        # First log return should be NaN
        assert pd.isna(df["log_return"].iloc[0]), "First log return should be NaN"
        
        # Check log returns are reasonable (not exploding)
        valid_returns = df["log_return"].dropna()
        assert (valid_returns.abs() < 1).all(), "Log returns look unrealistic (>100%)"
        
        progress.log(f"✅ Log returns computed. Mean: {valid_returns.mean():.6e}, Std: {valid_returns.std():.6e}")
    
    def test_load_data_head_tail(self):
        """Print head and tail for visual inspection."""
        progress.log("Starting test_load_data_head_tail...")
        filepath = "data/t822bpd5q8g1deky.csv"
        df = load_data(filepath)
        
        print("\n--- HEAD ---")
        print(df.head())
        print("\n--- TAIL ---")
        print(df.tail())
        progress.log("✅ test_load_data_head_tail PASSED")


class TestResampleTo1Min:
    """Test suite for resample_to_1min function."""
    
    def test_resample_reduces_rows(self):
        """Test that resampling reduces tick data to 1-minute frequency."""
        progress.log("Starting test_resample_reduces_rows...")
        filepath = "data/t822bpd5q8g1deky.csv"
        progress.log("Loading tick data...")
        df_tick = load_data(filepath)
        progress.log(f"Loaded {len(df_tick)} rows. Starting resampling (this may take a few minutes)...")
        
        # Resample with progress tracking
        start_resample = time.time()
        df_1min = resample_to_1min(df_tick)
        elapsed = time.time() - start_resample
        
        # Should have significantly fewer rows
        assert len(df_1min) < len(df_tick), "Resampling did not reduce rows"
        assert len(df_1min) > 0, "Resampled DataFrame is empty"
        
        progress.log(f"✅ Resampling done in {elapsed:.1f}s. Tick data: {len(df_tick)} rows → 1-min data: {len(df_1min)} rows")
    
    def test_resample_structure(self):
        """Test resampled data has proper structure."""
        progress.log("Starting test_resample_structure...")
        filepath = "data/t822bpd5q8g1deky.csv"
        df_tick = load_data(filepath)
        df_1min = resample_to_1min(df_tick)
        
        # Check it's a DataFrame
        assert isinstance(df_1min, pd.DataFrame)
        
        # Check required columns exist
        required_cols = ["mid", "log_return"]
        for col in required_cols:
            assert col in df_1min.columns, f"Missing column: {col}"
        
        # Check index is datetime
        assert isinstance(df_1min.index, pd.DatetimeIndex)
        
        progress.log(f"✅ 1-minute data structure valid")
    
    def test_resample_mid_continues(self):
        """Test mid price is continuous and uses last quote."""
        filepath = "data/t822bpd5q8g1deky.csv"
        df_tick = load_data(filepath)
        df_1min = resample_to_1min(df_tick)
        
        # Should have mostly non-NaN mid (except maybe first), no crazy values
        valid_mids = df_1min["mid"].dropna()
        assert len(valid_mids) > len(df_1min) * 0.9, "Too many NaN values in resampled mid"
        
        print(f"✅ Mid price coverage: {len(valid_mids)}/{len(df_1min)} ({100*len(valid_mids)/len(df_1min):.1f}%)")
    
    def test_resample_log_returns_recomputed(self):
        """Test that log returns are recomputed after resampling."""
        filepath = "data/t822bpd5q8g1deky.csv"
        df_tick = load_data(filepath)
        df_1min = resample_to_1min(df_tick)
        
        # First log return should be NaN
        assert pd.isna(df_1min["log_return"].iloc[0]), "First log return should be NaN"
        
        # Check log returns are reasonable
        valid_returns = df_1min["log_return"].dropna()
        assert (valid_returns.abs() < 0.1).all(), "1-min log returns look unrealistic"
        
        print(f"✅ 1-min log returns valid. Mean: {valid_returns.mean():.6e}, Std: {valid_returns.std():.6e}")
    
    def test_resample_index_frequency(self):
        """Test resampled data has 1-minute frequency."""
        filepath = "data/t822bpd5q8g1deky.csv"
        df_tick = load_data(filepath)
        df_1min = resample_to_1min(df_tick)
        
        # Check time deltas (mostly 1 minute, may have gaps)
        if len(df_1min) > 1:
            deltas = df_1min.index.to_series().diff().dropna()
            # Most should be 1 minute, but there can be gaps
            mode_delta = deltas.mode()[0]
            assert mode_delta == pd.Timedelta(minutes=1), f"Most common delta is not 1 minute: {mode_delta}"
        
        print(f"✅ Index frequency is 1-minute")
    
    def test_resample_head_tail(self):
        """Print head and tail for visual inspection."""
        filepath = "data/t822bpd5q8g1deky.csv"
        df_tick = load_data(filepath)
        df_1min = resample_to_1min(df_tick)
        
        print("\n--- 1-MIN HEAD ---")
        print(df_1min.head())
        print("\n--- 1-MIN TAIL ---")
        print(df_1min.tail())


if __name__ == "__main__":
    # Run basic tests manually when script is executed directly
    print("=" * 70)
    print("TESTING load_data()")
    print("=" * 70)
    
    test_load = TestLoadData()
    test_load.test_load_data_basic()
    test_load.test_load_data_timestamp_index()
    test_load.test_load_data_mid_price_valid()
    test_load.test_load_data_spread_positive()
    test_load.test_load_data_log_returns()
    test_load.test_load_data_head_tail()
    
    print("\n" + "=" * 70)
    print("TESTING resample_to_1min()")
    print("=" * 70)
    
    test_resample = TestResampleTo1Min()
    test_resample.test_resample_reduces_rows()
    test_resample.test_resample_structure()
    test_resample.test_resample_mid_continues()
    test_resample.test_resample_log_returns_recomputed()
    test_resample.test_resample_index_frequency()
    test_resample.test_resample_head_tail()
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED")
    print("=" * 70)
