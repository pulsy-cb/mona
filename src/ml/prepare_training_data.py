"""Pre-compute all features and indicators for ultra-fast RL training.

This module processes tick data once and generates all features vectorially,
eliminating the need for per-step calculations during training.

Expected speedup: 10-50x compared to FastTradingEnv.
"""

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    ta = None


def prepare_training_data(
    parquet_file: Path,
    timeframe: str = "1s",
    cache_file: Optional[Path] = None,
    force_recompute: bool = False
) -> dict[str, np.ndarray]:
    """
    Pre-calculate ALL features and indicators for training.
    
    This function performs vectorized computation of:
    - Bollinger Bands (upper, lower, middle, percent_b, width)
    - ATR (14-period)
    - RSI (14-period)
    - Tick-level volatility and velocity
    
    Args:
        parquet_file: Path to tick data parquet file
        timeframe: Candle timeframe for indicators (default: "1s")
        cache_file: Optional path to save/load cached results
        force_recompute: Force recomputation even if cache exists
        
    Returns:
        Dictionary containing:
            - timestamps: int64 array of Unix timestamps
            - bids: float32 array of bid prices
            - asks: float32 array of ask prices
            - features: float32 array of shape (N_ticks, N_features)
            - feature_names: list of feature names
    """
    if ta is None:
        raise ImportError(
            "pandas_ta not installed. Run:\n"
            "pip install pandas-ta"
        )
    
    # Check cache
    if cache_file and cache_file.exists() and not force_recompute:
        logging.info(f"Loading cached preprocessed data from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        return {
            'timestamps': data['timestamps'],
            'bids': data['bids'],
            'asks': data['asks'],
            'features': data['features'],
            'feature_names': data['feature_names'].tolist()
        }
    
    logging.info(f"Loading tick data from {parquet_file}...")
    df = pd.read_parquet(parquet_file)
    
    # Normalize column names to lowercase and remove spaces
    df.columns = [col.lower().replace(' ', '') for col in df.columns]
    
    # Handle different column naming conventions
    if 'bidprice' in df.columns:
        df = df.rename(columns={'bidprice': 'bid', 'askprice': 'ask'})
    
    # Ensure we have required columns
    if 'timestamp' not in df.columns or 'bid' not in df.columns or 'ask' not in df.columns:
        raise ValueError(f"Parquet file must contain 'timestamp', 'bid', and 'ask' columns. Found: {df.columns.tolist()}")
    
    # Convert timestamp to datetime and set as index
    logging.info("Converting timestamps to datetime index...")
    
    # Handle different timestamp formats
    if df['timestamp'].dtype == 'object':
        # String format like "20250929 00:00:00:092"
        df['datetime'] = pd.to_datetime(df['timestamp'], format='%Y%m%d %H:%M:%S:%f')
    elif pd.api.types.is_integer_dtype(df['timestamp']):
        # Unix timestamp (seconds)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    else:
        # Already datetime
        df['datetime'] = pd.to_datetime(df['timestamp'])
    
    df = df.set_index('datetime')
    
    # Keep original tick data
    tick_bids = df['bid'].values
    tick_asks = df['ask'].values
    tick_timestamps = df['timestamp'].values
    
    logging.info(f"Loaded {len(df):,} ticks")
    
    # ========================================
    # STEP 1: Resample to candles
    # ========================================
    logging.info(f"Resampling to {timeframe} candles...")
    
    # Use bid for OHLC (could also use mid = (bid + ask) / 2)
    candles = df['bid'].resample(timeframe).agg(['first', 'max', 'min', 'last'])
    candles.columns = ['open', 'high', 'low', 'close']
    
    # Drop NaN candles (if any)
    candles = candles.dropna()
    
    logging.info(f"Created {len(candles):,} candles")
    
    # ========================================
    # STEP 2: Vectorized Indicator Calculation
    # ========================================
    logging.info("Calculating Bollinger Bands...")
    bb = ta.bbands(candles['close'], length=20, std=2)
    
    # pandas_ta returns a DataFrame with column names that may vary
    # Dynamically find the column names
    bb_cols = bb.columns.tolist()
    candles['bb_upper'] = bb[bb_cols[0]]  # BBL (lower)
    candles['bb_mid'] = bb[bb_cols[1]]    # BBM (middle)
    candles['bb_lower'] = bb[bb_cols[2]]  # BBU (upper)
    candles['bb_percent'] = bb[bb_cols[3]] if len(bb_cols) > 3 else (candles['close'] - candles['bb_lower']) / (candles['bb_upper'] - candles['bb_lower'])
    
    # BB width normalized
    candles['bb_width'] = (candles['bb_upper'] - candles['bb_lower']) / candles['bb_mid']
    
    logging.info("Calculating ATR...")
    atr_result = ta.atr(candles['high'], candles['low'], candles['close'], length=14)
    candles['atr'] = atr_result if isinstance(atr_result, pd.Series) else atr_result.iloc[:, 0]
    
    logging.info("Calculating RSI...")
    rsi_result = ta.rsi(candles['close'], length=14)
    candles['rsi'] = rsi_result if isinstance(rsi_result, pd.Series) else rsi_result.iloc[:, 0]
    
    # ========================================
    # STEP 3: Normalize Indicators
    # ========================================
    logging.info("Normalizing indicators...")
    
    # ATR normalization: current ATR / rolling mean ATR
    candles['atr_norm'] = candles['atr'] / candles['atr'].rolling(window=1000, min_periods=1).mean()
    
    # RSI normalization: (RSI - 50) / 50 to get range [-1, 1]
    candles['rsi_norm'] = (candles['rsi'] - 50) / 50
    
    # BB width normalization: current width / rolling mean width
    candles['bb_width_norm'] = candles['bb_width'] / candles['bb_width'].rolling(window=1000, min_periods=1).mean()
    
    # ========================================
    # STEP 4: Realign with Ticks (Forward Fill)
    # ========================================
    logging.info("Realigning indicators to tick-level...")
    
    # Join candles back to original tick data (forward fill)
    df_features = df[['bid', 'ask']].join(candles, how='left').ffill()
    
    # Drop any remaining NaN (from warmup period)
    initial_len = len(df_features)
    df_features = df_features.dropna()
    dropped = initial_len - len(df_features)
    if dropped > 0:
        logging.info(f"Dropped {dropped:,} ticks during warmup period")
    
    # ========================================
    # STEP 5: Tick-Level Features
    # ========================================
    logging.info("Calculating tick-level features...")
    
    # Velocity: price change over last 5 ticks, normalized by ATR
    df_features['velocity_5'] = df_features['bid'].diff(5) / df_features['atr']
    
    # Volatility: rolling std dev over 10 ticks, normalized by ATR
    df_features['volatility_10'] = df_features['bid'].rolling(10).std() / df_features['atr']
    
    # Fill any NaN from diff/rolling operations
    df_features = df_features.fillna(0.0)
    
    # ========================================
    # STEP 6: Extract Feature Matrix
    # ========================================
    logging.info("Extracting feature matrix...")
    
    # Select features to include (STATIC features only - no position-dependent features)
    feature_columns = [
        'bb_percent',       # BB %B
        'bb_width_norm',    # BB width (normalized)
        'atr_norm',         # ATR (normalized)
        'rsi_norm',         # RSI (normalized)
        'volatility_10',    # Tick volatility
        'velocity_5'        # Tick velocity
    ]
    
    # Convert to numpy arrays (float32 for memory efficiency)
    features = df_features[feature_columns].values.astype(np.float32)
    bids = df_features['bid'].values.astype(np.float32)
    asks = df_features['ask'].values.astype(np.float32)
    
    # Get timestamps for the filtered data (convert datetime index to Unix timestamp)
    timestamps = (df_features.index.astype(np.int64) // 10**9).astype(np.int64)
    
    logging.info(f"Final dataset: {len(features):,} ticks with {len(feature_columns)} features")
    logging.info(f"Feature names: {feature_columns}")
    
    # ========================================
    # STEP 7: Save Cache (if requested)
    # ========================================
    result = {
        'timestamps': timestamps.astype(np.int64),
        'bids': bids,
        'asks': asks,
        'features': features,
        'feature_names': feature_columns
    }
    
    if cache_file:
        logging.info(f"Saving preprocessed data to {cache_file}...")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_file,
            timestamps=result['timestamps'],
            bids=result['bids'],
            asks=result['asks'],
            features=result['features'],
            feature_names=np.array(result['feature_names'])
        )
        logging.info(f"Cache saved ({cache_file.stat().st_size / 1024 / 1024:.1f} MB)")
    
    return result


def main():
    """CLI entry point for preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Pre-compute features for RL training"
    )
    parser.add_argument(
        "--data", "-d",
        type=Path,
        default=Path("XAUUSD.parquet"),
        help="Path to tick data parquet file"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("models/XAUUSD_preprocessed.npz"),
        help="Output path for preprocessed data"
    )
    parser.add_argument(
        "--timeframe", "-t",
        type=str,
        default="1s",
        help="Candle timeframe (e.g., '1s', '5s', '1min')"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force recomputation even if cache exists"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    result = prepare_training_data(
        parquet_file=args.data,
        timeframe=args.timeframe,
        cache_file=args.output,
        force_recompute=args.force
    )
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Total ticks: {len(result['bids']):,}")
    print(f"Features: {len(result['feature_names'])}")
    print(f"Feature names: {result['feature_names']}")
    print(f"Output file: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()
