"""Optimized tick data loader using numpy arrays for fast iteration."""

from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, NamedTuple
import numpy as np
import pandas as pd

from ..core.types import Tick


class TickArray(NamedTuple):
    """Numpy arrays for fast tick access."""
    timestamps: np.ndarray  # datetime64
    bids: np.ndarray        # float64
    asks: np.ndarray        # float64
    bid_volumes: np.ndarray # float64
    ask_volumes: np.ndarray # float64


class TickDataLoader:
    """Load and iterate over tick data from parquet files - optimized version."""
    
    def __init__(self, file_path: Path | str):
        """Initialize loader with parquet file path."""
        self.file_path = Path(file_path)
        self._arrays: Optional[TickArray] = None
        self._df: Optional[pd.DataFrame] = None
        
    def load(self) -> "TickDataLoader":
        """Load data from parquet file into numpy arrays."""
        df = pd.read_parquet(self.file_path)
        
        # Parse timestamp column
        df['datetime'] = pd.to_datetime(
            df['Timestamp'], 
            format='%Y%m%d %H:%M:%S:%f'
        )
        df = df.sort_values('datetime').reset_index(drop=True)
        self._df = df
        
        # Convert to numpy arrays for fast access
        self._arrays = TickArray(
            timestamps=df['datetime'].values,
            bids=df['Bid price'].values.astype(np.float64),
            asks=df['Ask price'].values.astype(np.float64),
            bid_volumes=df['Bid volume'].values.astype(np.float64),
            ask_volumes=df['Ask volume'].values.astype(np.float64)
        )
        
        return self
    
    def __len__(self) -> int:
        """Return number of ticks."""
        if self._arrays is None:
            return 0
        return len(self._arrays.bids)
    
    def get_arrays(self) -> TickArray:
        """Get raw numpy arrays for fast processing."""
        if self._arrays is None:
            raise ValueError("Data not loaded. Call load() first.")
        return self._arrays
    
    def __iter__(self) -> Iterator[Tick]:
        """Iterate over ticks (slower, for compatibility)."""
        if self._arrays is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        arr = self._arrays
        for i in range(len(arr.bids)):
            yield Tick(
                timestamp=pd.Timestamp(arr.timestamps[i]).to_pydatetime(),
                bid=arr.bids[i],
                ask=arr.asks[i],
                bid_volume=arr.bid_volumes[i],
                ask_volume=arr.ask_volumes[i]
            )
    
    def to_dataframe(self) -> pd.DataFrame:
        """Return the underlying DataFrame."""
        if self._df is None:
            raise ValueError("Data not loaded. Call load() first.")
        return self._df.copy()
    
    @property
    def date_range(self) -> tuple[datetime, datetime]:
        """Return the date range of loaded data."""
        if self._arrays is None:
            raise ValueError("Data not loaded. Call load() first.")
        return (
            pd.Timestamp(self._arrays.timestamps[0]).to_pydatetime(),
            pd.Timestamp(self._arrays.timestamps[-1]).to_pydatetime()
        )
