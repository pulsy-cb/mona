"""Optimized tick data loader using numpy arrays for fast iteration.

Memory-efficient implementation using PyArrow streaming.
Designed to handle 25M+ tick files on machines with limited RAM.
"""

from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, NamedTuple
import numpy as np
import pyarrow.parquet as pq
import logging
import gc

from ..core.types import Tick

logger = logging.getLogger(__name__)


class TickArray(NamedTuple):
    """Numpy arrays for fast tick access."""
    timestamps: np.ndarray  # int64 (seconds since epoch)
    bids: np.ndarray        # float32
    asks: np.ndarray        # float32
    bid_volumes: np.ndarray # float32
    ask_volumes: np.ndarray # float32


def parse_timestamp_batch(timestamps: np.ndarray) -> np.ndarray:
    """Parse timestamp strings to int64 seconds since epoch.
    
    Format: '20231001 00:00:00:123' -> seconds since epoch
    """
    result = np.zeros(len(timestamps), dtype=np.int64)
    
    for i, ts in enumerate(timestamps):
        if isinstance(ts, bytes):
            ts = ts.decode('utf-8')
        # Parse '20231001 00:00:00:123'
        try:
            year = int(ts[0:4])
            month = int(ts[4:6])
            day = int(ts[6:8])
            hour = int(ts[9:11])
            minute = int(ts[12:14])
            second = int(ts[15:17])
            
            # Simple calculation (ignoring leap years for speed, close enough for trading)
            # Days since 1970-01-01
            days = (year - 1970) * 365 + (year - 1970) // 4
            month_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
            days += month_days[month - 1] + day - 1
            
            result[i] = days * 86400 + hour * 3600 + minute * 60 + second
        except:
            result[i] = 0
    
    return result


class TickDataLoader:
    """Load and iterate over tick data from parquet files - streaming version.
    
    Memory optimizations:
    - Loads parquet in row-group batches (streaming)
    - Uses float32 instead of float64 (50% RAM reduction)
    - Supports max_rows to limit data size
    - Pre-computes timestamps as int64 seconds
    """
    
    def __init__(
        self, 
        file_path: Path | str,
        sample_ratio: float = 1.0,
        max_rows: int | None = None
    ):
        """Initialize loader with parquet file path.
        
        Args:
            file_path: Path to parquet file
            sample_ratio: Fraction of data to use (0.0 to 1.0). Default 1.0 = all data
            max_rows: Maximum number of rows to load. None = no limit
                     Recommended: 5_000_000 for 8GB RAM, 2_000_000 for 4GB RAM
        """
        self.file_path = Path(file_path)
        self.sample_ratio = min(1.0, max(0.01, sample_ratio))
        self.max_rows = max_rows
        self._arrays: Optional[TickArray] = None
        self._date_range: Optional[tuple] = None
        
    def load(self) -> "TickDataLoader":
        """Load data from parquet file in streaming mode (memory efficient)."""
        
        logger.info(f"Loading {self.file_path} with PyArrow streaming...")
        
        # Open parquet file for metadata
        parquet_file = pq.ParquetFile(self.file_path)
        total_rows = parquet_file.metadata.num_rows
        num_row_groups = parquet_file.metadata.num_row_groups
        
        logger.info(f"File has {total_rows:,} rows in {num_row_groups} row groups")
        
        # Calculate how many rows to load
        target_rows = total_rows
        use_sampling = False
        
        if self.sample_ratio < 1.0:
            target_rows = int(total_rows * self.sample_ratio)
            use_sampling = True
        
        if self.max_rows and self.max_rows < target_rows:
            target_rows = self.max_rows
            # If user wants max_rows but didn't ask for sampling ratio, 
            # we should effectively take the 'head' (contiguous), not sample.
            # Only use sampling if explicitly requested via sample_ratio.
        
        logger.info(f"Target: {target_rows:,} rows (Sampling: {use_sampling})")
        
        # Pre-allocate arrays
        timestamps = np.zeros(target_rows, dtype=np.int64)
        bids = np.zeros(target_rows, dtype=np.float32)
        asks = np.zeros(target_rows, dtype=np.float32)
        bid_vols = np.zeros(target_rows, dtype=np.float32)
        ask_vols = np.zeros(target_rows, dtype=np.float32)
        
        # Calculate step for sampling ONLY if requested
        step = 1
        if use_sampling:
            step = max(1, total_rows // target_rows)
        
        # Load row groups one at a time
        current_idx = 0
        
        for rg_idx in range(num_row_groups):
            if current_idx >= target_rows:
                break
                
            # Read single row group
            table = parquet_file.read_row_group(rg_idx, columns=[
                'Timestamp', 'Bid price', 'Ask price', 'Bid volume', 'Ask volume'
            ])
            
            rg_timestamps = table.column('Timestamp').to_numpy()
            rg_bids = table.column('Bid price').to_numpy().astype(np.float32)
            rg_asks = table.column('Ask price').to_numpy().astype(np.float32)
            rg_bid_vols = table.column('Bid volume').to_numpy().astype(np.float32)
            rg_ask_vols = table.column('Ask volume').to_numpy().astype(np.float32)
            
            rg_len = len(rg_bids)
            
            # Sample from this row group
            for j in range(0, rg_len, step):
                if current_idx >= target_rows:
                    break
                    
                # Parse timestamp inline
                ts = rg_timestamps[j]
                if isinstance(ts, bytes):
                    ts = ts.decode('utf-8')
                try:
                    year = int(ts[0:4])
                    month = int(ts[4:6])
                    day = int(ts[6:8])
                    hour = int(ts[9:11])
                    minute = int(ts[12:14])
                    second = int(ts[15:17])
                    
                    month_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
                    days = (year - 1970) * 365 + (year - 1970) // 4 + month_days[month - 1] + day - 1
                    timestamps[current_idx] = days * 86400 + hour * 3600 + minute * 60 + second
                except:
                    timestamps[current_idx] = 0
                
                bids[current_idx] = rg_bids[j]
                asks[current_idx] = rg_asks[j]
                bid_vols[current_idx] = rg_bid_vols[j]
                ask_vols[current_idx] = rg_ask_vols[j]
                current_idx += 1
            
            # Free row group memory
            del table, rg_timestamps, rg_bids, rg_asks, rg_bid_vols, rg_ask_vols
            gc.collect()
            
            if rg_idx % 10 == 0:
                logger.info(f"  Loaded {current_idx:,}/{target_rows:,} rows...")
        
        # Trim arrays to actual size
        actual_size = current_idx
        if actual_size < target_rows:
            timestamps = timestamps[:actual_size]
            bids = bids[:actual_size]
            asks = asks[:actual_size]
            bid_vols = bid_vols[:actual_size]
            ask_vols = ask_vols[:actual_size]
        
        # Sort by timestamp
        sort_idx = np.argsort(timestamps)
        
        self._arrays = TickArray(
            timestamps=timestamps[sort_idx],
            bids=bids[sort_idx],
            asks=asks[sort_idx],
            bid_volumes=bid_vols[sort_idx],
            ask_volumes=ask_vols[sort_idx]
        )
        
        # Store date range
        self._date_range = (
            datetime.fromtimestamp(int(self._arrays.timestamps[0])),
            datetime.fromtimestamp(int(self._arrays.timestamps[-1]))
        )
        
        del sort_idx
        gc.collect()
        
        logger.info(f"Loaded {len(self._arrays.bids):,} ticks into memory")
        
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
                timestamp=datetime.fromtimestamp(int(arr.timestamps[i])),
                bid=float(arr.bids[i]),
                ask=float(arr.asks[i]),
                bid_volume=float(arr.bid_volumes[i]),
                ask_volume=float(arr.ask_volumes[i])
            )
    
    @property
    def date_range(self) -> tuple[datetime, datetime]:
        """Return the date range of loaded data."""
        if self._date_range is None:
            raise ValueError("Data not loaded. Call load() first.")
        return self._date_range
