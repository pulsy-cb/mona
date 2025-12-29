"""Strategy module - Entry and exit strategies."""

from .base import BaseExitStrategy
from .original import OriginalTrailingStop
from .heuristic import HeuristicExit
from .dark_venus import DarkVenusStrategy

__all__ = [
    'BaseExitStrategy',
    'OriginalTrailingStop', 
    'HeuristicExit',
    'DarkVenusStrategy'
]
