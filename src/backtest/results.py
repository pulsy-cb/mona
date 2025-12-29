"""Backtest results container and statistics."""

from dataclasses import dataclass, field
from typing import Optional

from ..core.types import Trade


@dataclass
class BacktestResults:
    """Container for backtest results with statistics."""
    
    trades: list[Trade] = field(default_factory=list)
    initial_capital: float = 10000.0
    
    @property
    def total_trades(self) -> int:
        """Total number of completed trades."""
        return len(self.trades)
    
    @property
    def winning_trades(self) -> list[Trade]:
        """List of profitable trades."""
        return [t for t in self.trades if t.pnl > 0]
    
    @property
    def losing_trades(self) -> list[Trade]:
        """List of losing trades."""
        return [t for t in self.trades if t.pnl < 0]
    
    @property
    def win_rate(self) -> float:
        """Percentage of winning trades."""
        if self.total_trades == 0:
            return 0.0
        return len(self.winning_trades) / self.total_trades * 100
    
    @property
    def avg_win(self) -> float:
        """Average profit on winning trades."""
        wins = self.winning_trades
        if not wins:
            return 0.0
        return sum(t.pnl for t in wins) / len(wins)
    
    @property
    def avg_loss(self) -> float:
        """Average loss on losing trades (positive number)."""
        losses = self.losing_trades
        if not losses:
            return 0.0
        return abs(sum(t.pnl for t in losses) / len(losses))
    
    @property
    def total_pnl(self) -> float:
        """Total profit/loss."""
        return sum(t.pnl for t in self.trades)
    
    @property
    def profit_factor(self) -> float:
        """Gross profit / gross loss."""
        gross_profit = sum(t.pnl for t in self.winning_trades)
        gross_loss = abs(sum(t.pnl for t in self.losing_trades))
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss
    
    @property
    def avg_trade_pnl(self) -> float:
        """Average P&L per trade."""
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades
    
    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown in currency."""
        if not self.trades:
            return 0.0
        
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        
        for trade in self.trades:
            cumulative += trade.pnl
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def summary(self) -> dict:
        """Return summary statistics as dict."""
        return {
            'total_trades': self.total_trades,
            'win_rate': round(self.win_rate, 2),
            'avg_win': round(self.avg_win, 2),
            'avg_loss': round(self.avg_loss, 2),
            'total_pnl': round(self.total_pnl, 2),
            'profit_factor': round(self.profit_factor, 2),
            'avg_trade_pnl': round(self.avg_trade_pnl, 2),
            'max_drawdown': round(self.max_drawdown, 2)
        }
    
    def print_summary(self) -> None:
        """Print formatted summary."""
        s = self.summary()
        print("\n" + "=" * 50)
        print("BACKTEST RESULTS")
        print("=" * 50)
        print(f"Total Trades:    {s['total_trades']}")
        print(f"Win Rate:        {s['win_rate']}%")
        print(f"Avg Win:         {s['avg_win']} €")
        print(f"Avg Loss:        {s['avg_loss']} €")
        print(f"Total P&L:       {s['total_pnl']} €")
        print(f"Profit Factor:   {s['profit_factor']}")
        print(f"Avg Trade P&L:   {s['avg_trade_pnl']} €")
        print(f"Max Drawdown:    {s['max_drawdown']} €")
        print("=" * 50 + "\n")
