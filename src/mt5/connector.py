"""MetaTrader 5 connector for live trading."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Callable
import logging

from ..core.types import Tick, Side

logger = logging.getLogger(__name__)


@dataclass
class MT5Config:
    """MT5 connection configuration."""
    login: int
    password: str
    server: str
    timeout: int = 60000  # ms


class MT5Connector:
    """
    Connector for MetaTrader 5.
    
    Handles:
    - Connection management
    - Tick subscription
    - Order execution
    """
    
    def __init__(self, config: MT5Config):
        """Initialize connector with config."""
        self.config = config
        self._mt5 = None
        self._connected = False
    
    def connect(self) -> bool:
        """
        Connect to MT5 terminal.
        
        Returns:
            True if connected successfully
        """
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
        except ImportError:
            logger.error(
                "MetaTrader5 not installed. Run: pip install MetaTrader5"
            )
            return False
        
        if not self._mt5.initialize():
            logger.error(f"MT5 initialize failed: {self._mt5.last_error()}")
            return False
        
        authorized = self._mt5.login(
            self.config.login,
            password=self.config.password,
            server=self.config.server
        )
        
        if not authorized:
            logger.error(f"MT5 login failed: {self._mt5.last_error()}")
            self._mt5.shutdown()
            return False
        
        self._connected = True
        logger.info(f"Connected to MT5: {self.config.server}")
        return True
    
    def disconnect(self) -> None:
        """Disconnect from MT5."""
        if self._mt5 and self._connected:
            self._mt5.shutdown()
            self._connected = False
            logger.info("Disconnected from MT5")
    
    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected
    
    def get_tick(self, symbol: str) -> Optional[Tick]:
        """
        Get current tick for symbol.
        
        Args:
            symbol: Trading symbol (e.g., "XAUUSD")
            
        Returns:
            Current Tick or None if error
        """
        if not self._connected:
            return None
        
        tick = self._mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        
        return Tick(
            timestamp=datetime.fromtimestamp(tick.time),
            bid=tick.bid,
            ask=tick.ask,
            bid_volume=tick.volume,
            ask_volume=0.0
        )
    
    def subscribe_ticks(
        self, 
        symbol: str, 
        callback: Callable[[Tick], None]
    ) -> None:
        """
        Subscribe to tick updates.
        
        Args:
            symbol: Trading symbol
            callback: Function to call on each tick
        """
        if not self._connected:
            logger.error("Not connected to MT5")
            return
        
        logger.info(f"Subscribing to {symbol} ticks...")
        
        # Enable symbol in Market Watch
        if not self._mt5.symbol_select(symbol, True):
            logger.error(f"Failed to select symbol {symbol}")
            return
        
        # Poll for ticks (MT5 doesn't have push subscription in Python)
        import time
        last_time = 0
        
        while True:
            tick = self._mt5.symbol_info_tick(symbol)
            if tick and tick.time != last_time:
                last_time = tick.time
                callback(Tick(
                    timestamp=datetime.fromtimestamp(tick.time),
                    bid=tick.bid,
                    ask=tick.ask,
                    bid_volume=tick.volume,
                    ask_volume=0.0
                ))
            time.sleep(0.01)  # 10ms polling
    
    def open_position(
        self,
        symbol: str,
        side: Side,
        lot_size: float,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        magic: int = 12345,
        comment: str = "ml_exit"
    ) -> Optional[int]:
        """
        Open a position.
        
        Args:
            symbol: Trading symbol
            side: LONG or SHORT
            lot_size: Position size in lots
            sl_price: Stop loss price
            tp_price: Take profit price
            magic: Magic number for identification
            comment: Order comment
            
        Returns:
            Position ticket or None if error
        """
        if not self._connected:
            return None
        
        # Get current prices
        tick = self._mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {symbol}")
            return None
        
        if side == Side.LONG:
            order_type = self._mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = self._mt5.ORDER_TYPE_SELL
            price = tick.bid
        
        request = {
            "action": self._mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": magic,
            "comment": comment,
            "type_time": self._mt5.ORDER_TIME_GTC,
            "type_filling": self._mt5.ORDER_FILLING_IOC,
        }
        
        if sl_price:
            request["sl"] = sl_price
        if tp_price:
            request["tp"] = tp_price
        
        result = self._mt5.order_send(request)
        
        if result.retcode != self._mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.comment}")
            return None
        
        logger.info(f"Opened {side.value} position: ticket={result.order}")
        return result.order
    
    def close_position(
        self,
        ticket: int,
        symbol: str,
        lot_size: float,
        side: Side,
        magic: int = 12345,
        comment: str = "ml_exit_close"
    ) -> bool:
        """
        Close a position.
        
        Args:
            ticket: Position ticket
            symbol: Trading symbol
            lot_size: Position size
            side: Original position side
            magic: Magic number
            comment: Order comment
            
        Returns:
            True if closed successfully
        """
        if not self._connected:
            return False
        
        tick = self._mt5.symbol_info_tick(symbol)
        if tick is None:
            return False
        
        # Opposite order to close
        if side == Side.LONG:
            order_type = self._mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = self._mt5.ORDER_TYPE_BUY
            price = tick.ask
        
        request = {
            "action": self._mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": magic,
            "comment": comment,
            "type_time": self._mt5.ORDER_TIME_GTC,
            "type_filling": self._mt5.ORDER_FILLING_IOC,
        }
        
        result = self._mt5.order_send(request)
        
        if result.retcode != self._mt5.TRADE_RETCODE_DONE:
            logger.error(f"Close failed: {result.comment}")
            return False
        
        logger.info(f"Closed position: ticket={ticket}")
        return True
    
    def get_position(self, ticket: int) -> Optional[dict]:
        """Get position by ticket."""
        if not self._connected:
            return None
        
        positions = self._mt5.positions_get(ticket=ticket)
        if positions and len(positions) > 0:
            pos = positions[0]
            return {
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 'long' if pos.type == 0 else 'short',
                'volume': pos.volume,
                'price': pos.price_open,
                'sl': pos.sl,
                'tp': pos.tp,
                'profit': pos.profit
            }
        return None
