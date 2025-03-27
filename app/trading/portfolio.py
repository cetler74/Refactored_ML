import logging
from datetime import datetime
from typing import Dict, Optional

class PortfolioManager:
    """Manages the trading portfolio, including balance and positions."""
    
    def __init__(self, settings):
        """Initialize the portfolio manager with settings."""
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Initialize portfolio state
        self._total_balance = float(settings.INITIAL_BALANCE)
        self._available_balance = float(settings.INITIAL_BALANCE)
        self._allocated_balance = 0.0
        self._positions = {}
        self._trade_history = []
        
        self.logger.info(f"Portfolio initialized with {self._total_balance:.2f} USDT")
    
    @property
    def total_balance(self) -> float:
        """Get the total portfolio balance."""
        return self._total_balance
    
    @property
    def available_balance(self) -> float:
        """Get the available (unallocated) balance."""
        return self._available_balance
    
    @property
    def allocated_balance(self) -> float:
        """Get the allocated balance (in positions)."""
        return self._allocated_balance
    
    @property
    def positions(self) -> Dict:
        """Get current positions."""
        return self._positions.copy()
    
    def can_open_position(self, cost: float) -> bool:
        """Check if a new position can be opened with the given cost."""
        return cost <= self._available_balance
    
    def open_position(self, symbol: str, quantity: float, entry_price: float) -> bool:
        """Open a new position."""
        cost = quantity * entry_price
        
        if not self.can_open_position(cost):
            self.logger.warning(f"Cannot open position: Insufficient funds (Required: {cost:.2f} USDT, Available: {self._available_balance:.2f} USDT)")
            return False
        
        # Create position record
        position = {
            'symbol': symbol,
            'quantity': quantity,
            'entry_price': entry_price,
            'current_price': entry_price,
            'timestamp': datetime.now().isoformat(),
            'pnl': 0.0,
            'pnl_percent': 0.0
        }
        
        # Update balances
        self._available_balance -= cost
        self._allocated_balance += cost
        self._positions[symbol] = position
        
        self.logger.info(f"Opened position: {symbol} - {quantity} units at {entry_price:.2f} USDT")
        return True
    
    def close_position(self, symbol: str, exit_price: float) -> Optional[Dict]:
        """Close an existing position."""
        if symbol not in self._positions:
            self.logger.warning(f"Cannot close position: No open position found for {symbol}")
            return None
        
        position = self._positions[symbol]
        exit_value = position['quantity'] * exit_price
        entry_value = position['quantity'] * position['entry_price']
        
        # Calculate P&L
        pnl = exit_value - entry_value
        pnl_percent = (pnl / entry_value) * 100
        
        # Update balances
        self._available_balance += exit_value
        self._allocated_balance -= entry_value
        self._total_balance += pnl
        
        # Record trade in history
        trade_record = {
            **position,
            'exit_price': exit_price,
            'exit_time': datetime.now().isoformat(),
            'pnl': pnl,
            'pnl_percent': pnl_percent
        }
        self._trade_history.append(trade_record)
        
        # Remove position
        del self._positions[symbol]
        
        self.logger.info(f"Closed position: {symbol} at {exit_price:.2f} USDT - P&L: {pnl:.2f} USDT ({pnl_percent:.2f}%)")
        return trade_record
    
    def update_position(self, symbol: str, current_price: float) -> None:
        """Update position with current market price."""
        if symbol in self._positions:
            position = self._positions[symbol]
            old_price = position['current_price']
            
            # Update position metrics
            position['current_price'] = current_price
            position['pnl'] = position['quantity'] * (current_price - position['entry_price'])
            position['pnl_percent'] = (position['pnl'] / (position['quantity'] * position['entry_price'])) * 100
            
            # Log significant price changes (>1%)
            price_change_pct = ((current_price - old_price) / old_price) * 100
            if abs(price_change_pct) > 1:
                self.logger.info(f"Position {symbol} price updated: {current_price:.2f} USDT ({price_change_pct:+.2f}%)")
    
    def get_portfolio_summary(self) -> Dict:
        """Get a summary of the portfolio state."""
        return {
            'total_balance': self._total_balance,
            'available_balance': self._available_balance,
            'allocated_balance': self._allocated_balance,
            'num_positions': len(self._positions),
            'positions': self._positions,
            'trade_history': self._trade_history[-10:]  # Last 10 trades
        } 