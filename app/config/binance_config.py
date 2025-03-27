from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any
from enum import Enum

class BinanceMode(str, Enum):
    """Trading mode for Binance exchange."""
    SPOT = "spot"
    MARGIN = "margin"
    FUTURES = "futures"
    COIN_FUTURES = "coin-futures"
    PAPER = "paper"

class BinanceConfig(BaseModel):
    """
    Binance-specific configuration settings.
    
    This class encapsulates all Binance-specific settings used throughout the application,
    ensuring centralized configuration management.
    """
    
    # API configuration
    api_key: Optional[str] = Field(default=None, description="Binance API key")
    secret_key: Optional[str] = Field(default=None, description="Binance API secret")
    trading_mode: BinanceMode = Field(default=BinanceMode.SPOT, description="Trading mode")
    testnet: bool = Field(default=True, description="Whether to use testnet")
    quote_currency: str = Field(default="USDC", description="Default quote currency")
    
    # Rate limiting
    rate_limit_calls: int = Field(default=1200, description="Number of calls allowed in the rate limit window")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")
    
    # Trading pairs
    stablecoins: List[str] = Field(
        default=["USDT", "DAI", "BUSD", "TUSD", "USDC", "UST", "USDP", "USDN", "GUSD"],
        description="List of stablecoins to exclude from trading pairs"
    )
    
    # Order execution
    default_slippage_percent: float = Field(
        default=0.5, 
        description="Default slippage percentage for price calculations"
    )
    
    default_order_type: str = Field(
        default="limit", 
        description="Default order type (market, limit)"
    )
    
    # Futures trading (when in futures mode)
    default_leverage: int = Field(
        default=1, 
        description="Default leverage for futures trading"
    )
    
    # Order retry settings
    max_order_retries: int = Field(
        default=3,
        description="Maximum number of retries for failed orders"
    )
    
    retry_delay_seconds: int = Field(
        default=1,
        description="Delay between order retries in seconds"
    )
    
    # Networking settings
    timeout_seconds: int = Field(
        default=30,
        description="Timeout for API requests in seconds"
    )
    
    # Websocket settings
    use_websockets: bool = Field(
        default=True,
        description="Whether to use websockets for data streaming"
    )
    
    max_websocket_reconnect_attempts: int = Field(
        default=5,
        description="Maximum number of reconnection attempts for websockets"
    )
    
    # Risk management
    max_position_size_usdc: float = Field(
        default=1000.0,
        description="Maximum position size in USDC"
    )
    
    # Trading rule overrides - empty by default, can be populated with specific overrides
    trading_rule_overrides: Dict[str, Dict[str, Union[float, int, str]]] = Field(
        default={},
        description="Override specific trading rules for symbols"
    )
    
    @property
    def default_quote_currency(self) -> str:
        """Return the quote currency as the default_quote_currency for backward compatibility"""
        return self.quote_currency
    
    class Config:
        """Pydantic config"""
        validate_assignment = True
        extra = "forbid"  # Prevent extra fields 