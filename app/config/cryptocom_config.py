from enum import Enum
from typing import Dict, Optional
from pydantic import BaseModel, Field

class CryptocomMode(str, Enum):
    SPOT = "spot"
    MARGIN = "margin"
    DERIVATIVES = "derivatives"

class CryptocomConfig(BaseModel):
    """Crypto.com-specific configuration."""
    api_key: str = ""
    api_secret: str = ""
    trading_mode: CryptocomMode = CryptocomMode.SPOT
    default_quote_currency: str = "USDC"
    rate_limit_calls: int = 100
    rate_limit_window: int = 60
    default_slippage_percent: float = 0.5
    default_order_type: str = "limit"
    max_position_size_usdc: float = 100.0
    default_leverage: int = 1
    max_order_retries: int = 3
    retry_delay_seconds: int = 1
    timeout_seconds: int = 30
    trading_rule_overrides: Dict[str, Dict] = Field(default_factory=dict) 