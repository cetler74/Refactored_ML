from enum import Enum
from typing import List, Dict, Optional, Set
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
import json
import os
from dotenv import load_dotenv
import logging
from pathlib import Path

from app.config.binance_config import BinanceConfig, BinanceMode
from app.config.cryptocom_config import CryptocomConfig, CryptocomMode

# Load environment variables at module level
load_dotenv()

# Load environment variables from .env file if it exists
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)

class AppMode(str, Enum):
    SIMULATION = "simulation"
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"

class Exchange(str, Enum):
    BINANCE = "binance"
    CRYPTOCOM = "cryptocom"

class BinanceConfig(BaseModel):
    api_key: str
    secret_key: str
    trading_mode: str
    testnet: bool
    quote_currency: str
    rate_limit_calls: int
    rate_limit_window: int
    default_slippage: float
    default_order_type: str
    max_position_size: float
    default_leverage: int
    max_order_retries: int
    retry_delay_seconds: int
    timeout_seconds: int

class CryptocomConfig(BaseModel):
    api_key: str
    secret_key: str
    trading_mode: str
    quote_currency: str
    rate_limit_calls: int
    rate_limit_window: int
    default_slippage: float
    default_order_type: str
    max_position_size: float
    default_leverage: int
    max_order_retries: int
    retry_delay_seconds: int
    timeout_seconds: int

class Settings(BaseSettings):
    # Application settings
    APP_MODE: str = Field(default="simulation")
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    LOG_FILE: str = Field(default="logs/trading_bot.log")
    
    # Initial balance setting
    INITIAL_BALANCE: float = Field(default=5000.0)
    
    # Database settings
    DB_HOST: str = Field(default="localhost")
    DB_PORT: int = Field(default=5432)
    DB_NAME: str = Field(default="trading_bot")
    DB_USER: str = Field(default="postgres")
    DB_PASSWORD: str = Field(default="")
    DATABASE_URL: str = Field(default="postgresql://carloslarramba@localhost:5432/trading_bot")
    
    # Redis settings
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_DB: int = Field(default=0)
    REDIS_PASSWORD: str = Field(default="")
    
    # Active exchanges
    ACTIVE_EXCHANGES: str = Field(default="[]")
    
    # Risk management settings
    MAX_POSITION_SIZE_USD: float = Field(default=300.0)
    MAX_POSITIONS: int = Field(default=5, description="Maximum number of trading positions to hold simultaneously")
    RISK_PER_TRADE: float = Field(default=0.02)
    STOP_LOSS_PERCENTAGE: float = Field(default=0.05)
    TAKE_PROFIT_PERCENTAGE: float = Field(default=0.10)
    
    # ML model settings
    MODEL_SAVE_PATH: str = Field(default="./models")
    FEATURE_STORE_PATH: str = Field(default="./features")
    TRAINING_INTERVAL_MINUTES: int = Field(default=60)
    PREDICTION_INTERVAL_MINUTES: int = Field(default=5)
    ENABLE_ML: bool = Field(default=False)
    ML_CONFIDENCE_THRESHOLD: float = Field(default=0.7)
    ML_TRAINING_DATA_LOOKBACK: int = Field(default=30)
    
    # Trading settings
    DEFAULT_TIMEFRAMES: List[str] = Field(default=["1m", "5m", "15m", "1h"])
    COOLDOWN_MINUTES: int = Field(default=15)
    MIN_VOLATILITY: float = Field(default=0.05)
    MAX_VOLATILITY: float = Field(default=0.25)
    MIN_DAILY_VOLUME_USD: float = Field(default=1_000_000.0)
    
    # API and frontend settings
    API_PORT: int = Field(default=8080)
    FRONTEND_PORT: int = Field(default=3001)
    
    # Exchange-specific settings
    BINANCE_API_KEY: str = Field(default="")
    BINANCE_SECRET_KEY: str = Field(default="")
    BINANCE_TRADING_MODE: str = Field(default="spot")
    BINANCE_TESTNET: str = Field(default="True")
    BINANCE_QUOTE_CURRENCY: str = Field(default="USDC")
    BINANCE_RATE_LIMIT_CALLS: str = Field(default="1200")
    BINANCE_RATE_LIMIT_WINDOW: str = Field(default="60")
    BINANCE_DEFAULT_SLIPPAGE: str = Field(default="0.5")
    BINANCE_DEFAULT_ORDER_TYPE: str = Field(default="limit")
    BINANCE_MAX_POSITION_SIZE: str = Field(default="100.0")
    BINANCE_DEFAULT_LEVERAGE: str = Field(default="1")
    BINANCE_MAX_ORDER_RETRIES: str = Field(default="3")
    BINANCE_RETRY_DELAY_SECONDS: str = Field(default="1")
    BINANCE_TIMEOUT_SECONDS: str = Field(default="30")
    
    CRYPTOCOM_API_KEY: str = Field(default="")
    CRYPTOCOM_SECRET_KEY: str = Field(default="")
    CRYPTOCOM_TRADING_MODE: str = Field(default="spot")
    CRYPTOCOM_QUOTE_CURRENCY: str = Field(default="USDC")
    CRYPTOCOM_RATE_LIMIT_CALLS: str = Field(default="100")
    CRYPTOCOM_RATE_LIMIT_WINDOW: str = Field(default="60")
    CRYPTOCOM_DEFAULT_SLIPPAGE: str = Field(default="0.5")
    CRYPTOCOM_DEFAULT_ORDER_TYPE: str = Field(default="limit")
    CRYPTOCOM_MAX_POSITION_SIZE: str = Field(default="100.0")
    CRYPTOCOM_DEFAULT_LEVERAGE: str = Field(default="1")
    CRYPTOCOM_MAX_ORDER_RETRIES: str = Field(default="3")
    CRYPTOCOM_RETRY_DELAY_SECONDS: str = Field(default="1")
    CRYPTOCOM_TIMEOUT_SECONDS: str = Field(default="30")
    
    # Exchange configurations
    binance_config: Optional[BinanceConfig] = None
    cryptocom_config: Optional[CryptocomConfig] = None

    MODEL_DIR: str = "./models"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow"
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Ensure model_save_path exists
        self.model_save_path = self.MODEL_SAVE_PATH
        os.makedirs(self.model_save_path, exist_ok=True)
        logger.info(f"Model save path set to: {self.model_save_path}")
        
        # Parse active exchanges string from environment variable
        try:
            # Remove quotes around the JSON string if they exist
            active_exchanges_str = self.ACTIVE_EXCHANGES.strip('"\'')
            active_exchanges_list = json.loads(active_exchanges_str)
            self.active_exchanges = {Exchange(ex) for ex in active_exchanges_list}
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Error parsing ACTIVE_EXCHANGES: {e}")
            self.active_exchanges = set()
        
        # Parse default timeframes
        try:
            timeframes_str = os.environ.get("DEFAULT_TIMEFRAMES", '["1m","5m","15m","1h"]')
            # Remove quotes around the JSON string if they exist
            timeframes_str = timeframes_str.strip('"\'')
            self.default_timeframes = json.loads(timeframes_str)
            logger.info(f"Using default timeframes: {self.default_timeframes}")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Error parsing DEFAULT_TIMEFRAMES: {e}, using defaults")
            self.default_timeframes = ["1m", "5m", "15m", "1h"]
        
        # Initialize Binance config
        try:
            self.binance_config = BinanceConfig(
                api_key=os.environ.get("BINANCE_API_KEY", ""),
                secret_key=os.environ.get("BINANCE_SECRET_KEY", ""),
                trading_mode=os.environ.get("BINANCE_TRADING_MODE", "paper"),
                testnet=os.environ.get("BINANCE_TESTNET", "True").lower() == "true",
                quote_currency=os.environ.get("BINANCE_QUOTE_CURRENCY", "USDT"),
                rate_limit_calls=int(os.environ.get("BINANCE_RATE_LIMIT_CALLS", 1200)),
                rate_limit_window=int(os.environ.get("BINANCE_RATE_LIMIT_WINDOW", 60)),
                default_slippage=float(os.environ.get("BINANCE_DEFAULT_SLIPPAGE", 0.001)),
                default_order_type=os.environ.get("BINANCE_DEFAULT_ORDER_TYPE", "MARKET"),
                max_position_size=float(os.environ.get("BINANCE_MAX_POSITION_SIZE", 1000.0)),
                default_leverage=int(os.environ.get("BINANCE_DEFAULT_LEVERAGE", 1)),
                max_order_retries=int(os.environ.get("BINANCE_MAX_ORDER_RETRIES", 3)),
                retry_delay_seconds=int(os.environ.get("BINANCE_RETRY_DELAY_SECONDS", 1)),
                timeout_seconds=int(os.environ.get("BINANCE_TIMEOUT_SECONDS", 10))
            )
            logger.info("Binance config initialized from environment variables")
        except Exception as e:
            logger.warning(f"Error initializing Binance config: {e}")

        # Initialize Crypto.com config
        try:
            self.cryptocom_config = CryptocomConfig(
                api_key=os.environ.get("CRYPTOCOM_API_KEY", ""),
                secret_key=os.environ.get("CRYPTOCOM_SECRET_KEY", ""),
                trading_mode=os.environ.get("CRYPTOCOM_TRADING_MODE", "paper"),
                quote_currency=os.environ.get("CRYPTOCOM_QUOTE_CURRENCY", "USDT"),
                rate_limit_calls=int(os.environ.get("CRYPTOCOM_RATE_LIMIT_CALLS", 100)),
                rate_limit_window=int(os.environ.get("CRYPTOCOM_RATE_LIMIT_WINDOW", 60)),
                default_slippage=float(os.environ.get("CRYPTOCOM_DEFAULT_SLIPPAGE", 0.001)),
                default_order_type=os.environ.get("CRYPTOCOM_DEFAULT_ORDER_TYPE", "MARKET"),
                max_position_size=float(os.environ.get("CRYPTOCOM_MAX_POSITION_SIZE", 1000.0)),
                default_leverage=int(os.environ.get("CRYPTOCOM_DEFAULT_LEVERAGE", 1)),
                max_order_retries=int(os.environ.get("CRYPTOCOM_MAX_ORDER_RETRIES", 3)),
                retry_delay_seconds=int(os.environ.get("CRYPTOCOM_RETRY_DELAY_SECONDS", 1)),
                timeout_seconds=int(os.environ.get("CRYPTOCOM_TIMEOUT_SECONDS", 10))
            )
            logger.info("Crypto.com config initialized from environment variables")
        except Exception as e:
            logger.warning(f"Error initializing Crypto.com config: {e}")

        # Load exchange credentials from environment variables
        self.exchange_credentials = {
            "binance": {
                "api_key": os.environ.get("BINANCE_API_KEY", ""),
                "secret_key": os.environ.get("BINANCE_SECRET_KEY", "")
            },
            "cryptocom": {
                "api_key": os.environ.get("CRYPTOCOM_API_KEY", ""),
                "secret_key": os.environ.get("CRYPTOCOM_SECRET_KEY", "")
            }
        }
        
        # Log loaded credentials (security masked)
        has_api_key = bool(self.exchange_credentials["binance"]["api_key"])
        has_secret = bool(self.exchange_credentials["binance"]["secret_key"])
        logger.info(f"Loaded Binance credentials: API Key: {'Found' if has_api_key else 'Not found'}, Secret Key: {'Found' if has_secret else 'Not found'}")
        
        # Database settings
        self.database_url = os.environ.get("DATABASE_URL", "sqlite:///./data/trading_bot.db")
        self.create_tables = os.environ.get("CREATE_TABLES", "False").lower() == "true"
        logger.info(f"Database URL from environment: {self.database_url}")
        
        # MLflow settings
        self.use_mlflow = os.environ.get("USE_MLFLOW", "False").lower() == "true"
        self.mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///./mlflow.db")
        
        # Set up other settings from environment or defaults
        self.MAX_POSITION_SIZE_USD = float(kwargs.get("MAX_POSITION_SIZE_USD", "1000.0"))
        self.RISK_PER_TRADE = float(kwargs.get("RISK_PER_TRADE", "0.02"))  # 2% risk per trade

    @property
    def min_daily_volume_usd(self) -> float:
        """Return the minimum daily volume in USD for pair selection."""
        return self.MIN_DAILY_VOLUME_USD
    
    @property
    def min_volatility(self) -> float:
        """Return the minimum volatility for pair selection."""
        return self.MIN_VOLATILITY
    
    @property
    def max_volatility(self) -> float:
        """Return the maximum volatility for pair selection."""
        return self.MAX_VOLATILITY
    
    @property
    def max_positions(self) -> int:
        """Return the maximum number of positions."""
        return self.MAX_POSITIONS

    @property
    def db_connection_string(self) -> str:
        """Return the database connection string for backward compatibility."""
        return self.DATABASE_URL

    @property
    def simulation_mode(self) -> bool:
        """Return whether the app is running in simulation mode."""
        return self.APP_MODE.lower() in ["simulation", "backtest"]

# Global settings instance
_settings = None

def get_settings() -> Settings:
    """Get application settings singleton"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings 