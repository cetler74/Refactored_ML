"""
Database Manager

This module handles database connection and operations.
"""

import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import aiosqlite
import os
import uuid
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey, Boolean, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.sql import text
import traceback

logger = logging.getLogger(__name__)

# Create the base class for SQLAlchemy models
Base = declarative_base()

class DatabaseManager:
    """
    Manages database connections and operations.
    Acts as a facade for various repositories.
    """
    
    def __init__(self, db_path="data/trading_bot.db", debug=False):
        self.db_path = db_path
        self.connection = None
        self.engine = None
        self.async_session_factory = None
        self.debug = debug
        self.initialized = False
        
        # Check if db_path is already a fully qualified database URL
        self.is_connection_string = db_path.startswith('postgresql://') or db_path.startswith('sqlite:///')
        
        if not self.is_connection_string:
            # Make sure the data directory exists when using a file path
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Sub-repositories for different data types
        self.trade_repository = TradeRepository(self)
        self.balance_repository = BalanceRepository(self)
        self.market_data_repository = MarketDataRepository(self)
        
        logger.info(f"Database manager initialized with path: {db_path}")
    
    async def connect(self):
        """Establish connection to the database."""
        try:
            self.connection = await aiosqlite.connect(self.db_path)
            # Enable foreign keys
            await self.connection.execute("PRAGMA foreign_keys = ON")
            # Enable WAL mode for better concurrency
            await self.connection.execute("PRAGMA journal_mode = WAL")
            
            # Initialize tables
            await self.initialize_tables()
            
            logger.info("Database connection established")
            return True
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            return False
    
    async def initialize_tables(self):
        """Initialize database tables if they don't exist."""
        try:
            # Create trades table
            await self.connection.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    trading_pair TEXT NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    trading_pair_price REAL NOT NULL,
                    trading_fee REAL NOT NULL,
                    trading_pair_quantity REAL NOT NULL,
                    trade_amount REAL NOT NULL,
                    entry_conditions TEXT NOT NULL,
                    side TEXT NOT NULL,
                    status TEXT NOT NULL,
                    strategy TEXT,
                    exit_time TIMESTAMP,
                    exit_price REAL,
                    exit_condition TEXT,
                    pnl REAL,
                    pnl_percentage REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create balance history table
            await self.connection.execute("""
                CREATE TABLE IF NOT EXISTS balance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    balance REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create PnL history table
            await self.connection.execute("""
                CREATE TABLE IF NOT EXISTS pnl_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    pnl REAL NOT NULL,
                    cumulative_pnl REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Commit changes
            await self.connection.commit()
            logger.info("Database tables initialized")
        except Exception as e:
            logger.error(f"Error initializing database tables: {str(e)}")
            raise
    
    async def close(self):
        """Close the database connection."""
        if self.connection:
            await self.connection.close()
            self.connection = None
            logger.info("Database connection closed")

    async def store_completed_trade(self, trade_result: Dict[str, Any]) -> bool:
        """
        Store a completed trade with all metrics in the database.
        
        Args:
            trade_result: Dictionary containing trade result data
            
        Returns:
            Boolean indicating success or failure
        """
        if not trade_result or not self.initialized:
            return False
            
        try:
            # Extract trade data
            symbol = trade_result.get("symbol")
            trade_id = trade_result.get("trade_id")
            
            if not symbol or not trade_id:
                logger.error("Cannot store trade without symbol and trade_id")
                return False
                
            # Check if the trade already exists in database
            async with self.session_factory() as session:
                # Check if trade exists
                existing_trade = await session.execute(
                    select(models.Trade).where(models.Trade.id == trade_id)
                )
                existing_trade = existing_trade.scalars().first()
                
                if existing_trade:
                    # Update existing trade
                    existing_trade.status = models.TradeStatus.CLOSED
                    existing_trade.exit_price = trade_result.get("exit_price")
                    existing_trade.exit_time = trade_result.get("exit_time") or datetime.now()
                    existing_trade.realized_pnl = trade_result.get("pnl")
                    existing_trade.realized_pnl_pct = trade_result.get("pnl_percentage")
                    existing_trade.updated_at = datetime.now()
                    existing_trade.note = trade_result.get("reason", "")
                    
                    # Add any other fields that need updating
                    await session.commit()
                    logger.info(f"Updated existing trade record for {symbol} with ID: {trade_id}")
                    
                else:
                    # Create new trade record
                    entry_time = trade_result.get("entry_time") or datetime.now()
                    trade = models.Trade(
                        exchange=trade_result.get("exchange", "binance"),
                        symbol=trade_result.get("symbol", ""),
                        trade_type=models.TradeType(trade_result.get("trade_type", "buy").lower()),
                        order_type=models.OrderType(trade_result.get("order_type", "market").lower()),
                        status=models.TradeStatus(trade_result.get("status", "open").lower()),
                        entry_price=trade_result.get("price", 0.0),
                        amount=trade_result.get("amount", 0.0),
                        fee=trade_result.get("fee", 0.0),
                        entry_time=entry_time,
                        stop_loss=trade_result.get("stop_loss"),
                        take_profit=trade_result.get("take_profit"),
                        strategy=trade_result.get("strategy"),
                        timeframe=trade_result.get("timeframe"),
                        simulation=trade_result.get("simulation", False)
                    )
                    
                    session.add(trade)
                    await session.commit()
                    logger.info(f"Stored new completed trade for {symbol} with ID: {trade_id}")
                
                # Also store the trade metrics for analysis
                await self.store_trade_metrics(trade_result, session)
                
            return True
            
        except Exception as e:
            logger.exception(f"Error storing completed trade: {str(e)}")
            return False
    
    async def store_trade_metrics(self, trade_result: Dict[str, Any], session = None) -> bool:
        """
        Store detailed trade metrics for post-trade analysis.
        
        Args:
            trade_result: Dictionary containing trade result data
            session: Optional SQLAlchemy session (reused if provided)
            
        Returns:
            Boolean indicating success or failure
        """
        if not trade_result:
            return False
            
        should_close_session = False
        try:
            # Create metrics record
            if not session:
                should_close_session = True
                session = self.session_factory()
                
            # Extract metrics data
            trade_id = trade_result.get("trade_id")
            if not trade_id:
                logger.error("Cannot store metrics without trade_id")
                return False
                
            # Check if we have TradeMetrics model (create it if needed)
            if not hasattr(models, "TradeMetrics"):
                # Create the model dynamically
                class TradeMetrics(models.Base):
                    __tablename__ = "trade_metrics"
                    
                    id = Column(Integer, primary_key=True, autoincrement=True)
                    trade_id = Column(Integer, ForeignKey("trades.id"), nullable=False)
                    execution_slippage = Column(Float, nullable=True)  # Actual vs expected entry price
                    exit_slippage = Column(Float, nullable=True)  # Actual vs expected exit price
                    market_volatility = Column(Float, nullable=True)  # ATR at entry
                    execution_time_ms = Column(Integer, nullable=True)  # Order execution time
                    market_impact = Column(Float, nullable=True)  # Price movement caused by order
                    entry_reason = Column(String(200), nullable=True)
                    exit_reason = Column(String(200), nullable=True)
                    ml_prediction_confidence = Column(Float, nullable=True)
                    technical_signal_strength = Column(Float, nullable=True)
                    higher_timeframe_aligned = Column(Boolean, nullable=True)
                    created_at = Column(DateTime, default=datetime.utcnow)
                    
                    # Add relationship to Trade
                    trade = relationship("Trade", backref="metrics")
                
                # Add to models module
                models.TradeMetrics = TradeMetrics
                
                # Create table if it doesn't exist
                model_metadata = Base.metadata
                async with self.engine.begin() as conn:
                    await conn.run_sync(lambda ctx: model_metadata.create_all(ctx))
            
            # Now store the metrics
            metrics = models.TradeMetrics(
                trade_id=trade_id,
                execution_slippage=trade_result.get("entry_slippage"),
                exit_slippage=trade_result.get("exit_slippage"),
                market_volatility=trade_result.get("market_volatility"),
                execution_time_ms=trade_result.get("execution_time_ms"),
                market_impact=trade_result.get("market_impact"),
                entry_reason=trade_result.get("entry_reason"),
                exit_reason=trade_result.get("exit_reason"),
                ml_prediction_confidence=trade_result.get("ml_confidence"),
                technical_signal_strength=trade_result.get("signal_strength"),
                higher_timeframe_aligned=trade_result.get("higher_tf_aligned")
            )
            
            session.add(metrics)
            await session.commit()
            logger.debug(f"Stored detailed metrics for trade {trade_id}")
            
            return True
            
        except Exception as e:
            logger.exception(f"Error storing trade metrics: {str(e)}")
            return False
            
        finally:
            # Close session if we created it
            if should_close_session and session:
                await session.close()
    
    async def add_ml_feedback(self, feedback_data: Dict[str, Any]) -> bool:
        """
        Store ML model feedback from trade results.
        
        Args:
            feedback_data: Dictionary with feedback data for ML training
            
        Returns:
            Boolean indicating success or failure
        """
        if not feedback_data or not self.initialized:
            return False
            
        try:
            # Check if we have MLFeedback model (create it if needed)
            if not hasattr(models, "MLFeedback"):
                # Create the model dynamically
                class MLFeedback(models.Base):
                    __tablename__ = "ml_feedback"
                    
                    id = Column(Integer, primary_key=True, autoincrement=True)
                    trade_id = Column(String(50), nullable=False, index=True)
                    symbol = Column(String(20), nullable=False)
                    entry_time = Column(DateTime, nullable=False)
                    exit_time = Column(DateTime, nullable=False)
                    entry_price = Column(Float, nullable=False)
                    exit_price = Column(Float, nullable=False)
                    side = Column(String(10), nullable=False)
                    pnl = Column(Float, nullable=False)
                    pnl_percentage = Column(Float, nullable=False)
                    duration_minutes = Column(Float, nullable=True)
                    exit_reason = Column(String(50), nullable=True)
                    strategy = Column(String(50), nullable=True)
                    success = Column(Boolean, nullable=False)
                    features_json = Column(String(5000), nullable=True)  # Store features as JSON
                    created_at = Column(DateTime, default=datetime.utcnow)
                
                # Add to models module
                models.MLFeedback = MLFeedback
                
                # Create table if it doesn't exist
                model_metadata = Base.metadata
                async with self.engine.begin() as conn:
                    await conn.run_sync(lambda ctx: model_metadata.create_all(ctx))
            
            # Feature serialization
            features_json = None
            if "features" in feedback_data:
                features_json = json.dumps(feedback_data["features"])
                
            # Create and store the feedback record
            async with self.session_factory() as session:
                feedback_record = models.MLFeedback(
                    trade_id=feedback_data.get("trade_id"),
                    symbol=feedback_data.get("symbol"),
                    entry_time=feedback_data.get("entry_time"),
                    exit_time=feedback_data.get("exit_time"),
                    entry_price=feedback_data.get("entry_price"),
                    exit_price=feedback_data.get("exit_price"),
                    side=feedback_data.get("side"),
                    pnl=feedback_data.get("pnl"),
                    pnl_percentage=feedback_data.get("pnl_percentage"),
                    duration_minutes=feedback_data.get("duration_minutes"),
                    exit_reason=feedback_data.get("exit_reason"),
                    strategy=feedback_data.get("strategy"),
                    success=feedback_data.get("success"),
                    features_json=features_json
                )
                
                session.add(feedback_record)
                await session.commit()
                
            logger.debug(f"Stored ML feedback for trade {feedback_data.get('trade_id')}")
            return True
            
        except Exception as e:
            logger.exception(f"Error storing ML feedback: {str(e)}")
            return False

    async def initialize(self, create_tables=False):
        """Initialize the database connection and create tables if needed."""
        try:
            logger.info(f"Initializing database with {self.db_path}")
            
            # Import models here to avoid circular imports
            from app.database import models
            
            # Determine the proper database URL with async driver
            if self.db_path.startswith('postgresql://'):
                database_url = self.db_path.replace('postgresql://', 'postgresql+asyncpg://')
                logger.info(f"Updated PostgreSQL URL to use asyncpg driver: {database_url}")
            elif self.db_path.startswith('postgresql+asyncpg://'):
                database_url = self.db_path
                logger.info(f"Using existing PostgreSQL URL with asyncpg driver: {database_url}")
            elif self.db_path.startswith('sqlite://'):
                database_url = self.db_path.replace('sqlite://', 'sqlite+aiosqlite://')
                logger.info(f"Updated SQLite URL to use aiosqlite driver: {database_url}")
            elif self.db_path.startswith('sqlite+aiosqlite://'):
                database_url = self.db_path
                logger.info(f"Using existing SQLite URL with aiosqlite driver: {database_url}")
            else:
                # Default to SQLite for file paths
                database_url = f"sqlite+aiosqlite:///{self.db_path}"
                logger.info(f"Using SQLite database at: {database_url}")
            
            # Install required drivers if needed
            try:
                if 'postgresql+asyncpg' in database_url:
                    import asyncpg
                    logger.info("Using asyncpg driver for PostgreSQL")
                elif 'sqlite+aiosqlite' in database_url:
                    import aiosqlite
                    logger.info("Using aiosqlite driver for SQLite")
            except ImportError as e:
                logger.error(f"Required driver not installed: {e}")
                logger.error("Please install missing drivers:")
                logger.error("  pip install asyncpg aiosqlite")
                return False
                
            # Create async engine
            self.engine = create_async_engine(database_url, echo=self.debug)
            self.async_session_factory = async_sessionmaker(
                bind=self.engine, 
                expire_on_commit=False,
                class_=AsyncSession
            )
            
            # Test connection with a simple query
            async with self.async_session_factory() as session:
                await session.execute(text("SELECT 1"))
                logger.info("Database connection successful")
            
            # Create tables if requested
            if create_tables:
                logger.info("Creating database tables")
                async with self.engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
                logger.info("Database tables created")
            else:
                logger.info("Using existing database structure (tables not created)")
            
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            traceback.print_exc()
            self.initialized = False
            return False

    async def get_recent_trades(self, limit: int = 20, include_simulated: bool = True) -> List[Dict[str, Any]]:
        """Get recent trades, both open and closed."""
        try:
            async with self.session_factory() as session:
                query = select(models.Trade).order_by(models.Trade.entry_time.desc())
                
                if not include_simulated:
                    query = query.where(models.Trade.simulation == False)
                    
                query = query.limit(limit)
                result = await session.execute(query)
                trades = result.scalars().all()
                
                return [
                    {
                        "id": trade.id,
                        "symbol": trade.symbol,
                        "trade_type": trade.trade_type.value,
                        "status": trade.status.value,
                        "entry_price": trade.entry_price,
                        "exit_price": trade.exit_price,
                        "amount": trade.amount,
                        "entry_time": trade.entry_time,
                        "exit_time": trade.exit_time,
                        "realized_pnl": trade.realized_pnl,
                        "realized_pnl_pct": trade.realized_pnl_pct,
                        "strategy": trade.strategy,
                        "simulation": trade.simulation
                    }
                    for trade in trades
                ]
        except Exception as e:
            logger.error(f"Error getting recent trades: {str(e)}")
            return []
        
    async def get_trade_metrics(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed metrics for a trade."""
        try:
            async with self.session_factory() as session:
                # Get the basic trade data
                query = select(models.Trade).where(models.Trade.id == trade_id)
                result = await session.execute(query)
                trade = result.scalar_one_or_none()
                
                if not trade:
                    return None
                    
                # Get associated metrics
                metrics_query = select(models.TradeMetrics).where(models.TradeMetrics.trade_id == trade_id)
                metrics_result = await session.execute(metrics_query)
                metrics = metrics_result.scalar_one_or_none()
                
                trade_data = {
                    "id": trade.id,
                    "symbol": trade.symbol,
                    "trade_type": trade.trade_type.value,
                    "status": trade.status.value,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "amount": trade.amount,
                    "entry_time": trade.entry_time,
                    "exit_time": trade.exit_time,
                    "stop_loss": trade.stop_loss,
                    "take_profit": trade.take_profit,
                    "realized_pnl": trade.realized_pnl,
                    "realized_pnl_pct": trade.realized_pnl_pct,
                    "strategy": trade.strategy,
                    "metrics": {}
                }
                
                if metrics:
                    trade_data["metrics"] = {
                        "market_conditions": metrics.market_conditions,
                        "entry_indicators": metrics.entry_indicators,
                        "exit_indicators": metrics.exit_indicators,
                        "risk_reward_ratio": metrics.risk_reward_ratio,
                        "market_volatility": metrics.market_volatility,
                        "drawdown": metrics.drawdown,
                        "holding_period": metrics.holding_period,
                        "exit_reason": metrics.exit_reason
                    }
                    
                return trade_data
        except Exception as e:
            logger.error(f"Error getting trade metrics: {str(e)}")
            return None

class TradeRepository:
    """
    Repository for trade-related database operations.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
    
    async def save_trade(self, trade_data: Dict[str, Any]) -> bool:
        """
        Save a new trade to the database.
        
        Args:
            trade_data: Dictionary containing trade information
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure we have a connection
            if not self.db_manager.connection:
                await self.db_manager.connect()
            
            # Generate a trade ID if not present
            if 'trade_id' not in trade_data:
                trade_data['trade_id'] = f"trade-{uuid.uuid4()}"
            
            # Convert entry_conditions to JSON string if it's a dict
            entry_conditions = trade_data.get('entry_conditions', {})
            if isinstance(entry_conditions, dict):
                entry_conditions = json.dumps(entry_conditions)
            
            # Convert exit_condition to JSON string if it's a dict
            exit_condition = trade_data.get('exit_condition', {})
            if isinstance(exit_condition, dict) and exit_condition:
                exit_condition = json.dumps(exit_condition)
            else:
                exit_condition = None
            
            # Handle the mapping between 'type' field (in code) and 'side' field (in database)
            # First try to get 'side', if not present use 'type'
            trade_type = trade_data.get('trade_type')
            if not trade_type:
                trade_type = trade_data.get('type', 'buy')  # Default to 'buy' if neither is present
            
            # Insert the trade
            await self.db_manager.connection.execute("""
                INSERT INTO trades (
                    trade_id, trading_pair, entry_time, trading_pair_price, 
                    trading_fee, trading_pair_quantity, trade_amount, 
                    entry_conditions, side, status, strategy,
                    exit_time, exit_price, exit_condition, pnl, pnl_percentage
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data.get('trade_id'),
                trade_data.get('trading_pair'),
                trade_data.get('entry_time'),
                trade_data.get('trading_pair_price'),
                trade_data.get('trading_fee'),
                trade_data.get('trading_pair_quantity'),
                trade_data.get('trade_amount'),
                entry_conditions,
                trade_type,  # Use the determined side
                trade_data.get('status', 'open'),
                trade_data.get('strategy'),
                trade_data.get('exit_time'),
                trade_data.get('exit_price'),
                exit_condition,
                trade_data.get('pnl'),
                trade_data.get('pnl_percentage')
            ))
            
            # Commit the transaction
            await self.db_manager.connection.commit()
            
            self.logger.info(f"Trade saved with ID: {trade_data.get('trade_id')}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving trade: {str(e)}")
            return False
    
    async def update_trade(self, trade_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update an existing trade with new information.
        
        Args:
            trade_id: ID of the trade to update
            update_data: Dictionary containing fields to update
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure we have a connection
            if not self.db_manager.connection:
                await self.db_manager.connect()
            
            # Convert exit_condition to JSON string if it's a dict
            if 'exit_condition' in update_data and isinstance(update_data['exit_condition'], dict):
                update_data['exit_condition'] = json.dumps(update_data['exit_condition'])
            
            # Build the SQL query dynamically based on the fields being updated
            query_parts = []
            params = []
            
            for key, value in update_data.items():
                if key not in ['trade_id']: # Don't update the primary key
                    query_parts.append(f"{key} = ?")
                    params.append(value)
            
            # Add the updated_at timestamp
            query_parts.append("updated_at = CURRENT_TIMESTAMP")
            
            # Add the trade_id to the params for the WHERE clause
            params.append(trade_id)
            
            # Execute the update
            query = f"UPDATE trades SET {', '.join(query_parts)} WHERE trade_id = ?"
            await self.db_manager.connection.execute(query, params)
            
            # Commit the transaction
            await self.db_manager.connection.commit()
            
            self.logger.info(f"Trade updated with ID: {trade_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating trade: {str(e)}")
            return False
    
    async def get_trade_by_id(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a trade by its ID.
        
        Args:
            trade_id: ID of the trade to retrieve
        
        Returns:
            Optional[Dict[str, Any]]: The trade data if found, None otherwise
        """
        try:
            # Ensure we have a connection
            if not self.db_manager.connection:
                await self.db_manager.connect()
            
            # Query the trade
            async with self.db_manager.connection.execute(
                "SELECT * FROM trades WHERE trade_id = ?", (trade_id,)
            ) as cursor:
                row = await cursor.fetchone()
                
                if row:
                    # Convert row to dictionary
                    columns = [desc[0] for desc in cursor.description]
                    trade = dict(zip(columns, row))
                    
                    # Parse JSON strings
                    if trade.get('entry_conditions'):
                        try:
                            trade['entry_conditions'] = json.loads(trade['entry_conditions'])
                        except:
                            pass
                    
                    if trade.get('exit_condition'):
                        try:
                            trade['exit_condition'] = json.loads(trade['exit_condition'])
                        except:
                            pass
                    
                    return trade
                
                return None
        except Exception as e:
            self.logger.error(f"Error getting trade by ID: {str(e)}")
            return None
    
    async def get_trades(self, status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get a list of trades, optionally filtered by status.
        
        Args:
            status: Filter trades by status (open, closed, etc.)
            limit: Maximum number of trades to return
        
        Returns:
            List[Dict[str, Any]]: List of trade data
        """
        try:
            # Ensure we have a connection
            if not self.db_manager.connection:
                await self.db_manager.connect()
            
            # Build the query
            query = "SELECT * FROM trades"
            params = []
            
            if status:
                query += " WHERE status = ?"
                params.append(status)
            
            query += " ORDER BY entry_time DESC LIMIT ?"
            params.append(limit)
            
            # Execute the query
            results = []
            async with self.db_manager.connection.execute(query, params) as cursor:
                async for row in cursor:
                    # Convert row to dictionary
                    columns = [desc[0] for desc in cursor.description]
                    trade = dict(zip(columns, row))
                    
                    # Parse JSON strings
                    if trade.get('entry_conditions'):
                        try:
                            trade['entry_conditions'] = json.loads(trade['entry_conditions'])
                        except:
                            pass
                    
                    if trade.get('exit_condition'):
                        try:
                            trade['exit_condition'] = json.loads(trade['exit_condition'])
                        except:
                            pass
                    
                    results.append(trade)
            
            return results
        except Exception as e:
            self.logger.error(f"Error getting trades: {str(e)}")
            return []
    
    async def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all open positions (trades with status 'open').
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of open positions keyed by trading pair
        """
        try:
            trades = await self.get_trades(status='open')
            positions = {}
            
            for trade in trades:
                trading_pair = trade.get('trading_pair')
                # Format as expected by the balance manager
                positions[trading_pair] = {
                    'symbol': trading_pair,
                    'side': trade.get('trade_type'),
                    'entry_price': trade.get('trading_pair_price'),
                    'current_price': trade.get('trading_pair_price'),  # Will be updated later
                    'amount': trade.get('trading_pair_quantity'),
                    'value': trade.get('trade_amount'),
                    'timestamp': trade.get('entry_time'),
                    'strategy': trade.get('strategy'),
                    'trade_id': trade.get('trade_id')
                }
            
            return positions
        except Exception as e:
            self.logger.error(f"Error getting open positions: {str(e)}")
            return {}
    
    async def get_open_trades(self) -> List[Dict[str, Any]]:
        """
        Get all open trades (trades with status 'open').
        
        Returns:
            List[Dict[str, Any]]: List of open trades
        """
        try:
            return await self.get_trades(status='open')
        except Exception as e:
            self.logger.error(f"Error getting open trades: {str(e)}")
            return []
    
    async def clear_all_trades(self) -> bool:
        """
        Delete all trades from the database, both active and historical.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure we have a connection
            if not self.db_manager.connection:
                await self.db_manager.connect()
            
            # Execute DELETE statement to remove all trades
            await self.db_manager.connection.execute("DELETE FROM trades")
            
            # Commit the transaction
            await self.db_manager.connection.commit()
            
            self.logger.info("All trades cleared from database")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing trades: {str(e)}")
            return False

class BalanceRepository:
    """
    Repository for balance-related database operations.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
    
    async def save_balance_history(self, timestamp: str, balance: float) -> bool:
        """
        Save a balance history record.
        
        Args:
            timestamp: ISO format timestamp
            balance: Balance amount
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure we have a connection
            if not self.db_manager.connection:
                await self.db_manager.connect()
            
            # Insert the balance record
            await self.db_manager.connection.execute(
                "INSERT INTO balance_history (timestamp, balance) VALUES (?, ?)",
                (timestamp, balance)
            )
            
            # Commit the transaction
            await self.db_manager.connection.commit()
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving balance history: {str(e)}")
            return False
    
    async def save_pnl_history(self, timestamp: str, pnl: float, cumulative_pnl: float) -> bool:
        """
        Save a PnL history record.
        
        Args:
            timestamp: ISO format timestamp
            pnl: Profit and loss amount
            cumulative_pnl: Cumulative profit and loss
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure we have a connection
            if not self.db_manager.connection:
                await self.db_manager.connect()
            
            # Insert the PnL record
            await self.db_manager.connection.execute(
                "INSERT INTO pnl_history (timestamp, pnl, cumulative_pnl) VALUES (?, ?, ?)",
                (timestamp, pnl, cumulative_pnl)
            )
            
            # Commit the transaction
            await self.db_manager.connection.commit()
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving PnL history: {str(e)}")
            return False
    
    async def get_balance_history(self, days: int = 90) -> List[Dict[str, Any]]:
        """
        Get balance history records.
        
        Args:
            days: Number of days of history to retrieve
        
        Returns:
            List[Dict[str, Any]]: List of balance history records
        """
        try:
            # Ensure we have a connection
            if not self.db_manager.connection:
                await self.db_manager.connect()
            
            # Query the balance history
            results = []
            async with self.db_manager.connection.execute(
                "SELECT timestamp, balance FROM balance_history ORDER BY timestamp DESC LIMIT ?",
                (days,)
            ) as cursor:
                async for row in cursor:
                    results.append({
                        "timestamp": row[0],
                        "balance": row[1]
                    })
            
            return results
        except Exception as e:
            self.logger.error(f"Error getting balance history: {str(e)}")
            return []
    
    async def get_pnl_history(self, days: int = 90) -> List[Dict[str, Any]]:
        """
        Get PnL history records.
        
        Args:
            days: Number of days of history to retrieve
        
        Returns:
            List[Dict[str, Any]]: List of PnL history records
        """
        try:
            # Ensure we have a connection
            if not self.db_manager.connection:
                await self.db_manager.connect()
            
            # Query the PnL history
            results = []
            async with self.db_manager.connection.execute(
                "SELECT timestamp, pnl, cumulative_pnl FROM pnl_history ORDER BY timestamp DESC LIMIT ?",
                (days,)
            ) as cursor:
                async for row in cursor:
                    results.append({
                        "timestamp": row[0],
                        "pnl": row[1],
                        "cumulative_pnl": row[2]
                    })
            
            return results
        except Exception as e:
            self.logger.error(f"Error getting PnL history: {str(e)}")
            return []

class MarketDataRepository:
    """
    Repository for market data operations.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
    
    # Methods for market data operations will be implemented as needed 

    async def store_trade(self, trade_result: Dict[str, Any]) -> Optional[int]:
        """Store a trade in the database."""
        if not trade_result or not isinstance(trade_result, dict):
            return None
        
        logger.info(f"Storing trade: {trade_result}")
        
        try:
            # Prepare the object
            trade_id = trade_result.get("id")
            symbol = trade_result.get("symbol")
            
            if not symbol:
                logger.error("Cannot store trade without symbol information")
                return None
            
            async with self.session_maker() as session:
                # Check if this trade exists
                if trade_id:
                    existing_trade = await session.execute(
                        select(models.Trade).where(models.Trade.id == trade_id)
                    )
                    existing_trade = existing_trade.scalar_one_or_none()
                    
                    if existing_trade:
                        # Update existing trade
                        if trade_result.get("exit_price"):
                            existing_trade.exit_price = trade_result.get("exit_price")
                            existing_trade.exit_time = trade_result.get("exit_time") or datetime.now()
                            existing_trade.status = models.TradeStatus.CLOSED
                            existing_trade.realized_pnl = trade_result.get("pnl")
                            existing_trade.realized_pnl_pct = trade_result.get("pnl_percentage")
                            existing_trade.note = trade_result.get("reason", "")
                            existing_trade.updated_at = datetime.now()
                            
                        await session.commit()
                        logger.info(f"Updated existing trade {trade_id}")
                        return trade_id
                        
                # Convert trade_type to proper enum format
                trade_type_str = trade_result.get("trade_type", "buy").lower()
                trade_type = models.TradeType.BUY if trade_type_str == "buy" else models.TradeType.SELL
                
                # Create a new trade record
                entry_time = trade_result.get("entry_time") or datetime.now()
                new_trade = models.Trade(
                    exchange=trade_result.get("exchange", self.default_exchange),
                    symbol=symbol,
                    trade_type=trade_type,
                    order_type=models.OrderType(trade_result.get("order_type", "market").lower()),
                    status=models.TradeStatus.OPEN,
                    entry_price=trade_result.get("price", 0.0),
                    amount=trade_result.get("amount", 0.0),
                    fee=trade_result.get("fee", 0.0),
                    entry_time=entry_time,
                    stop_loss=trade_result.get("stop_loss"),
                    take_profit=trade_result.get("take_profit"),
                    strategy=trade_result.get("strategy"),
                    timeframe=trade_result.get("timeframe"),
                    simulation=trade_result.get("simulation", False)
                )
                
                session.add(new_trade)
                await session.commit()
                await session.refresh(new_trade)
                
                # If we have a signal ID, update the signal with the trade ID
                signal_id = trade_result.get("signal_id")
                if signal_id:
                    signal = await session.execute(
                        select(models.Signal).where(models.Signal.id == signal_id)
                    )
                    signal = signal.scalar_one_or_none()
                    
                    if signal:
                        signal.trade_id = new_trade.id
                        signal.is_executed = True
                        await session.commit()
                
                logger.info(f"Created new trade with ID: {new_trade.id}")
                return new_trade.id
                
        except Exception as e:
            logger.exception(f"Error storing trade: {str(e)}")
            return None

    async def get_active_trades(self) -> List[Dict[str, Any]]:
        """Get all active trades."""
        try:
            async with self.session_maker() as session:
                query = select(models.Trade).where(models.Trade.status == models.TradeStatus.OPEN)
                result = await session.execute(query)
                trades = result.scalars().all()
                
                return [
                    {
                        "id": trade.id,
                        "symbol": trade.symbol,
                        "trade_type": trade.trade_type.value,
                        "entry_price": trade.entry_price,
                        "amount": trade.amount,
                        "entry_time": trade.entry_time,
                        "stop_loss": trade.stop_loss,
                        "take_profit": trade.take_profit,
                        "strategy": trade.strategy
                    }
                    for trade in trades
                ]
        except Exception as e:
            logger.error(f"Error getting active trades: {str(e)}")
            return []
            
    async def get_trade_by_id(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """Get a trade by ID."""
        try:
            async with self.session_maker() as session:
                query = select(models.Trade).where(models.Trade.id == trade_id)
                result = await session.execute(query)
                trade = result.scalar_one_or_none()
                
                if not trade:
                    return None
                    
                return {
                    "id": trade.id,
                    "symbol": trade.symbol,
                    "trade_type": trade.trade_type.value,
                    "status": trade.status.value,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "amount": trade.amount,
                    "entry_time": trade.entry_time,
                    "exit_time": trade.exit_time,
                    "stop_loss": trade.stop_loss,
                    "take_profit": trade.take_profit,
                    "realized_pnl": trade.realized_pnl,
                    "realized_pnl_pct": trade.realized_pnl_pct,
                    "strategy": trade.strategy
                }
        except Exception as e:
            logger.error(f"Error getting trade by ID: {str(e)}")
            return None 