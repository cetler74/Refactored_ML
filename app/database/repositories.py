import logging
import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

# Set up logger
logger = logging.getLogger(__name__)

class TradingPairRepository:
    """
    Repository for trading pair related database operations.
    """
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
    
    async def save_trading_pair(self, pair_data: Dict[str, Any]) -> bool:
        """
        Save a trading pair to the database.
        
        Args:
            pair_data: Dictionary containing trading pair information
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure we have a connection
            if not self.db_manager.connection:
                await self.db_manager.connect()
            
            # Check if pair already exists
            async with self.db_manager.connection.execute(
                "SELECT * FROM trading_pairs WHERE symbol = ?", (pair_data.get('symbol'),)
            ) as cursor:
                existing_pair = await cursor.fetchone()
                
                if existing_pair:
                    # Update existing pair
                    await self.db_manager.connection.execute("""
                        UPDATE trading_pairs SET
                            base_asset = ?,
                            quote_asset = ?,
                            price_precision = ?,
                            quantity_precision = ?,
                            min_qty = ?,
                            max_qty = ?,
                            min_notional = ?,
                            status = ?,
                            last_updated = ?
                        WHERE symbol = ?
                    """, (
                        pair_data.get('base_asset'),
                        pair_data.get('quote_asset'),
                        pair_data.get('price_precision'),
                        pair_data.get('quantity_precision'),
                        pair_data.get('min_qty'),
                        pair_data.get('max_qty'),
                        pair_data.get('min_notional'),
                        pair_data.get('status', 'active'),
                        datetime.now().isoformat(),
                        pair_data.get('symbol')
                    ))
                else:
                    # Insert new pair
                    await self.db_manager.connection.execute("""
                        INSERT INTO trading_pairs (
                            symbol, base_asset, quote_asset, price_precision,
                            quantity_precision, min_qty, max_qty, min_notional, 
                            status, last_updated
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pair_data.get('symbol'),
                        pair_data.get('base_asset'),
                        pair_data.get('quote_asset'),
                        pair_data.get('price_precision'),
                        pair_data.get('quantity_precision'),
                        pair_data.get('min_qty'),
                        pair_data.get('max_qty'),
                        pair_data.get('min_notional'),
                        pair_data.get('status', 'active'),
                        datetime.now().isoformat()
                    ))
            
            # Commit the transaction
            await self.db_manager.connection.commit()
            
            self.logger.info(f"Trading pair saved: {pair_data.get('symbol')}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving trading pair: {str(e)}")
            return False
    
    async def get_trading_pair(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get a trading pair by symbol.
        
        Args:
            symbol: Symbol of the trading pair to retrieve
        
        Returns:
            Optional[Dict[str, Any]]: Trading pair data if found, None otherwise
        """
        try:
            # Ensure we have a connection
            if not self.db_manager.connection:
                await self.db_manager.connect()
            
            # Query the trading pair
            async with self.db_manager.connection.execute(
                "SELECT * FROM trading_pairs WHERE symbol = ?", (symbol,)
            ) as cursor:
                row = await cursor.fetchone()
                
                if row:
                    # Convert row to dictionary
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, row))
                
                return None
        except Exception as e:
            self.logger.error(f"Error getting trading pair: {str(e)}")
            return None
    
    async def get_all_active_pairs(self) -> List[Dict[str, Any]]:
        """
        Get all active trading pairs.
        
        Returns:
            List[Dict[str, Any]]: List of active trading pairs
        """
        try:
            # Ensure we have a connection
            if not self.db_manager.connection:
                await self.db_manager.connect()
            
            # Query active trading pairs
            results = []
            async with self.db_manager.connection.execute(
                "SELECT * FROM trading_pairs WHERE status = 'active'"
            ) as cursor:
                async for row in cursor:
                    # Convert row to dictionary
                    columns = [desc[0] for desc in cursor.description]
                    results.append(dict(zip(columns, row)))
            
            return results
        except Exception as e:
            self.logger.error(f"Error getting active trading pairs: {str(e)}")
            return []
    
    async def get_trading_pairs_by_quote(self, quote_asset: str) -> List[Dict[str, Any]]:
        """
        Get trading pairs by quote asset.
        
        Args:
            quote_asset: Quote asset to filter by (e.g., 'USDT')
        
        Returns:
            List[Dict[str, Any]]: List of trading pairs with the specified quote asset
        """
        try:
            # Ensure we have a connection
            if not self.db_manager.connection:
                await self.db_manager.connect()
            
            # Query trading pairs with the specified quote asset
            results = []
            async with self.db_manager.connection.execute(
                "SELECT * FROM trading_pairs WHERE quote_asset = ? AND status = 'active'",
                (quote_asset,)
            ) as cursor:
                async for row in cursor:
                    # Convert row to dictionary
                    columns = [desc[0] for desc in cursor.description]
                    results.append(dict(zip(columns, row)))
            
            return results
        except Exception as e:
            self.logger.error(f"Error getting trading pairs by quote asset: {str(e)}")
            return []

    async def update_pair_status(self, symbol: str, status: str) -> bool:
        """
        Update the status of a trading pair.
        
        Args:
            symbol: Symbol of the trading pair to update
            status: New status ('active', 'inactive', etc.)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure we have a connection
            if not self.db_manager.connection:
                await self.db_manager.connect()
            
            # Update the trading pair status
            await self.db_manager.connection.execute(
                "UPDATE trading_pairs SET status = ?, last_updated = ? WHERE symbol = ?",
                (status, datetime.now().isoformat(), symbol)
            )
            
            # Commit the transaction
            await self.db_manager.connection.commit()
            
            self.logger.info(f"Trading pair status updated: {symbol} -> {status}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating trading pair status: {str(e)}")
            return False

class SignalRepository:
    """
    Repository for signal-related database operations.
    """
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
    
    async def save_signal(self, signal_data: Dict[str, Any]) -> bool:
        """
        Save a signal to the database.
        
        Args:
            signal_data: Dictionary containing signal information
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure we have a connection
            if not self.db_manager.connection:
                await self.db_manager.connect()
            
            # Generate a signal ID if not present
            if 'signal_id' not in signal_data:
                signal_data['signal_id'] = f"signal-{uuid.uuid4()}"
            
            # Convert entry_conditions to JSON string if it's a dict
            entry_conditions = signal_data.get('entry_conditions', {})
            if isinstance(entry_conditions, dict):
                entry_conditions = json.dumps(entry_conditions)
            
            # Insert the signal
            await self.db_manager.connection.execute("""
                INSERT INTO signals (
                    signal_id, symbol, timestamp, trade_type, price, 
                    confidence, strategy, timeframe, entry_conditions,
                    status, executed, executed_time, trade_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_data.get('signal_id'),
                signal_data.get('symbol'),
                signal_data.get('timestamp'),
                signal_data.get('trade_type'),
                signal_data.get('price'),
                signal_data.get('confidence'),
                signal_data.get('strategy'),
                signal_data.get('timeframe'),
                entry_conditions,
                signal_data.get('status', 'pending'),
                signal_data.get('executed', False),
                signal_data.get('executed_time'),
                signal_data.get('trade_id')
            ))
            
            # Commit the transaction
            await self.db_manager.connection.commit()
            
            self.logger.info(f"Signal saved with ID: {signal_data.get('signal_id')}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving signal: {str(e)}")
            return False
    
    async def get_signal_by_id(self, signal_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a signal by its ID.
        
        Args:
            signal_id: ID of the signal to retrieve
        
        Returns:
            Optional[Dict[str, Any]]: The signal data if found, None otherwise
        """
        try:
            # Ensure we have a connection
            if not self.db_manager.connection:
                await self.db_manager.connect()
            
            # Query the signal
            async with self.db_manager.connection.execute(
                "SELECT * FROM signals WHERE signal_id = ?", (signal_id,)
            ) as cursor:
                row = await cursor.fetchone()
                
                if row:
                    # Convert row to dictionary
                    columns = [desc[0] for desc in cursor.description]
                    signal = dict(zip(columns, row))
                    
                    # Parse JSON strings
                    if signal.get('entry_conditions'):
                        try:
                            signal['entry_conditions'] = json.loads(signal['entry_conditions'])
                        except:
                            pass
                    
                    return signal
                
                return None
        except Exception as e:
            self.logger.error(f"Error getting signal by ID: {str(e)}")
            return None
    
    async def update_signal_status(self, signal_id: str, status: str, trade_id: str = None) -> bool:
        """
        Update the status of a signal.
        
        Args:
            signal_id: ID of the signal to update
            status: New status ('executed', 'rejected', etc.)
            trade_id: Associated trade ID if executed
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure we have a connection
            if not self.db_manager.connection:
                await self.db_manager.connect()
            
            # Build the query
            query = "UPDATE signals SET status = ?"
            params = [status]
            
            # Add executed status and time if status is 'executed'
            if status == 'executed':
                query += ", executed = ?, executed_time = ?"
                params.extend([True, datetime.now().isoformat()])
                
                # Add trade ID if provided
                if trade_id:
                    query += ", trade_id = ?"
                    params.append(trade_id)
            
            # Add WHERE clause
            query += " WHERE signal_id = ?"
            params.append(signal_id)
            
            # Execute the query
            await self.db_manager.connection.execute(query, params)
            
            # Commit the transaction
            await self.db_manager.connection.commit()
            
            self.logger.info(f"Signal status updated: {signal_id} -> {status}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating signal status: {str(e)}")
            return False
    
    async def get_recent_signals(self, limit: int = 100, status: str = None) -> List[Dict[str, Any]]:
        """
        Get recent signals, optionally filtered by status.
        
        Args:
            limit: Maximum number of signals to return
            status: Filter signals by status
        
        Returns:
            List[Dict[str, Any]]: List of signal data
        """
        try:
            # Ensure we have a connection
            if not self.db_manager.connection:
                await self.db_manager.connect()
            
            # Build the query
            query = "SELECT * FROM signals"
            params = []
            
            if status:
                query += " WHERE status = ?"
                params.append(status)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            # Execute the query
            results = []
            async with self.db_manager.connection.execute(query, params) as cursor:
                async for row in cursor:
                    # Convert row to dictionary
                    columns = [desc[0] for desc in cursor.description]
                    signal = dict(zip(columns, row))
                    
                    # Parse JSON strings
                    if signal.get('entry_conditions'):
                        try:
                            signal['entry_conditions'] = json.loads(signal['entry_conditions'])
                        except:
                            pass
                    
                    results.append(signal)
            
            return results
        except Exception as e:
            self.logger.error(f"Error getting recent signals: {str(e)}")
            return [] 