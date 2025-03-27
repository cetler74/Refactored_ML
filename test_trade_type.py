#!/usr/bin/env python
"""
Integration test to verify the trade_type implementation
"""
import os
import sys
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import application modules
from app.config.settings import Settings
from app.database.models import TradeType, Trade, Signal
from app.database.manager import DatabaseManager
from app.strategies.strategy import StrategyModule
from app.exchange.manager import ExchangeManager

async def test_trade_type_implementation():
    """Test the trade_type implementation across the system"""
    logger.info("Testing trade_type implementation...")
    
    # Initialize settings
    settings = Settings()
    
    # Initialize database manager
    db_manager = DatabaseManager(settings)
    await db_manager.initialize()
    
    # Initialize exchange manager
    exchange_manager = ExchangeManager(settings)
    await exchange_manager.initialize()
    
    # Test creating a signal with trade_type
    logger.info("1. Testing signal creation with trade_type...")
    signal = {
        'symbol': 'BTC/USDC',
        'trade_type': TradeType.BUY.value,
        'price': 85000.0,
        'confidence': 0.8,
        'timestamp': datetime.now(),
        'timeframe': '1h',
        'strategy': 'TestStrategy'
    }
    
    # Store signal in database
    signal_id = await db_manager.store_signal(signal)
    logger.info(f"Created signal with ID: {signal_id}")
    
    # Test retrieving the signal
    if signal_id:
        db_signal = await db_manager.get_signal(signal_id)
        logger.info(f"Retrieved signal: {db_signal}")
        assert db_signal['trade_type'] == TradeType.BUY.value, f"Expected trade_type to be 'buy', got {db_signal.get('trade_type')}"
    
    # Test creating a trade with trade_type
    logger.info("2. Testing trade creation with trade_type...")
    trade = {
        'symbol': 'BTC/USDC',
        'trade_type': TradeType.BUY.value,
        'order_type': 'market',
        'price': 85000.0,
        'amount': 0.01,
        'status': 'open',
        'strategy': 'TestStrategy',
        'timeframe': '1h',
        'signal_id': signal_id
    }
    
    # Store trade in database
    trade_id = await db_manager.store_trade(trade)
    logger.info(f"Created trade with ID: {trade_id}")
    
    # Test retrieving the trade
    if trade_id:
        db_trade = await db_manager.trade_repository.get_trade_by_id(trade_id)
        logger.info(f"Retrieved trade: {db_trade}")
        assert db_trade['trade_type'] == TradeType.BUY.value, f"Expected trade_type to be 'buy', got {db_trade.get('trade_type')}"
    
    # Test creating an order with trade_type
    logger.info("3. Testing order creation with trade_type...")
    try:
        order_result = await exchange_manager.create_order(
            symbol='BTC/USDC',
            type='market',
            trade_type='buy',
            amount=0.001,
            price=85000.0
        )
        logger.info(f"Created order: {order_result}")
        assert 'trade_type' in order_result, f"Expected 'trade_type' field in order result, got keys: {list(order_result.keys())}"
    except Exception as e:
        logger.error(f"Error creating order: {str(e)}")
    
    # Clean up test data
    if trade_id:
        await db_manager.execute_query(f"DELETE FROM trades WHERE id = {trade_id}")
    if signal_id:
        await db_manager.execute_query(f"DELETE FROM signals WHERE id = {signal_id}")
    
    # Close connections
    await exchange_manager.close()
    await db_manager.close()
    
    logger.info("Testing completed!")

if __name__ == "__main__":
    logger.info("Starting trade_type implementation test")
    asyncio.run(test_trade_type_implementation()) 