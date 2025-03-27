#!/usr/bin/env python3
"""
Simple test script to verify that the exchange manager methods are working properly.
"""

import asyncio
import sys
import logging
from app.config.settings import Settings
from app.exchange.manager import ExchangeManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_exchange_manager():
    """Test the exchange manager methods."""
    try:
        logger.info("Creating settings")
        settings = Settings()
        
        logger.info("Creating exchange manager")
        exchange_manager = ExchangeManager(settings)
        
        logger.info("Initializing exchange manager")
        await exchange_manager.initialize()
        
        # Test fetch_usdc_pairs
        try:
            logger.info("Testing fetch_usdc_pairs")
            pairs = await exchange_manager.fetch_usdc_pairs()
            logger.info(f"fetch_usdc_pairs returned {len(pairs)} pairs: {pairs[:5]} ...")
            
            # Test fetch_usdc_trading_pairs
            logger.info("Testing fetch_usdc_trading_pairs")
            trading_pairs = await exchange_manager.fetch_usdc_trading_pairs()
            logger.info(f"fetch_usdc_trading_pairs returned {len(trading_pairs)} pairs: {trading_pairs[:5]} ...")
            
            # Test fetch_current_prices
            logger.info("Testing fetch_current_prices")
            prices = await exchange_manager.fetch_current_prices(trading_pairs[:5])
            logger.info(f"fetch_current_prices returned {len(prices)} prices")
            
            logger.info("All tests passed!")
            return True
        except Exception as e:
            logger.error(f"Error testing exchange manager methods: {str(e)}")
            return False
        finally:
            logger.info("Closing exchange manager")
            await exchange_manager.close()
    except Exception as e:
        logger.error(f"Error creating exchange manager: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        result = asyncio.run(test_exchange_manager())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("Test interrupted")
        sys.exit(1) 