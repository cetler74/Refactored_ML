#!/usr/bin/env python3
import asyncio
import logging
import sys
import os

# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set specific modules to DEBUG
logging.getLogger('app.exchange.manager').setLevel(logging.DEBUG)

# Import after logging setup
from app.config.settings import get_settings
from app.exchange.manager import ExchangeManager

async def test_fetch_current_prices():
    """Test fetch_current_prices method for all USDC pairs to find NoneType errors"""
    try:
        # Initialize settings and exchange manager
        settings = get_settings()
        
        # Test with sandbox mode
        exchange_manager = ExchangeManager(settings, sandbox=True)
        await exchange_manager.initialize()
        
        # Get all USDC trading pairs
        logger.info("Fetching all USDC trading pairs")
        usdc_pairs = await exchange_manager.fetch_usdc_trading_pairs()
        logger.info(f"Found {len(usdc_pairs)} USDC trading pairs")
        
        # Take only a few pairs for testing
        test_pairs = usdc_pairs[:20]
        logger.info(f"Testing with {len(test_pairs)} pairs: {test_pairs}")
        
        # Add specific pairs that might cause the error
        problem_pairs = ["WAVES/USDC", "BCHABC/USDC", "BSV/USDC", "USDS/USDC", "BTT/USDC"]
        for pair in problem_pairs:
            if pair not in test_pairs:
                test_pairs.append(pair)
        
        # Fetch current prices for these pairs
        logger.info("Fetching current prices")
        prices = await exchange_manager.fetch_current_prices(test_pairs)
        
        # Check results
        for pair, data in prices.items():
            if 'error' in data:
                logger.error(f"Error fetching price for {pair}: {data['error']}")
            else:
                logger.info(f"Successfully fetched price for {pair}: {data['price']}")
        
        # Close the exchange manager
        await exchange_manager.close()
        
    except Exception as e:
        logger.exception(f"Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_fetch_current_prices()) 