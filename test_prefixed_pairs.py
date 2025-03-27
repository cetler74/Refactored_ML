"""
Test the fetch_usdc_pairs method with numeric prefixed tokens
"""

import asyncio
import logging
import sys
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config.settings import Settings
from app.exchange.manager import ExchangeManager

async def test_fetch_usdc_pairs():
    try:
        print("Testing fetch_usdc_pairs with numeric prefixed tokens")
        
        # Create settings
        settings = Settings()
        
        # Create exchange manager
        exchange_manager = ExchangeManager(settings)
        await exchange_manager.initialize()
        
        # Fetch USDC pairs
        usdc_pairs = await exchange_manager.fetch_usdc_pairs()
        
        # Check if PEPE, SHIB, BONK are included
        print(f"\nFound {len(usdc_pairs)} USDC pairs")
        
        pepe_pairs = [p for p in usdc_pairs if 'PEPE' in p]
        shib_pairs = [p for p in usdc_pairs if 'SHIB' in p]
        bonk_pairs = [p for p in usdc_pairs if 'BONK' in p]
        
        print(f"\nPEPE pairs: {pepe_pairs}")
        print(f"SHIB pairs: {shib_pairs}")
        print(f"BONK pairs: {bonk_pairs}")
        
        # Check for prefixed tokens
        prefixed_pairs = [p for p in usdc_pairs if p.startswith('1000')]
        print(f"\nPrefixed pairs (should be filtered out if non-prefixed versions exist):")
        for pair in prefixed_pairs:
            print(f"  {pair}")
        
        # Now test fetch_usdc_trading_pairs to make sure the format is consistent
        trading_pairs = await exchange_manager.fetch_usdc_trading_pairs()
        print(f"\nFound {len(trading_pairs)} trading pairs with / format")
        
        # Test fetch_current_prices to ensure we can get prices for these pairs
        test_pairs = trading_pairs[:5]  # Just test the first 5 pairs
        print(f"\nTesting fetch_current_prices with pairs: {test_pairs}")
        prices = await exchange_manager.fetch_current_prices(test_pairs)
        
        for pair, data in prices.items():
            if 'error' in data and data['error']:
                print(f"Error fetching price for {pair}: {data['error']}")
            else:
                print(f"{pair}: Price = {data['price']}, Volume = {data['volume_24h']}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await exchange_manager.close()

if __name__ == "__main__":
    asyncio.run(test_fetch_usdc_pairs()) 