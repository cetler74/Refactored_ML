import asyncio
import logging
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.config.settings import Settings
from app.exchange.manager import ExchangeManager

async def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    settings = Settings()
    exchange_manager = ExchangeManager(settings)
    await exchange_manager.initialize()
    
    pairs = await exchange_manager.fetch_usdc_trading_pairs()
    filtered_pairs = [p for p in pairs if any(token in p for token in ['PEPE', 'CHEEMS', 'CAT', 'SHIB', 'BONK'])]
    print(f'Found {len(filtered_pairs)} filtered pairs: {filtered_pairs}')
    
    # Test price fetching for a few prefixed tokens
    prices = await exchange_manager.fetch_current_prices(['1000CATUSDC', 'PEPEUSDC', 'SHIBUSDC', 'BONKUSDC'])
    print("\nPrices for tokens:")
    for pair, price in prices.items():
        print(f"{pair}: {price}")
    
    await exchange_manager.close()

if __name__ == "__main__":
    asyncio.run(main()) 