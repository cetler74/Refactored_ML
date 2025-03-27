#!/usr/bin/env python3
"""
Test script for running the trading bot with full implementations.
"""

import os
import sys
import logging
import asyncio
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """Main function to test the full implementation."""
    try:
        print("Starting test with full implementations...")
        logger.info("Initializing all modules with real implementations...")
        
        print("Importing necessary modules...")
        # Import necessary modules
        from app.config.settings import Settings
        
        # Load settings
        print("Loading settings...")
        settings = Settings()
        
        # Initialize modules one by one with try/except blocks
        try:
            print("Initializing exchange manager...")
            from app.exchange.manager import ExchangeManager
            exchange_manager = ExchangeManager(settings=settings)
            await exchange_manager.initialize()
            print("Exchange manager initialized successfully")
        except Exception as e:
            print(f"Error initializing exchange manager: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return
        
        try:
            print("Initializing database manager...")
            from app.database.manager import DatabaseManager
            db_manager = DatabaseManager(db_path="data/trading_bot.db", debug=True)
            await db_manager.initialize()
            print("Database manager initialized successfully")
        except Exception as e:
            print(f"Error initializing database manager: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return
        
        try:
            print("Initializing trading pair selector...")
            from app.core.pair_selector import TradingPairSelector
            trading_pair_selector = TradingPairSelector(
                exchange_manager=exchange_manager,
                settings=settings
            )
            print("Trading pair selector initialized successfully")
        except Exception as e:
            print(f"Error initializing trading pair selector: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return
        
        try:
            print("Initializing technical analysis module...")
            from app.ta.indicators import TechnicalAnalysisModule as TAModule
            ta_module = TAModule()
            print("Technical analysis module initialized successfully")
        except Exception as e:
            print(f"Error initializing TA module: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return
        
        try:
            print("Initializing machine learning module...")
            from app.ml.model import MLModule
            ml_module = MLModule(settings=settings, db_manager=db_manager)
            print("Machine learning module initialized successfully")
        except Exception as e:
            print(f"Error initializing ML module: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return
        
        try:
            print("Initializing strategy module...")
            from app.strategies.strategy import StrategyModule
            strategy_module = StrategyModule(
                settings=settings,
                ta_module=ta_module,
                ml_module=ml_module
            )
            print("Strategy module initialized successfully")
        except Exception as e:
            print(f"Error initializing strategy module: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return
        
        # Test trading pair selection
        print("Testing trading pair selection...")
        try:
            selected_pairs = await trading_pair_selector.select_pairs()
            print(f"Selected trading pairs: {selected_pairs}")
        except Exception as e:
            print(f"Error selecting trading pairs: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return
        
        print("Test completed successfully")
        
    except Exception as e:
        print(f"Error in test: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main()) 