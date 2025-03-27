#!/usr/bin/env python3
"""Test script for initializing all main modules"""

import asyncio
import logging
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def test_modules():
    """Test initialization of all main modules."""
    success = True
    error_messages = []
    
    # Import settings
    try:
        from app.config.settings import get_settings, Settings
        settings = get_settings()
        logger.info("Settings loaded successfully")
    except Exception as e:
        logger.error(f"Error loading settings: {e}")
        import traceback
        traceback.print_exc()
        error_messages.append(f"Settings error: {str(e)}")
        success = False
        return "Error in settings initialization"
    
    # Initialize database manager
    try:
        from app.database.manager import DatabaseManager
        db_manager = DatabaseManager(settings.database_url)
        db_init = await db_manager.initialize(create_tables=False)
        logger.info(f"Database initialized: {db_init}")
        if not db_init:
            success = False
            error_messages.append("Database initialization failed")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        import traceback
        traceback.print_exc()
        error_messages.append(f"Database error: {str(e)}")
        success = False
        db_manager = None
    
    # Initialize exchange manager
    try:
        from app.exchange.manager import ExchangeManager
        exchange_manager = ExchangeManager(settings, sandbox=True)
        logger.info("Exchange manager initialized")
    except Exception as e:
        logger.error(f"Error initializing exchange manager: {e}")
        import traceback
        traceback.print_exc()
        error_messages.append(f"Exchange manager error: {str(e)}")
        success = False
        exchange_manager = None
    
    # Initialize TA module
    try:
        from app.ta.indicators import TechnicalAnalysisModule
        ta_module = TechnicalAnalysisModule(settings)
        logger.info("TA module initialized")
    except Exception as e:
        logger.error(f"Error initializing TA module: {e}")
        import traceback
        traceback.print_exc()
        error_messages.append(f"TA module error: {str(e)}")
        success = False
        ta_module = None
    
    # Initialize ML module
    try:
        from app.ml.model import MLModule
        ml_module = MLModule(settings, db_manager)
        logger.info("ML module initialized")
    except Exception as e:
        logger.error(f"Error initializing ML module: {e}")
        import traceback
        traceback.print_exc()
        error_messages.append(f"ML module error: {str(e)}")
        success = False
        ml_module = None
    
    # Initialize strategy module
    try:
        if ta_module is not None and ml_module is not None:
            from app.strategies.strategy import StrategyModule
            strategy_module = StrategyModule(settings, ta_module, ml_module)
            logger.info("Strategy module initialized")
        else:
            logger.warning("Skipping strategy module initialization due to missing dependencies")
            error_messages.append("Strategy module not initialized due to missing dependencies")
    except Exception as e:
        logger.error(f"Error initializing strategy module: {e}")
        import traceback
        traceback.print_exc()
        error_messages.append(f"Strategy module error: {str(e)}")
        success = False
    
    # Initialize balance manager
    try:
        if exchange_manager is not None and db_manager is not None:
            from app.utils.balance_manager import BalanceManager
            balance_manager = BalanceManager(settings, exchange_manager, settings.APP_MODE, db_manager)
            await balance_manager.initialize()
            logger.info("Balance manager initialized")
        else:
            logger.warning("Skipping balance manager initialization due to missing dependencies")
            error_messages.append("Balance manager not initialized due to missing dependencies")
    except Exception as e:
        logger.error(f"Error initializing balance manager: {e}")
        import traceback
        traceback.print_exc()
        error_messages.append(f"Balance manager error: {str(e)}")
        success = False
    
    # Test orchestrator import (but don't initialize it yet)
    try:
        from app.core.orchestrator import BotOrchestrator
        logger.info("Orchestrator imported successfully")
    except Exception as e:
        logger.error(f"Error importing orchestrator: {e}")
        import traceback
        traceback.print_exc()
        error_messages.append(f"Orchestrator import error: {str(e)}")
        success = False
    
    if success:
        return "All modules initialized successfully"
    else:
        return f"Errors occurred: {', '.join(error_messages)}"

if __name__ == "__main__":
    result = asyncio.run(test_modules())
    print(f"\nFinal result: {result}") 