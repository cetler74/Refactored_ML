import logging
import asyncio
import traceback
from app.core.orchestrator import BotOrchestrator
from app.config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

async def test_market_data():
    try:
        # Initialize orchestrator
        settings = get_settings()
        orchestrator = BotOrchestrator(settings)
        
        # Initialize modules
        print("Initializing modules...")
        await orchestrator.initialize_modules()
        
        # Select trading pairs
        print("Selecting trading pairs...")
        pairs = await orchestrator.select_trading_pairs()
        print(f"Selected pairs: {pairs}")
        
        # Fetch OHLCV data for a single pair
        if pairs:
            try:
                print(f"Fetching OHLCV for {pairs[0]}...")
                symbol = pairs[0]
                timeframe = "1h"  # Use a larger timeframe for testing
                ohlcv_data = await orchestrator.exchange_manager.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=20  # Limit to 20 rows for readability
                )
                
                if ohlcv_data is not None and not ohlcv_data.empty:
                    print(f"OHLCV data for {symbol} ({timeframe}):")
                    print(ohlcv_data.head(5))
                    
                    # Test calculating indicators
                    print("Calculating indicators...")
                    try:
                        if orchestrator.ta_module:
                            ohlcv_with_indicators = await orchestrator.ta_module.calculate_indicators(ohlcv_data)
                            print("Indicator calculation succeeded")
                            print(f"Added indicators: {set(ohlcv_with_indicators.columns) - set(ohlcv_data.columns)}")
                    except Exception as e:
                        print(f"Error calculating indicators: {str(e)}")
                        traceback.print_exc()
                else:
                    print(f"No OHLCV data available for {symbol} ({timeframe})")
            except Exception as e:
                print(f"Error fetching OHLCV data: {str(e)}")
                traceback.print_exc()
        
        # Collect market data for all pairs
        print("\nCollecting market data for all pairs...")
        try:
            result = await orchestrator.collect_market_data()
            
            # Print result
            print(f"Market data collection result: {result['success']}")
            if result['success']:
                print(f"Collected data for {len(result['data'])} symbols:")
                for symbol, timeframes in result['data'].items():
                    print(f"Symbol: {symbol}")
                    for timeframe, data in timeframes.items():
                        print(f"  Timeframe: {timeframe}, Rows: {len(data) if data is not None else 0}")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"Error collecting market data: {str(e)}")
            traceback.print_exc()
            
    except Exception as e:
        print(f"Error in test: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_market_data()) 