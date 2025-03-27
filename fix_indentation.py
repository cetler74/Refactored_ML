#!/usr/bin/env python
"""Fix indentation errors in the manager.py file"""

def main():
    with open('app/exchange/manager.py.bak', 'r') as f:
        content = f.read()

    # Fix indentation for initialize method
    content = content.replace('    async def initialize(self):\n        """Initialize connections to configured exchanges."""\n        # Get credentials from settings\n            credentials = self.exchange_credentials', 
                             '    async def initialize(self):\n        """Initialize connections to configured exchanges."""\n        # Get credentials from settings\n        credentials = self.exchange_credentials')

    # Fix indentation for trading mode options
    content = content.replace('            # Add trading mode to options if needed\n                    trading_mode = binance_config.trading_mode', 
                             '            # Add trading mode to options if needed\n            trading_mode = binance_config.trading_mode')

    # Fix indentation for fetch_ticker method
    content = content.replace('        if exchange_id not in self.exchanges:\n            logger.error(f"Exchange {exchange_id} not initialized")\n                return {}', 
                             '        if exchange_id not in self.exchanges:\n            logger.error(f"Exchange {exchange_id} not initialized")\n            return {}')

    # Fix indentation for fetch_usdc_pairs method
    content = content.replace('        if exchange_id not in self.exchanges:\n            logger.error(f"Exchange {exchange_id} not initialized")\n                return []', 
                             '        if exchange_id not in self.exchanges:\n            logger.error(f"Exchange {exchange_id} not initialized")\n            return []')

    # Fix indentation for close method
    content = content.replace('            for name, exchange in self.exchanges.items():\n                try:\n                    logger.info(f"Closing exchange connection: {name}")\n                    if hasattr(exchange, \'close\'):\n                await exchange.close()', 
                             '            for name, exchange in self.exchanges.items():\n                try:\n                    logger.info(f"Closing exchange connection: {name}")\n                    if hasattr(exchange, \'close\'):\n                        await exchange.close()')

    # Fix indentation for load markets
    content = content.replace('            for exchange_id, exchange in self.exchanges.items():\n                if hasattr(exchange, \'load_markets\'):\n                max_retries = 3', 
                             '            for exchange_id, exchange in self.exchanges.items():\n                if hasattr(exchange, \'load_markets\'):\n                    max_retries = 3')

    # Fix indentation for fetch_ohlcv method - DataFrame handling
    content = content.replace('            df[\'timestamp\'] = pd.to_datetime(df[\'timestamp\'], unit=\'ms\')\n            df.set_index(\'timestamp\', inplace=True)\n            return df\n                else:', 
                             '            df[\'timestamp\'] = pd.to_datetime(df[\'timestamp\'], unit=\'ms\')\n            df.set_index(\'timestamp\', inplace=True)\n            return df\n            else:')

    # Fix indentation for direct API close
    content = content.replace('            try:\n                        if hasattr(api, \'close\'):', 
                             '                    try:\n                        if hasattr(api, \'close\'):')

    # Fix unexpected unindent
    content = content.replace('                        if retry == max_retries - 1:\n                        logger.info(f"Continuing with {exchange_id} initialization despite market loading error")', 
                             '                        if retry == max_retries - 1:\n                            logger.info(f"Continuing with {exchange_id} initialization despite market loading error")')

    # Fix try statement with no except or finally
    content = content.replace('        try:\n            # Try direct API first', 
                             '        try:\n            # Try direct API first')

    # Write the fixed content back to a new file
    with open('app/exchange/manager.py', 'w') as f:
        f.write(content)

    print('File fixed and saved.')

if __name__ == "__main__":
    main() 