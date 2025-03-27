import logging

class TradingPairSelector:
    def __init__(self, exchange_manager, settings):
        """Initialize the trading pair selector."""
        self.exchange_manager = exchange_manager
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.selected_pairs = []
        self.min_volume = getattr(settings, 'MIN_24H_VOLUME', 1000000)  # Default 1M USDT
        self.min_price = getattr(settings, 'MIN_PRICE', 1)  # Default 1 USDT
        self.max_pairs = getattr(settings, 'MAX_POSITIONS', 5)  # Default to 5 pairs
        
    async def select_pairs(self):
        """Select trading pairs based on volume and other criteria."""
        try:
            self.logger.info("Selecting trading pairs...")
            
            # Fetch all USDC pairs from the exchange
            pairs = await self.exchange_manager.fetch_usdc_pairs()
            if not pairs:
                self.logger.warning("No USDC pairs found, unable to select trading pairs")
                self.selected_pairs = []
                return self.selected_pairs
            
            # Fetch current market data for all pairs
            market_data = await self.exchange_manager.fetch_current_prices(pairs)
            
            # Filter and sort pairs based on criteria
            filtered_pairs = []
            for symbol, data in market_data.items():
                # Skip pairs with errors
                if 'error' in data and data['error']:
                    self.logger.debug(f"Skipping {symbol} due to error: {data['error']}")
                    continue
                    
                if data['volume_24h'] >= self.min_volume and data['price'] >= self.min_price:
                    filtered_pairs.append({
                        'symbol': symbol,
                        'volume': data['volume_24h'],
                        'price': data['price'],
                        'change': data['change_24h']
                    })
            
            # Sort by 24h volume
            sorted_pairs = sorted(filtered_pairs, key=lambda x: x['volume'], reverse=True)
            
            # Determine max pairs from settings (using MAX_POSITIONS)
            max_pairs = getattr(self.settings, 'MAX_POSITIONS', 5)
            if max_pairs <= 0:
                max_pairs = 5
                self.logger.warning(f"Invalid MAX_POSITIONS value, using default: {max_pairs}")
            
            # Select top pairs up to max_pairs
            available_pairs = min(len(sorted_pairs), max_pairs)
            if available_pairs == 0:
                self.logger.warning("No pairs met the selection criteria")
                self.selected_pairs = []
            else:
                self.selected_pairs = [pair['symbol'] for pair in sorted_pairs[:available_pairs]]
                self.logger.info(f"Selected {len(self.selected_pairs)} trading pairs: {', '.join(self.selected_pairs)}")
            
        except Exception as e:
            self.logger.error(f"Error selecting trading pairs: {str(e)}")
            self.selected_pairs = []
            return self.selected_pairs
            
    def get_selected_pairs(self):
        """Return the currently selected trading pairs."""
        return self.selected_pairs 