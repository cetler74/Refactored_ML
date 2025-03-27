import logging
import asyncio
import time
from datetime import datetime, timedelta
import random  # For demo purposes

class TradingBot:
    """Main trading bot class that manages strategies, exchanges, and order execution."""
    
    def __init__(self, settings, exchange_manager):
        """Initialize the trading bot with settings and exchange manager."""
        self.settings = settings
        self.exchange_manager = exchange_manager
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.logger.info("Trading bot initialized")
        
        # Track trading statistics
        self.stats = {
            "trades_executed": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "ml_models_trained": 0,
            "signals_generated": 0,
            "profit_loss": 0.0,
            "start_time": datetime.now()
        }
        
    async def start(self):
        """Start the trading bot."""
        self.logger.info("Starting trading bot...")
        self.running = True
        
        # Start the main trading loop in a separate task
        asyncio.create_task(self.run_trading_loop())
        
        # Start the ML training loop in a separate task
        asyncio.create_task(self.run_ml_training_loop())
        
        # Log trading environment and settings
        self.log_trading_environment()
        
    async def stop(self):
        """Stop the trading bot."""
        self.logger.info("Stopping trading bot...")
        self.running = False
        
        # Log final statistics
        self.log_trading_statistics()
    
    async def run_trading_loop(self):
        """Main trading loop that processes market data and executes trades."""
        self.logger.info("Trading loop started")
        
        try:
            # Run the trading loop until stopped
            while self.running:
                try:
                    self.logger.info("Trading cycle started")
                    
                    # Step 1: Fetch latest market data
                    self.logger.info("Fetching market data...")
                    await self.fetch_market_data()
                    
                    # Step 2: Generate trading signals
                    self.logger.info("Generating trading signals...")
                    signals = await self.generate_signals()
                    
                    # Step 3: Execute trades based on signals
                    if signals:
                        self.logger.info(f"Generated {len(signals)} trading signals")
                        self.stats["signals_generated"] += len(signals)
                        
                        for signal in signals:
                            self.logger.info(f"Processing signal: {signal['symbol']} - {signal['trade_type']} - Confidence: {signal['confidence']}")
                            await self.execute_trade(signal)
                    else:
                        self.logger.info("No trading signals generated in this cycle")
                    
                    # Step 4: Update portfolio and risk metrics
                    self.logger.info("Updating portfolio metrics...")
                    await self.update_portfolio_metrics()
                    
                    # Log status and wait for next cycle
                    self.log_trading_statistics()
                    self.logger.info("Trading cycle completed")
                    
                except Exception as e:
                    self.logger.error(f"Error in trading cycle: {str(e)}")
                
                # Wait for next trading cycle
                await asyncio.sleep(30)  # Run trading cycle every 30 seconds
        
        except asyncio.CancelledError:
            self.logger.info("Trading loop cancelled")
        except Exception as e:
            self.logger.error(f"Fatal error in trading loop: {str(e)}")
            raise
    
    async def run_ml_training_loop(self):
        """Background loop for training and updating ML models."""
        self.logger.info("ML training loop started")
        
        try:
            # Run the ML training loop until stopped
            while self.running:
                try:
                    self.logger.info("Starting ML model training cycle")
                    
                    # Log the start of training
                    training_start = datetime.now()
                    self.logger.info(f"ML model training started at {training_start}")
                    
                    # Simulate ML training process
                    await self.train_ml_models()
                    
                    # Log completion
                    training_end = datetime.now()
                    duration = (training_end - training_start).total_seconds()
                    self.logger.info(f"ML model training completed in {duration:.2f} seconds")
                    self.stats["ml_models_trained"] += 1
                    
                except Exception as e:
                    self.logger.error(f"Error in ML training cycle: {str(e)}")
                
                # Wait for next training cycle (every hour)
                self.logger.info("Waiting for next ML training cycle")
                await asyncio.sleep(3600)  # Train models every hour
        
        except asyncio.CancelledError:
            self.logger.info("ML training loop cancelled")
        except Exception as e:
            self.logger.error(f"Fatal error in ML training loop: {str(e)}")
    
    async def fetch_market_data(self):
        """Fetch latest market data from exchanges."""
        # This would fetch actual market data from the exchanges
        # For demo purposes, we'll just log the activity
        exchange_names = [ex.name for ex in self.exchange_manager.get_active_exchanges()]
        self.logger.info(f"Fetching market data from exchanges: {', '.join(exchange_names)}")
        
        # Simulate data fetching time
        await asyncio.sleep(0.5)
        self.logger.info(f"Market data fetched successfully for {random.randint(5, 20)} trading pairs")
    
    async def generate_signals(self):
        """Generate trading signals based on market data and ML predictions."""
        # This would use ML models to generate actual trading signals
        # For demo purposes, we'll generate random signals
        signals = []
        
        # Simulate some random trading signals
        if random.random() > 0.7:  # 30% chance of generating signals
            num_signals = random.randint(1, 3)
            symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT"]
            actions = ["buy", "sell"]
            
            for _ in range(num_signals):
                signal = {
                    "symbol": random.choice(symbols),
                    "trade_type": random.choice(actions),
                    "price": round(random.uniform(100, 50000), 2),
                    "quantity": round(random.uniform(0.01, 2), 4),
                    "confidence": round(random.uniform(0.6, 0.95), 2),
                    "timestamp": datetime.now().isoformat()
                }
                signals.append(signal)
        
        return signals
    
    async def execute_trade(self, signal):
        """Execute a trade based on a trading signal."""
        # This would execute actual trades through the exchange
        try:
            self.logger.info(f"Executing {signal['trade_type']} order for {signal['quantity']} {signal['symbol']} at ~{signal['price']}")
            
            # Simulate order execution
            await asyncio.sleep(0.3)
            
            # Simulate success/failure (90% success rate)
            if random.random() < 0.9:
                trade_id = f"TRADE-{int(time.time())}-{random.randint(1000, 9999)}"
                actual_price = signal['price'] * (1 + random.uniform(-0.002, 0.002))  # Slight price slippage
                
                self.logger.info(f"Trade executed successfully: ID {trade_id}, Price: {actual_price}, Slippage: {((actual_price - signal['price']) / signal['price']) * 100:.3f}%")
                
                # Update statistics
                self.stats["trades_executed"] += 1
                self.stats["successful_trades"] += 1
                
                # Calculate P&L for this trade (simplified)
                trade_cost = actual_price * signal['quantity']
                trade_fee = trade_cost * 0.001  # Assume 0.1% fee
                
                self.logger.info(f"Trade cost: {trade_cost:.2f} USDT, Fee: {trade_fee:.2f} USDT")
                
                # Mock P&L calculation
                pnl_pct = random.uniform(-1.5, 2.5)  # More likely to be positive
                pnl_amount = trade_cost * (pnl_pct / 100)
                self.stats["profit_loss"] += pnl_amount
                
                self.logger.info(f"Estimated trade P&L: {pnl_amount:.2f} USDT ({pnl_pct:+.2f}%)")
                
                return True
            else:
                error_reason = random.choice([
                    "insufficient funds", 
                    "exchange timeout", 
                    "price moved", 
                    "minimum order size not met"
                ])
                self.logger.warning(f"Trade execution failed: {error_reason}")
                self.stats["trades_executed"] += 1
                self.stats["failed_trades"] += 1
                return False
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            self.stats["failed_trades"] += 1
            return False
    
    async def train_ml_models(self):
        """Train and update machine learning models."""
        # This would train actual ML models
        # For demo purposes, we'll just simulate training
        
        # Simulate different types of ML models
        models = [
            "price_prediction_lstm", 
            "trend_classification_svm", 
            "volatility_prediction_garch",
            "pattern_recognition_cnn"
        ]
        
        for model in models:
            # Log start of training for this specific model
            self.logger.info(f"Training {model} model with latest data")
            
            # Simulate training time
            training_time = random.uniform(5, 20)
            await asyncio.sleep(0.5)  # Actual sleep is shorter for demo
            
            # Simulate training metrics
            accuracy = random.uniform(0.65, 0.92)
            loss = random.uniform(0.08, 0.35)
            
            self.logger.info(f"Model {model} trained successfully in {training_time:.1f} seconds")
            self.logger.info(f"Training metrics - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
            
            # Simulate model evaluation on test data
            test_accuracy = accuracy * random.uniform(0.9, 1.02)  # Slightly different from training
            self.logger.info(f"Model evaluation on test data - Accuracy: {test_accuracy:.4f}")
            
        self.logger.info(f"All ML models trained and updated successfully")
    
    async def update_portfolio_metrics(self):
        """Update portfolio metrics and risk statistics."""
        # This would calculate actual portfolio metrics
        # For demo purposes, we'll simulate the metrics
        
        # Simulate portfolio metrics
        total_balance = random.uniform(5000, 50000)
        allocated_balance = total_balance * random.uniform(0.1, 0.8)
        available_balance = total_balance - allocated_balance
        
        # Log portfolio statistics
        self.logger.info(f"Portfolio metrics updated - Total: {total_balance:.2f} USDT")
        self.logger.info(f"  Allocated: {allocated_balance:.2f} USDT ({(allocated_balance/total_balance)*100:.1f}%)")
        self.logger.info(f"  Available: {available_balance:.2f} USDT ({(available_balance/total_balance)*100:.1f}%)")
        
        # Simulate risk metrics
        max_drawdown = random.uniform(2, 15)
        sharpe_ratio = random.uniform(0.5, 2.5)
        
        # Log risk statistics
        self.logger.info(f"Risk metrics - Max Drawdown: {max_drawdown:.2f}%, Sharpe Ratio: {sharpe_ratio:.2f}")
    
    def log_trading_environment(self):
        """Log details about the trading environment and configuration."""
        mode = self.exchange_manager.mode
        exchanges = [ex.name for ex in self.exchange_manager.get_active_exchanges()]
        
        self.logger.info(f"Trading bot environment: {mode} mode")
        self.logger.info(f"Active exchanges: {', '.join(exchanges)}")
        self.logger.info(f"Trading pairs: {self.settings.DEFAULT_TIMEFRAMES if hasattr(self.settings, 'DEFAULT_TIMEFRAMES') else 'Default'}")
        self.logger.info(f"Risk per trade: {self.settings.RISK_PER_TRADE if hasattr(self.settings, 'RISK_PER_TRADE') else 'Default'}%")
        self.logger.info(f"Max positions: {self.settings.MAX_POSITIONS if hasattr(self.settings, 'MAX_POSITIONS') else 'Default'}")
    
    def log_trading_statistics(self):
        """Log trading statistics and performance metrics."""
        # Calculate elapsed time
        elapsed = datetime.now() - self.stats["start_time"]
        elapsed_str = str(timedelta(seconds=int(elapsed.total_seconds())))
        
        # Prepare statistics log
        self.logger.info(f"Trading statistics after {elapsed_str} runtime:")
        self.logger.info(f"  Trades executed: {self.stats['trades_executed']}")
        self.logger.info(f"  Successful trades: {self.stats['successful_trades']}")
        self.logger.info(f"  Failed trades: {self.stats['failed_trades']}")
        
        if self.stats['trades_executed'] > 0:
            success_rate = (self.stats['successful_trades'] / self.stats['trades_executed']) * 100
            self.logger.info(f"  Success rate: {success_rate:.1f}%")
        
        self.logger.info(f"  ML models trained: {self.stats['ml_models_trained']}")
        self.logger.info(f"  Signals generated: {self.stats['signals_generated']}")
        self.logger.info(f"  Profit/Loss: {self.stats['profit_loss']:.2f} USDT") 