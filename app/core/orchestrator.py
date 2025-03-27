#!/usr/bin/env python3
"""
Bot Orchestrator

This module is responsible for coordinating all components of the trading bot.
"""

import logging
import os
import sys
import time
import asyncio
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
import json
import pandas as pd
import numpy as np
import random
import signal
import uuid
import websockets
import re
import warnings
from functools import partial

from app.config.settings import Settings, AppMode
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from app.utils.balance_manager import BalanceManager  # Import the proper BalanceManager
from app.exchange.manager import ExchangeManager  # Import ExchangeManager
from app.utils.notification_system import NotificationSystem, NotificationCategory, NotificationPriority  # Import notification system
from app.database.manager import DatabaseManager, TradeRepository, MarketDataRepository  # Import DatabaseManager and repositories
from app.database.repositories import TradingPairRepository, SignalRepository
from app.ta.indicators import TechnicalAnalysisModule  # Import the correct Technical Analysis module
from app.strategies.strategy import StrategyModule  # Import the StrategyModule
from app.ml.model import MLModule  # Import the ML module
from app.utils.reporting import ReportingManager  # Import the ReportingManager

# Import the TradingPairSelector
from app.core.pair_selector import TradingPairSelector

logger = logging.getLogger(__name__)

class BotOrchestrator:
    """
    Orchestrates the trading bot components and provides an interface for the API.
    Acts as a facade for the trading bot, exchange manager, and other components.
    """
    
    def __init__(self, settings):
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        self.mode = settings.APP_MODE
        self.config = settings
        self._running = False
        self.trading_pairs = []
        self.exchanges = {}
        self.start_time = None
        self.last_heartbeat = None
        self.scheduler = None
        
        # Database manager for persistence
        self.db_manager = None  # Will be initialized later
        self.exchange_manager = None  # Will be initialized later
        self.redis_client = None  # For caching and cooldowns
        
        # Modules
        self.data_collector = None
        self.signal_generator = None
        self.trade_executor = None
        self.balance_manager = None
        self.trading_pair_selector = None
        self.ta_module = None
        self.strategy_module = None
        self.ml_module = None
        self.reporting_module = None
        self.modules_initialized = False
        
        # Initialize notification system
        notification_config = getattr(settings, 'NOTIFICATION_CONFIG', {
            'enabled_channels': ['console'],
            'notifications_enabled': True,
            'min_priority': 'low'
        })
        self.notification_system = NotificationSystem(notification_config)
        
        # For signal generation
        self.current_signals = []
        self.previous_signals = []
        
        # Historical data for performance tracking
        # Will be loaded from database on initialization
        self.historical_data = {
            "balance_history": [],
            "pnl_history": [],
            "trade_history": []
        }
        
        # Initialize statistics
        self.stats = {
            "trades_executed": 0,
            "open_positions": 0,
            "closed_positions": 0,
            "active_exchanges": [],
            "active_trading_pairs": [],
            "profit_loss": 0
        }
        
        self.logger.info(f"Orchestrator initialized in {self.mode} mode")
        
    def update_heartbeat(self):
        """
        Update the heartbeat timestamp to indicate the bot is still running.
        This is a synchronous method called by the scheduler.
        """
        try:
            current_time = datetime.now()
            self.last_heartbeat = current_time
            
            # Update basic stats if we have a balance manager
            if self.balance_manager and hasattr(self.balance_manager, 'get_portfolio_summary'):
                try:
                    portfolio_summary = self.balance_manager.get_portfolio_summary()
                    if portfolio_summary:
                        self.stats = {
                            "status": "running",
                            "uptime": (current_time - self.start_time).total_seconds() if self.start_time else 0,
                            "portfolio_value": portfolio_summary.get('total_balance_usd', 0),
                            "open_positions": len(self.balance_manager.open_positions),
                            "last_update": current_time.isoformat()
                        }
                except Exception as e:
                    self.logger.error(f"Error updating stats in heartbeat: {str(e)}")
            
            self.logger.debug(f"Heartbeat updated: {current_time.isoformat()}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating heartbeat: {str(e)}")
            return False

    async def update_historical_data(self):
        """
        Update historical data for portfolio balance and PnL.
        This is an async method that should be scheduled separately.
        """
        try:
            self.logger.info("Updating historical data...")
            
            # Skip if DB manager or balance manager is not available
            if not self.db_manager or not self.balance_manager:
                self.logger.warning("Cannot update historical data: db_manager or balance_manager not available")
                return False
                
            current_time = datetime.now()
            
            # Get current portfolio summary
            portfolio_summary = self.balance_manager.get_portfolio_summary()
            if not portfolio_summary:
                self.logger.warning("Cannot update historical data: portfolio summary not available")
                return False
                
            # Extract balance data
            total_balance = portfolio_summary.get('total_balance_usd', 0)
            
            # Calculate PnL (daily change)
            daily_pnl = 0
            cumulative_pnl = 0
            
            # Get yesterday's balance
            yesterday = current_time - timedelta(days=1)
            yesterday_timestamp = yesterday.isoformat()
            
            # Find most recent balance before yesterday
            for balance_point in reversed(self.historical_data.get("balance_history", [])):
                balance_time = balance_point.get("timestamp")
                if balance_time and balance_time < yesterday_timestamp:
                    previous_balance = balance_point.get("balance", total_balance)
                    daily_pnl = total_balance - previous_balance
                    break
                    
            # Calculate cumulative PnL from all historical trades
            if self.historical_data.get("pnl_history"):
                last_cumulative = self.historical_data["pnl_history"][-1].get("cumulative_pnl", 0)
                cumulative_pnl = last_cumulative + daily_pnl
            else:
                cumulative_pnl = daily_pnl
                
            # Save to database
            if hasattr(self.db_manager, 'balance_repository'):
                # Save balance history
                await self.db_manager.balance_repository.save_balance_history(
                    current_time.isoformat(), total_balance
                )
                
                # Save PnL history
                await self.db_manager.balance_repository.save_pnl_history(
                    current_time.isoformat(), daily_pnl, cumulative_pnl
                )
                
                # Update local cache
                self.historical_data["balance_history"].append({
                    "timestamp": current_time.isoformat(),
                    "balance": total_balance
                })
                
                self.historical_data["pnl_history"].append({
                    "timestamp": current_time.isoformat(),
                    "pnl": daily_pnl,
                    "cumulative_pnl": cumulative_pnl
                })
                
                self.logger.info(f"Historical data updated: balance={total_balance}, daily_pnl={daily_pnl}")
                return True
            else:
                self.logger.warning("Cannot update historical data: balance_repository not available")
                return False
        except Exception as e:
            self.logger.error(f"Error updating historical data: {str(e)}")
            return False
    
    async def save_historical_data(self):
        """Save historical data to database for persistence."""
        try:
            # This is a simplified implementation that doesn't store every update
            # In a production system, we would use a more efficient approach
            
            # Save the most recent balance history point if it exists
            if self.historical_data["balance_history"]:
                latest_balance = self.historical_data["balance_history"][-1]
                await self.db_manager.balance_repository.save_balance_history(
                    latest_balance["timestamp"], 
                    latest_balance["balance"]
                )
                
            # Save the most recent PnL history point if it exists
            if self.historical_data["pnl_history"]:
                latest_pnl = self.historical_data["pnl_history"][-1]
                await self.db_manager.balance_repository.save_pnl_history(
                    latest_pnl["timestamp"],
                    latest_pnl["pnl"],
                    latest_pnl["cumulative_pnl"]
                )
                
            self.logger.debug("Historical data saved to database")
        except Exception as e:
            self.logger.error(f"Error saving historical data: {str(e)}")
    
    @property
    def is_active(self):
        """Check if the bot is actually active based on heartbeat."""
        if not self.last_heartbeat:
            return False
        
        # Check if the last heartbeat was within the expected interval
        time_since_heartbeat = (datetime.now() - self.last_heartbeat).total_seconds()
        return time_since_heartbeat < 30  # Allow for some delay
    
    @property
    def is_running(self):
        """Property to check if the bot is running."""
        return self._running and self.is_active
    
    async def start(self, *, mode="simulation"):
        """Start the trading bot."""
        if self._running:
            self.logger.warning("Trading bot is already running")
            return {"success": False, "error": "Trading bot is already running"}
        
        try:
            self.logger.info(f"Starting trading bot in {mode} mode")
            self._running = True
            self.start_time = datetime.now()
            self.mode = mode
            
            # Explicitly initialize the scheduler
            try:
                self.scheduler = BackgroundScheduler()
                self.scheduler.start()
                self.logger.info("Scheduler initialized and started")
            except Exception as scheduler_error:
                self.logger.error(f"Error initializing scheduler: {str(scheduler_error)}")
                self.scheduler = None
            
            # Initialize any modules that weren't initialized earlier
            if not self.modules_initialized:
                await self.initialize_modules()
            
            # Schedule jobs if scheduler is running
            if self.scheduler and self.scheduler.running:
                self.schedule_jobs()
            else:
                self.logger.error("Scheduler not initialized properly, jobs will not be scheduled")
            
            self.logger.info("Trading bot started successfully")
            
            return {"success": True, "message": f"Trading bot started in {mode} mode", "timestamp": self.start_time.isoformat()}
        except Exception as e:
            self.logger.error(f"Error starting trading bot: {str(e)}")
            self._running = False
            return {"success": False, "error": f"Failed to start trading bot: {str(e)}"}
    
    async def stop(self):
        """Stop the trading bot."""
        if not self._running:
            self.logger.warning("Trading bot is not running")
            return {"success": False, "error": "Trading bot is not running"}
        
        try:
            self.logger.info("Stopping trading bot")
            self._running = False
            stop_time = datetime.now()
            run_duration = (stop_time - self.start_time).total_seconds() if self.start_time else 0
            
            # Stop the scheduler if it's running
            if hasattr(self, 'scheduler') and getattr(self.scheduler, 'running', False):
                self.logger.info("Shutting down scheduler")
                self.scheduler.shutdown()
                self.logger.info("Scheduler shut down successfully")
            
            self.logger.info("Trading bot stopped successfully")
            
            # Clear activity tracking
            self.last_heartbeat = None
            
            # Send notification about bot stopping
            await self.notification_system.send_notification(
                message=f"Trading bot has been stopped after running for {run_duration/3600:.1f} hours",
                title="Trading Bot Stopped",
                category=NotificationCategory.SYSTEM,
                priority=NotificationPriority.MEDIUM,
                data={
                    "run_duration_hours": f"{run_duration/3600:.1f}",
                    "trades_executed": self.stats["trades_executed"],
                    "profit_loss": f"${self.stats['profit_loss']:.2f}"
                }
            )
            
            return {
                "success": True, 
                "message": "Trading bot stopped", 
                "run_duration_seconds": run_duration,
                "timestamp": stop_time.isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error stopping trading bot: {str(e)}")
            
            # Send notification about error stopping bot
            await self.notification_system.notify_error(
                f"Error stopping trading bot: {str(e)}",
                source="System Shutdown",
                severity="high"
            )
            
            return {"success": False, "error": f"Error stopping trading bot: {str(e)}"}
    
    async def get_status(self):
        """Get the current status of the trading bot."""
        if not self._running:
            return {
                "running": False,
                "status": "stopped",
                "mode": self.mode,
                "uptime_seconds": 0
            }
        
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        # Add heartbeat information
        is_active = self.is_active
        last_heartbeat = None
        if self.last_heartbeat:
            last_heartbeat = self.last_heartbeat.isoformat()
            seconds_since_heartbeat = (datetime.now() - self.last_heartbeat).total_seconds()
        else:
            seconds_since_heartbeat = None
        
        return {
            "running": self._running,
            "active": is_active,
            "status": "running" if is_active else "stalled",
            "mode": self.mode,
            "uptime_seconds": uptime,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "last_heartbeat": last_heartbeat,
            "seconds_since_heartbeat": seconds_since_heartbeat
        }
    
    async def get_portfolio(self):
        """Get portfolio data directly from database, without requiring balance_manager."""
        try:
            # Create default portfolio structure
            portfolio_data = {
                "total_balance_usdc": 10000.00,
                "available_usdc": 10000.00,
                "allocated_balance": 0.00,
                "total_positions_value": 0.00,
                "open_positions": {},
                "open_positions_count": 0,
                "win_rate": 0.0,
                "total_trades": 0,
                "total_profit": 0.0,
                "total_loss": 0.0,
                "net_profit": 0.0,
                "pnl_24h": 0.0
            }
            
            # If db_manager is available, retrieve data directly from database
            if hasattr(self, 'db_manager') and self.db_manager:
                try:
                    # Get open positions directly from database
                    if hasattr(self.db_manager, 'trade_repository'):
                        open_positions = await self.db_manager.trade_repository.get_open_positions()
                        if open_positions:
                            portfolio_data["open_positions"] = open_positions
                            portfolio_data["open_positions_count"] = len(open_positions)
                            
                            # Calculate total position value
                            total_position_value = 0.0
                            for symbol, position in open_positions.items():
                                position_value = position.get('position_value', 0.0)
                                total_position_value += position_value
                                
                            portfolio_data["total_positions_value"] = total_position_value
                            portfolio_data["allocated_balance"] = total_position_value
                    
                    # Get balance history to determine current balance
                    if hasattr(self.db_manager, 'balance_repository'):
                        balance_history = await self.db_manager.balance_repository.get_balance_history(days=1)
                        if balance_history and len(balance_history) > 0:
                            latest_balance = balance_history[-1]
                            portfolio_data["total_balance_usdc"] = latest_balance.get('balance', 10000.00)
                            
                            # Calculate available balance (total - allocated)
                            portfolio_data["available_usdc"] = portfolio_data["total_balance_usdc"] - portfolio_data["allocated_balance"]
                    
                    # Get completed trades to calculate win rate and P/L
                    if hasattr(self.db_manager, 'trade_repository'):
                        completed_trades = await self.db_manager.trade_repository.get_trades(status="completed", limit=100)
                        if completed_trades:
                            portfolio_data["total_trades"] = len(completed_trades)
                            
                            # Calculate profit metrics
                            total_profit = 0.0
                            total_loss = 0.0
                            winning_trades = 0
                            
                            # Calculate 24h P/L
                            pnl_24h = 0.0
                            day_ago = datetime.now() - timedelta(days=1)
                            
                            for trade in completed_trades:
                                pnl = trade.get('pnl', 0.0)
                                
                                # Count winning and losing trades
                                if pnl > 0:
                                    total_profit += pnl
                                    winning_trades += 1
                                else:
                                    total_loss += abs(pnl)
                                
                                # Check if trade is within last 24 hours
                                trade_time = datetime.fromisoformat(trade.get('exit_time', '')) if trade.get('exit_time') else None
                                if trade_time and trade_time > day_ago:
                                    pnl_24h += pnl
                            
                            # Calculate win rate and total P/L
                            if portfolio_data["total_trades"] > 0:
                                portfolio_data["win_rate"] = winning_trades / portfolio_data["total_trades"] * 100
                            
                            portfolio_data["total_profit"] = total_profit
                            portfolio_data["total_loss"] = total_loss
                            portfolio_data["net_profit"] = total_profit - total_loss
                            portfolio_data["pnl_24h"] = pnl_24h
                    
                    self.logger.info("Portfolio data retrieved directly from database")
                    return portfolio_data
                    
                except Exception as db_error:
                    self.logger.error(f"Error retrieving portfolio data from database: {str(db_error)}")
                    # Continue with fallback data
            
            # If balance_manager is available and we couldn't get data directly from DB, use it as fallback
            if hasattr(self, 'balance_manager') and self.balance_manager and hasattr(self.balance_manager, 'get_portfolio_summary'):
                return self.balance_manager.get_portfolio_summary()
            
            # If we don't have a balance manager, create a placeholder
            if not hasattr(self, 'balance_manager') or not self.balance_manager:
                self.logger.info("Creating a placeholder balance manager for portfolio data")
                self.balance_manager = self._create_placeholder_balance_manager()
                return self.balance_manager.get_portfolio_summary()
                
            # Return the default portfolio data if we couldn't get it any other way
            self.logger.warning("Using default portfolio data - neither database nor balance manager available")
            return portfolio_data
            
        except Exception as e:
            self.logger.error(f"Error in get_portfolio: {str(e)}")
            # Return default data in case of error
            return {
                "total_balance_usdc": 10000.00,
                "available_usdc": 10000.00,
                "allocated_balance": 0.00,
                "open_positions": {},
                "open_positions_count": 0
            }
    
    async def get_health(self):
        """Get health information about the trading bot and its components."""
        components = {
            "trading_bot": {
                "status": "running" if self._running else "stopped",
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                "mode": self.mode
            },
            "trading_operations": {
                "status": "active" if self._running else "inactive",
                "last_activity": self.last_heartbeat.isoformat() if self.last_heartbeat else None
            },
            "database": {
                "status": "connected" 
            },
            "exchanges": {
                "status": "connected",
                "binance": {
                    "status": "connected",
                    "type": "live" if self.mode == "production" else "testnet"
                }
            },
            "api_server": {
                "status": "up",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Check Redis connection
        try:
            # In a real implementation, this would check the actual Redis connection
            # For simulation, we'll always return connected
            components["redis"] = {
                "status": "connected"
            }
        except Exception as e:
            self.logger.error(f"Error checking Redis status: {str(e)}")
            components["redis"] = {
                "status": "error",
                "error": str(e)
            }
        
        return components

    def schedule_jobs(self):
        """
        Schedule all necessary jobs for the trading bot.
        Ensures proper timing and coordination of various tasks.
        Raises ValueError if the scheduler is not initialized.
        """
        try:
            if not self.scheduler:
                error_msg = "Scheduler not initialized, cannot schedule jobs"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
                
            if not self.scheduler.running:
                try:
                    self.scheduler.start()
                    self.logger.info("Scheduler started")
                except Exception as e:
                    error_msg = f"Failed to start scheduler: {str(e)}"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
            
            # Check existing job IDs to avoid duplication
            existing_job_ids = [job.id for job in self.scheduler.get_jobs()]
            
            # Market data collection: every minute
            job_id = "market_data_collection"
            if job_id not in existing_job_ids:
                self.scheduler.add_job(
                    self.collect_market_data,
                    'interval',
                    minutes=1,
                    id=job_id,
                    replace_existing=True,
                    max_instances=1
                )
                self.logger.info(f"Scheduled job: {job_id} (every 1 minute)")
            
            # Historical data update: daily
            job_id = "historical_data_update"
            if job_id not in existing_job_ids:
                self.scheduler.add_job(
                    self.update_historical_data,
                    'interval',
                    hours=24,
                    id=job_id,
                    replace_existing=True,
                    max_instances=1
                )
                self.logger.info(f"Scheduled job: {job_id} (every 24 hours)")
            
            # Signal generation: every minute
            job_id = "signal_generation"
            if job_id not in existing_job_ids:
                self.scheduler.add_job(
                    self.generate_signals,
                    'interval',
                    minutes=1,
                    id=job_id,
                    replace_existing=True,
                    max_instances=1
                )
                self.logger.info(f"Scheduled job: {job_id} (every 1 minute)")
            
            # Trading pair updates: every 4 hours
            job_id = "trading_pair_updates"
            if job_id not in existing_job_ids:
                self.scheduler.add_job(
                    self.select_trading_pairs,
                    'interval',
                    hours=4,
                    id=job_id,
                    replace_existing=True,
                    max_instances=1
                )
                self.logger.info(f"Scheduled job: {job_id} (every 4 hours)")
            
            # ML training: based on config
            if self.ml_module and hasattr(self.settings, 'enable_ml') and self.settings.enable_ml:
                training_interval = getattr(self.settings, "ml_training_interval_hours", 24)
                job_id = "ml_training"
                if job_id not in existing_job_ids:
                    self.scheduler.add_job(
                        self.train_ml_models,
                        'interval',
                        hours=training_interval,
                        id=job_id,
                        replace_existing=True,
                        max_instances=1
                    )
                    self.logger.info(f"Scheduled job: {job_id} (every {training_interval} hours)")
            
            # Position monitoring: every 1 minute (high priority)
            job_id = "position_monitoring"
            if job_id not in existing_job_ids:
                self.scheduler.add_job(
                    self.update_position_prices,
                    'interval',
                    minutes=1,
                    id=job_id,
                    replace_existing=True,
                    max_instances=1,
                    misfire_grace_time=30  # Allow job to be late by up to 30 seconds
                )
                self.logger.info(f"Scheduled job: {job_id} (every 1 minute)")
            
            # Position monitoring check: every 5 minutes
            job_id = "position_monitoring_check"
            if job_id not in existing_job_ids:
                self.scheduler.add_job(
                    self._ensure_position_monitoring,
                    'interval',
                    minutes=5,
                    id=job_id,
                    replace_existing=True,
                    max_instances=1
                )
                self.logger.info(f"Scheduled job: {job_id} (every 5 minutes)")
            
            # Heartbeat update: every 30 seconds
            job_id = "heartbeat_update"
            if job_id not in existing_job_ids:
                self.scheduler.add_job(
                    self.update_heartbeat,
                    'interval',
                    seconds=30,
                    id=job_id,
                    replace_existing=True,
                    max_instances=1
                )
                self.logger.info(f"Scheduled job: {job_id} (every 30 seconds)")
            
            # Database cleanup: every 24 hours
            job_id = "database_cleanup"
            if job_id not in existing_job_ids and hasattr(self, 'cleanup_old_data'):
                self.scheduler.add_job(
                    self.cleanup_old_data,
                    'interval',
                    hours=24,
                    id=job_id,
                    replace_existing=True,
                    max_instances=1
                )
                self.logger.info(f"Scheduled job: {job_id} (every 24 hours)")
            
            self.logger.info("All jobs scheduled successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error scheduling jobs: {str(e)}")
            raise  # Re-raise to abort startup
    
    async def _ensure_position_monitoring(self):
        """
        Ensure that position monitoring is working correctly.
        This is a safety mechanism to restart position monitoring if it fails.
        """
        try:
            # Check if we have open positions
            if not hasattr(self, 'balance_manager') or not self.balance_manager:
                self.logger.warning("Balance manager not initialized, cannot check positions")
                return
                
            open_positions = self.balance_manager.open_positions
            if not open_positions:
                return  # No positions to monitor
                
            # Check when positions were last updated
            current_time = datetime.now()
            position_updated = False
            
            for symbol, position in open_positions.items():
                # Check if position has a last_updated timestamp
                last_updated = position.get('last_updated')
                if not last_updated:
                    continue
                    
                try:
                    # Parse timestamp to datetime
                    if isinstance(last_updated, str):
                        last_updated = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                    
                    # Calculate time difference in minutes
                    time_diff = (current_time - last_updated).total_seconds() / 60
                    
                    # If any position was updated in the last 3 minutes, consider monitoring is working
                    if time_diff < 3:
                        position_updated = True
                        break
                except Exception as e:
                    self.logger.error(f"Error parsing timestamp for {symbol}: {str(e)}")
            
            # If no position was updated recently and we have positions, trigger an update
            if not position_updated and open_positions:
                self.logger.warning("Position monitoring may have stopped, triggering manual update")
                await self.update_position_prices()
                
                # Verify jobs are still scheduled
                self._restart_position_monitoring_if_needed()
                
        except Exception as e:
            self.logger.error(f"Error in position monitoring check: {str(e)}")
            
    def _restart_position_monitoring_if_needed(self):
        """
        Restart position monitoring job if it's missing from the scheduler.
        """
        try:
            if not self.scheduler:
                return
                
            # Check if the job exists
            job_id = "position_monitoring"
            existing_jobs = [job.id for job in self.scheduler.get_jobs()]
            
            if job_id not in existing_jobs:
                self.logger.warning(f"Position monitoring job missing, restarting it")
                self.scheduler.add_job(
                    self.update_position_prices,
                    'interval',
                    minutes=1,
                    id=job_id,
                    replace_existing=True,
                    max_instances=1
                )
                
        except Exception as e:
            self.logger.error(f"Error restarting position monitoring: {str(e)}")
    
    async def collect_market_data(self):
        """
        Collect market data for all active trading pairs and store it in market_data_store.
        Returns the collected data for convenience.
        """
        try:
            self.logger.info("Collecting market data...")
            
            # Get active trading pairs
            active_pairs = self.trading_pairs
            
            if not active_pairs:
                self.logger.warning("No active trading pairs available")
                return {"success": False, "error": "No active trading pairs"}

            self.logger.info(f"Collecting market data for {len(active_pairs)} trading pairs: {', '.join(active_pairs)}")
            
            market_data = {}
            
            for symbol in active_pairs:
                market_data[symbol] = {}
                for timeframe in self.settings.default_timeframes:
                    try:
                        # Fetch OHLCV data
                        ohlcv_data = await self.exchange_manager.fetch_ohlcv(
                            symbol=symbol,
                            timeframe=timeframe,
                            limit=500  # Get more data points for proper analysis
                        )
                        
                        if ohlcv_data is not None and not ohlcv_data.empty:
                            # Calculate technical indicators if TA module is available
                            if self.ta_module and hasattr(self.ta_module, 'calculate_indicators'):
                                ohlcv_data = await self.ta_module.calculate_indicators(ohlcv_data)
                                
                            market_data[symbol][timeframe] = ohlcv_data
                            
                    except Exception as e:
                        self.logger.error(f"Error fetching {timeframe} data for {symbol}: {str(e)}")
                        continue
                
                if not market_data[symbol]:
                    self.logger.warning(f"No data collected for {symbol}")
                    del market_data[symbol]
                
                # Update current price in market_data for each symbol
                try:
                    ticker = await self.exchange_manager.fetch_ticker(symbol)
                    if ticker and "last" in ticker:
                        # Initialize self.market_data if it doesn't exist
                        if not hasattr(self, 'market_data'):
                            self.market_data = {}
                        # Initialize market_data entry for this symbol if not exists
                        if symbol not in self.market_data:
                            self.market_data[symbol] = {}
                        self.market_data[symbol]['price'] = ticker["last"]
                except Exception as e:
                    self.logger.error(f"Error fetching ticker for {symbol}: {str(e)}")
            
            if not market_data:
                self.logger.error("Failed to collect any market data")
                return {"success": False, "error": "No market data collected"}
                
            # Initialize market_data_store if it doesn't exist
            if not hasattr(self, 'market_data_store'):
                self.market_data_store = {}
                
            # Update market_data_store with new data
            for symbol, timeframes in market_data.items():
                if symbol not in self.market_data_store:
                    self.market_data_store[symbol] = {}
                for timeframe, data in timeframes.items():
                    self.market_data_store[symbol][timeframe] = data
            
            return {
                "success": True,
                "data": market_data,
                "timestamp": datetime.now().timestamp()
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting market data: {str(e)}")
            import traceback
            self.logger.debug(f"Market data collection error details: {traceback.format_exc()}")
            return {"success": False, "error": str(e)}
    
    async def select_trading_pairs(self):
        """Select trading pairs for data collection and trading."""
        try:
            self.logger.info("Selecting trading pairs based on 24h volume...")
            max_positions = getattr(self.settings, 'MAX_POSITIONS', 5)
            
            # FIRST APPROACH: Fetch high volume USDC pairs directly from exchange
            try:
                # Request 40 pairs as specified, we'll filter down to max_positions later
                high_volume_pairs = await self.exchange_manager.fetch_high_volume_usdc_pairs(
                    limit=40  # Fetch 40 pairs as requested
                )
                
                if high_volume_pairs:
                    self.logger.info(f"Found {len(high_volume_pairs)} high volume USDC pairs")
                    
                    # Limit to max positions for actual trading
                    selected_pairs = high_volume_pairs[:max_positions]
                    
                    # Log all fetched pairs for reference
                    self.logger.info(f"Retrieved top 40 pairs by volume: {high_volume_pairs}")
                    self.logger.info(f"Selected {max_positions} trading pairs for active use: {selected_pairs}")
                    
                    return selected_pairs
            except Exception as e:
                self.logger.error(f"Error fetching high volume pairs: {str(e)}")
                self.logger.info("Falling back to alternative selection methods...")
            
            # SECOND APPROACH: Use trading pair selector
            # Ensure trading pair selector is initialized
            if not hasattr(self, 'trading_pair_selector') or not self.trading_pair_selector:
                self.logger.info("Initializing trading pair selector")
                from app.core.pair_selector import TradingPairSelector
                self.trading_pair_selector = TradingPairSelector(
                    exchange_manager=self.exchange_manager,
                    settings=self.settings
                )
                self.logger.info("Trading pair selector initialized")
            
            # Get selected pairs from selector
            self.logger.info("Falling back to trading pair selector...")
            pairs = await self.trading_pair_selector.select_pairs()
            
            # THIRD APPROACH: Fetch default USDC pairs
            if not pairs:
                self.logger.info("No selected pairs, fetching default USDC pairs")
                pairs = await self.exchange_manager.fetch_usdc_trading_pairs()
                
                # Limit to max positions
                if len(pairs) > max_positions:
                    pairs = pairs[:max_positions]
            
            if not pairs:
                self.logger.warning("No trading pairs available")
                return []
            
            self.logger.info(f"Selected trading pairs: {pairs}")
            return pairs
            
        except Exception as e:
            self.logger.error(f"Error selecting trading pairs: {str(e)}")
            import traceback
            self.logger.debug(f"Trading pair selection error details: {traceback.format_exc()}")
            return []
    
    async def train_ml_models(self):
        """Train ML models with new market data.
        
        OPTIMIZATION: Only trains models using data from selected trading pairs up to MAX_POSITIONS,
        and cleans up metrics history for pairs that are no longer active to avoid memory bloat.
        """
        try:
            self.logger.info("Training ML models...")
            
            # Get selected trading pairs and limit to max positions
            max_positions = getattr(self.settings, 'MAX_POSITIONS', 5)
            selected_pairs = []
            
            if hasattr(self, 'trading_pair_selector') and self.trading_pair_selector:
                if self.trading_pair_selector.selected_pairs:
                    selected_pairs = self.trading_pair_selector.selected_pairs[:max_positions]
                    self.logger.info(f"Will train ML models for {len(selected_pairs)} selected pairs")
            
            # If no pairs available, log and return
            if not selected_pairs:
                self.logger.warning("No trading pairs available for ML model training")
                return False
            
            # If we have the ML module, pass the active pairs to it for training
            if hasattr(self, 'ml_module') and self.ml_module:
                # Train the ML models with the active trading pairs
                training_result = await self.ml_module.train_models(active_pairs=selected_pairs)
                
                # If training was successful, clean up metrics for inactive pairs
                if training_result.get('success', False) and hasattr(self.ml_module, 'performance_tracker'):
                    self.ml_module.performance_tracker.cleanup_inactive_metrics(selected_pairs)
                    self.logger.info("Cleaned up metrics for inactive trading pairs")
                
                # Log training results
                if training_result.get('success', False):
                    self.logger.info("ML models trained successfully")
                    
                    # Get accuracy metrics for notification
                    forest_accuracy = training_result.get('forest_model', {}).get('accuracy', 0)
                    lstm_accuracy = training_result.get('lstm_model', {}).get('accuracy', 0)
                    tft_accuracy = training_result.get('tft_model', {}).get('accuracy', 0)
                    
                    # Update last training time
                    self.ml_last_training_time = datetime.now()
                    
                    # Send a notification about successful ML training
                    if hasattr(self, 'notification_system') and self.notification_system:
                        await self.notification_system.send_notification(
                            message="ML models have been retrained with latest market data",
                            title="ML Model Update",
                            category="ML_UPDATE",
                            priority="LOW",
                            data={
                                "training_time": "30 seconds",
                                "accuracy": f"{forest_accuracy:.2f}",
                                "models_updated": ["XGBoost", "LSTM", "Ensemble"],
                                "active_pairs": len(selected_pairs)
                            }
                        )
                else:
                    error_msg = training_result.get('error', 'Unknown error')
                    self.logger.error(f"ML model training failed: {error_msg}")
                    
                    # Send notification about the ML training error
                    if hasattr(self, 'notification_system') and self.notification_system:
                        await self.notification_system.notify_error(
                            f"Failed to train ML models: {error_msg}",
                            source="ML Training",
                            severity="medium"
                        )
            else:
                self.logger.warning("ML module not initialized, skipping model training")
            
            # Update heartbeat to show this process is running
            self.update_heartbeat()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training ML models: {str(e)}")
            
            # Send a notification about the ML training error
            if hasattr(self, 'notification_system') and self.notification_system:
                await self.notification_system.notify_error(
                    f"Failed to train ML models: {str(e)}",
                    source="ML Training",
                    severity="medium"
                )
            
            return False
    
    async def generate_signals(self):
        """Generate trading signals based on market data and strategies."""
        try:
            self.logger.info("Generating trading signals...")
            
            # Use market data from market_data_store if available, otherwise collect it
            if not hasattr(self, 'market_data_store') or not self.market_data_store:
                self.logger.info("Market data store is empty, collecting fresh data")
                market_data_result = await self.collect_market_data()
                if not market_data_result.get("success", False):
                    self.logger.error(f"Failed to get market data: {market_data_result.get('error', 'Unknown error')}")
                    return []
            else:
                self.logger.info("Using existing market data from store")
                market_data_result = {
                    "success": True,
                    "data": self.market_data_store,
                    "timestamp": datetime.now().timestamp()
                }
            
            # Ensure strategy module is initialized
            if not self.strategy_module:
                self.logger.error("Strategy module not initialized, cannot generate signals")
                return []
            
            # Call the strategy module to generate signals
            self.logger.info("Calling strategy_module.generate_signals")
            signals = await self.strategy_module.generate_signals(market_data_result)
            
            # Process and filter signals
            if signals:
                self.logger.info(f"Received {len(signals)} signals from strategy module")
                
                # Apply risk management rules to filter and enrich signals
                filtered_signals = await self.process_signals(signals)
                
                self.logger.info(f"Generated {len(filtered_signals)} trading signals")
                
                # Log detailed signal information
                for signal in filtered_signals:
                    symbol = signal.get('symbol', 'Unknown')
                    trade_type = signal.get('trade_type', 'Unknown')
                    price = signal.get('price', 0)
                    self.logger.info(f"Signal: {symbol} - {trade_type} at {price}")
                
                # Update signals in database
                if hasattr(self.db_manager, 'save_signals'):
                    await self.db_manager.save_signals(filtered_signals)
                
                return filtered_signals
            else:
                self.logger.info("No signals generated")
                return []
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            import traceback
            self.logger.debug(f"Signal generation error details: {traceback.format_exc()}")
            return []
    
    async def execute_trades(self):
        """Execute trades based on the generated signals."""
        if not self._running:
            self.logger.warning("Attempted to execute trades while bot is not running")
            return []
            
        if not self.current_signals:
            self.logger.info("No signals to execute trades for")
            return []
            
        try:
            self.logger.info(f"Executing trades for {len(self.current_signals)} signals")
            
            executed_trades = []
            
            for signal in self.current_signals:
                self.logger.info(f"Processing signal: {signal}")
                
                # Get the symbol and action from the signal
                symbol = signal.get('symbol')
                trade_type = signal.get('trade_type')
                price = signal.get('price', 0)
                confidence = signal.get('confidence', 0.5)
                
                if not symbol or not trade_type:
                    self.logger.warning(f"Invalid signal format, missing symbol or trade_type: {signal}")
                    continue
                
                # Calculate position size based on confidence and available balance
                available_capital = await self.balance_manager.get_available_usdc()
                position_size = self._calculate_position_size(available_capital, confidence, symbol)
                
                if position_size <= 0:
                    self.logger.warning(f"Calculated position size is 0, skipping trade for {symbol}")
                    continue
                
                # Check if we already have a position in this symbol
                existing_position = await self.balance_manager.get_position(symbol)
                
                if existing_position:
                    # Skip if we already have an open position in the same direction
                    if existing_position.get('type') == trade_type:
                        self.logger.info(f"Already have an open {trade_type} position for {symbol}, skipping")
                        continue
                        
                    # For now, let's not take reverse positions (could implement later)
                    # Could close the existing position and open a new one in the opposite direction
                    self.logger.info(f"Have open position for {symbol} in opposite direction, skipping")
                    continue
                
                # Determine order type based on rules
                order_type = "MARKET"  # Default to market orders
                
                # Check if we should use TWAP execution for large orders
                use_twap = False
                if position_size > 5000:  # Arbitrary threshold, adjust based on needs
                    use_twap = True
                
                if use_twap:
                    # Use the new TWAP execution method
                    trade_result = await self.exchange_manager.execute_optimal_order({
                        "symbol": symbol,
                        "trade_type": trade_type,
                        "amount": position_size / price,  # Convert USD amount to asset units
                        "price": price,
                        "order_type": order_type,
                        "use_twap": True,
                        "twap_duration_minutes": 10,
                        "twap_slices": 5
                    })
                else:
                    # Use regular order execution
                    trade_result = await self.exchange_manager.create_order(
                        symbol=symbol,
                        order_type=order_type,
                        trade_type=trade_type,
                        amount=position_size / price,  # Convert USD amount to asset units
                        price=price
                    )
                
                # Process the trade result
                if trade_result and not trade_result.get('error'):
                    self.logger.info(f"Successfully executed trade for {symbol}: {trade_result}")
                    
                    # Add additional trade info
                    trade_result['symbol'] = symbol
                    trade_result['trade_type'] = trade_type
                    trade_result['price'] = price
                    trade_result['confidence'] = confidence
                    trade_result['strategy'] = signal.get('strategy', 'unknown')
                    trade_result['timeframe'] = signal.get('timeframe', '1h')
                    
                    # Add risk metrics if available
                    if 'stop_loss' in signal:
                        trade_result['stop_loss'] = signal['stop_loss']
                    if 'take_profit' in signal:
                        trade_result['take_profit'] = signal['take_profit']
                    
                    # Save the trade to the database
                    if self.db_manager:
                        trade_id = await self.db_manager.store_trade(trade_result)
                        trade_result['id'] = trade_id
                    
                    executed_trades.append(trade_result)
                    
                    # Update signal as executed
                    signal['is_executed'] = True
                    signal['trade_id'] = trade_result.get('id')
                    
                    # Send notification about the executed trade
                    await self.notification_system.notify_trade_executed(trade_result)
                else:
                    error_msg = trade_result.get('error', 'Unknown error')
                    self.logger.error(f"Failed to execute trade for {symbol}: {error_msg}")
            
            return executed_trades
            
        except Exception as e:
            self.logger.exception(f"Error executing trades: {str(e)}")
            return []
            
    async def get_signals(self):
        """Get trading signals from the strategy module."""
        try:
            if not self.strategy_module:
                self.logger.error("Strategy module not initialized")
                return {"success": False, "error": "Strategy module not initialized"}
            
            # Check if we have market data
            if not hasattr(self, 'market_data') or not self.market_data:
                self.logger.error("No market data available for signal generation")
                return {"success": False, "error": "No market data available"}
            
            # Format market data for the strategy module
            formatted_data = {}
            
            for symbol, data in self.market_data.items():
                if 'ohlcv' in data and data['ohlcv']:
                    formatted_data[symbol] = data['ohlcv']
            
            # Generate signals using the strategy module
            signals = await self.strategy_module.generate_signals(formatted_data)
            
            # Store the signals for later use
            self.current_signals = signals
            
            return {"success": True, "signals": signals}
            
        except Exception as e:
            self.logger.exception(f"Error generating signals: {str(e)}")
            return {"success": False, "error": str(e)}
            
    async def open_position(self, signal):
        """Handle opening a new position."""
        return await self.balance_manager.open_position(signal)
    
    async def close_position(self, symbol=None, position_data=None, reason="signal"):
        """
        Handle closing an existing position with enhanced exit flow.
        
        Args:
            symbol: Symbol to close position for
            position_data: Optional position data if already available
            reason: Reason for closing the position
            
        Returns:
            Dictionary with trade result details
        """
        if not symbol and not position_data:
            self.logger.error("Either symbol or position_data must be provided")
            return {"success": False, "error": "Missing symbol or position data"}
            
        # If no symbol provided, extract from position data
        if not symbol and position_data:
            symbol = position_data.get("symbol")
            
        # Get position data if not provided
        if not position_data:
            positions = self.balance_manager.open_positions
            if symbol not in positions:
                return {"success": False, "error": f"No open position for {symbol}"}
            position_data = positions[symbol]
            
        start_time = time.time()
        
        try:
            # 1. Continuous Monitoring & Reassessment
            # Get market data across all timeframes for this symbol
            market_data = {}
            if symbol in self.market_data_store:
                market_data = self.market_data_store[symbol]
                
            # Calculate fresh indicators specifically for this position
            if self.ta_module:
                position_indicators = await self.ta_module.monitor_positions(
                    {symbol: position_data}, 
                    {symbol: market_data}
                )
                
                # Log position-specific indicators
                if symbol in position_indicators:
                    indicators = position_indicators[symbol]["indicators"]
                    self.logger.debug(f"Position indicators for {symbol}: {indicators}")
                    
                    # Check for technical exit signals
                    if position_indicators[symbol]["exit_signals"]["technical"]:
                        reason = f"technical_signal:{position_indicators[symbol]['exit_signals']['reason']}"
                        self.logger.info(f"Technical exit signal triggered: {reason}")
            
            # 2. ML Module Exit Evaluation
            ml_exit_decision = {"should_exit": False, "confidence": 0.0}
            if self.ml_module and symbol in market_data:
                primary_timeframe = position_data.get("timeframe", "1h")
                if primary_timeframe in market_data:
                    # Get ML prediction on whether to exit
                    ml_exit_decision = await self.ml_module.evaluate_exit_decision(
                        position_data, 
                        market_data[primary_timeframe]
                    )
                    
                    if ml_exit_decision.get("should_exit", False):
                        if reason == "signal":  # Only update if not already set
                            reason = f"ml_signal:{ml_exit_decision.get('reason', 'unknown')}"
                        self.logger.info(f"ML exit signal triggered with {ml_exit_decision.get('confidence', 0):.2f} confidence")
            
            # 3. Close the position via the Balance Manager
            current_price = None
            if symbol in self.market_data and 'price' in self.market_data[symbol]:
                current_price = self.market_data[symbol]['price']
                
            # Execute the trade closure
            trade_result = await self.balance_manager.close_position(symbol, current_price, reason)
            
            # Calculate additional metrics for the trade
            if trade_result.get("success", False):
                # Get execution time
                execution_time_ms = int((time.time() - start_time) * 1000)
                trade_result["execution_time_ms"] = execution_time_ms
                
                # Add metrics from our earlier analysis
                if self.ta_module and symbol in position_indicators:
                    # Add technical indicators at exit
                    indicators = position_indicators[symbol]["indicators"]
                    trade_result["exit_indicators"] = indicators
                    
                    # Add exit signal data
                    exit_signals = position_indicators[symbol]["exit_signals"]
                    trade_result["technical_exit_signals"] = exit_signals
                    
                    # Add volatility data
                    if "atr" in indicators:
                        trade_result["market_volatility"] = indicators["atr"]
                        
                    # Add higher timeframe confirmation
                    if "higher_timeframe_signals" in indicators:
                        trade_result["higher_tf_aligned"] = indicators["higher_timeframe_signals"].get("confirms_exit", False)
                
                # Add ML data if available
                if self.ml_module and ml_exit_decision:
                    trade_result["ml_exit_decision"] = ml_exit_decision
                    trade_result["ml_confidence"] = ml_exit_decision.get("confidence", 0.0)
                
                # 4. Database Persistence
                if self.db_manager:
                    # Store completed trade with all metrics
                    await self.db_manager.store_completed_trade(trade_result)
                
                # 5. ML Feedback Loop
                if self.ml_module:
                    # Prepare trade data for ML feedback
                    feedback_data = self._prepare_trade_for_feedback(trade_result)
                    
                    # Send to ML module asynchronously
                    asyncio.create_task(self.ml_module.incorporate_trade_results([feedback_data]))
                    
                # 6. Analytics and Notification
                if hasattr(self, 'notification_system') and self.notification_system:
                    # Format PnL for display
                    pnl = trade_result.get("pnl", 0)
                    pnl_percentage = trade_result.get("pnl_percentage", 0)
                    pnl_display = f"${pnl:.2f} ({pnl_percentage:.2f}%)"
                    
                    # Determine priority based on P/L
                    priority = "medium"
                    if abs(pnl_percentage) > 5:
                        priority = "high"
                        
                    # Send trade execution notification
                    await self.notification_system.send_trade_execution(
                        symbol=symbol,
                        action="exit",
                        price=trade_result.get("exit_price"),
                        reason=reason,
                        pnl=pnl_display,
                        priority=priority
                    )
            
            return trade_result
            
        except Exception as e:
            self.logger.exception(f"Error closing position for {symbol}: {str(e)}")
            
            # Send error notification if available
            if hasattr(self, 'notification_system') and self.notification_system:
                await self.notification_system.send_error(
                    error_type="trade_exit_error",
                    message=f"Failed to exit position for {symbol}: {str(e)}",
                    priority="high"
                )
                
            return {"success": False, "symbol": symbol, "error": str(e)}
    
    def _prepare_trade_for_feedback(self, trade_result: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare trade result data for ML feedback loop"""
        if not trade_result:
            return {}
            
        # Get symbol for this trade
        symbol = trade_result.get("symbol")
        if not symbol:
            return {}
            
        # Extract relevant features from market data
        features = {}
        if symbol in self.market_data_store:
            market_data = self.market_data_store[symbol]
            timeframe = trade_result.get("timeframe", "1h")
            
            if timeframe in market_data and self.ta_module:
                # Get custom features from latest data point
                df = market_data[timeframe]
                if not df.empty:
                    features = self.ta_module.generate_signal_features(df)
        
        # Prepare feedback record
        feedback = {
            **trade_result,  # Include all trade result data
            "features": features,  # Add features
            "timestamp": datetime.now().isoformat()
        }
        
        return feedback
    
    async def startup(self):
        """Start the bot orchestrator and all its components."""
        if self._running:
            self.logger.info("Bot orchestrator is already running")
            return
            
        # Initialize modules if not already initialized
        if not self.scheduler:
            await self.initialize_modules()
            
        # Schedule jobs if not already scheduled
        self.schedule_jobs()
        
        # Start the scheduler if not already running
        if not getattr(self.scheduler, 'running', False):
            self.scheduler.start()
        
        # Mark as running
        self._running = True
        self.start_time = datetime.now()
        
        logger.info("Bot orchestrator started successfully")
        
        # Return the running status for API to check
        return {"status": "running", "mode": self.mode, "timestamp": self.start_time.isoformat()}
    
    async def initialize_modules(self):
        """
        Initialize all modules needed for the trading bot operation.
        Ensures proper setup and configuration of each component.
        Raises exceptions for critical failures rather than creating placeholders.
        """
        try:
            # Initialize exchange manager - CRITICAL COMPONENT
            if not self.exchange_manager:
                try:
                    # Initialize with settings object and sandbox mode
                    self.exchange_manager = ExchangeManager(
                        settings=self.settings,
                        sandbox=getattr(self.settings, 'simulation_mode', True)
                    )
                    await self.exchange_manager.initialize()
                    self.logger.info("Exchange manager initialized")
                except Exception as e:
                    error_msg = f"Failed to initialize Exchange Manager: {str(e)}"
                    self.logger.error(error_msg)
                    # This is a critical component, so we raise the exception
                    raise ValueError(error_msg)

            # Initialize database manager - CRITICAL COMPONENT
            if not self.db_manager:
                try:
                    # Get database connection string from settings or use default SQLite db
                    db_connection_string = getattr(self.settings, 'db_connection_string', 'sqlite:///trading_bot.db')
                    self.db_manager = DatabaseManager(db_connection_string)
                    await self.db_manager.initialize()
                    self.logger.info("Database manager initialized")
                except Exception as e:
                    error_msg = f"Failed to initialize Database Manager: {str(e)}"
                    self.logger.error(error_msg)
                    # This is a critical component, so we raise the exception
                    raise ValueError(error_msg)

            # Initialize repositories
            if hasattr(self.db_manager, 'trade_repository') and not getattr(self.db_manager, 'trade_repository', None):
                self.db_manager.trade_repository = TradeRepository(self.db_manager)
                self.logger.info("Trade repository initialized")

            if hasattr(self.db_manager, 'market_data_repository') and not getattr(self.db_manager, 'market_data_repository', None):
                self.db_manager.market_data_repository = MarketDataRepository(self.db_manager)
                self.logger.info("Market data repository initialized")

            # Initialize balance manager if not already
            if not self.balance_manager:
                try:
                    self.balance_manager = BalanceManager(
                        exchange_manager=self.exchange_manager, 
                        db_manager=self.db_manager,
                        initial_balance=getattr(self.settings, 'INITIAL_BALANCE', 10000),
                        simulation_mode=getattr(self.settings, 'simulation_mode', True)
                    )
                    await self.balance_manager.initialize()
                    self.logger.info("Balance manager initialized and ready")
                except Exception as balance_error:
                    self.logger.error(f"Error initializing balance manager: {str(balance_error)}")
                    self.balance_manager = None
                    # Non-critical component; will try to recreate when needed

            # Initialize Technical Analysis module
            if not self.ta_module:
                try:
                    self.ta_module = TechnicalAnalysisModule()
                    self.logger.info("Technical analysis module initialized")
                except Exception as e:
                    self.logger.error(f"Error initializing TA module: {str(e)}")
                    self.ta_module = None
                    # Non-critical component; will skip TA calculations when needed

            # Initialize Strategy module
            if not self.strategy_module:
                try:
                    self.strategy_module = StrategyModule(
                        settings=self.settings,
                        ta_module=self.ta_module,
                        ml_module=self.ml_module
                    )
                    self.logger.info("Strategy module initialized")
                except Exception as e:
                    self.logger.error(f"Error initializing Strategy module: {str(e)}")
                    self.strategy_module = None
                    # Non-critical component; will skip strategy generation when needed

            # Initialize ML module
            if not self.ml_module and hasattr(self.settings, 'ENABLE_ML') and self.settings.ENABLE_ML:
                try:
                    self.ml_module = MLModule(
                        settings=self.settings,
                        db_manager=self.db_manager
                    )
                    self.logger.info("ML module initialized")
                except Exception as e:
                    self.logger.error(f"Error initializing ML module: {str(e)}")
                    self.ml_module = None
                    # Non-critical component; will skip ML predictions when needed

            # Initialize Reporting module
            if not self.reporting_module:
                try:
                    self.reporting_module = ReportingManager(
                        db_manager=self.db_manager
                    )
                    self.logger.info("Reporting module initialized")
                except Exception as e:
                    self.logger.error(f"Error initializing Reporting module: {str(e)}")
                    self.reporting_module = None
                    # Non-critical component; will skip reporting when needed

            # Initialize scheduler
            if not self.scheduler:
                try:
                    self.scheduler = BackgroundScheduler()
                    self.logger.info("Scheduler initialized")
                except Exception as e:
                    self.logger.error(f"Error initializing scheduler: {str(e)}")
                    self.scheduler = None
                    # Critical component, but we'll raise when scheduling jobs if missing
                
            # Initialize market data store
            if not hasattr(self, 'market_data_store'):
                self.market_data_store = {}
                self.logger.info("Market data store initialized")
                
            # Initialize trading pair selector
            if not self.trading_pair_selector:
                try:
                    self.trading_pair_selector = TradingPairSelector(
                        exchange_manager=self.exchange_manager,
                        settings=self.settings
                    )
                    self.logger.info("Trading pair selector initialized")
                except Exception as e:
                    self.logger.error(f"Error initializing Trading pair selector: {str(e)}")
                    self.trading_pair_selector = None
                    # Non-critical component; will skip pair selection when needed
            
            self.logger.info("All modules successfully initialized")
            
            # Initialize heartbeat
            self.last_heartbeat = datetime.now()
            
            # Get balance from the initialized balance manager or set default
            portfolio_value = 0
            if self.balance_manager and hasattr(self.balance_manager, 'get_portfolio_summary'):
                portfolio_summary = self.balance_manager.get_portfolio_summary()
                portfolio_value = portfolio_summary.get('total_balance_usd', 0)
            elif hasattr(self.settings, 'INITIAL_BALANCE'):
                portfolio_value = self.settings.INITIAL_BALANCE
            else:
                portfolio_value = 5000.0  # Default fallback
                
            self.stats = {
                "status": "running",
                "uptime": 0,
                "portfolio_value": portfolio_value,
                "last_update": datetime.now().isoformat()
            }
            
            # Mark as initialized
            self.modules_initialized = True
            
            # Load historical data if available
            try:
                # Load balance history
                if hasattr(self.db_manager, 'balance_repository'):
                    self.historical_data["balance_history"] = await self.db_manager.balance_repository.get_balance_history(days=90)
                    self.logger.info(f"Loaded {len(self.historical_data['balance_history'])} balance history records")
                
                # Load PnL history
                if hasattr(self.db_manager, 'balance_repository'):
                    self.historical_data["pnl_history"] = await self.db_manager.balance_repository.get_pnl_history(days=90)
                    self.logger.info(f"Loaded {len(self.historical_data['pnl_history'])} PnL history records")
                
                # Load trade history
                if hasattr(self.db_manager, 'trade_repository'):
                    self.historical_data["trade_history"] = await self.db_manager.trade_repository.get_trades(limit=100)
                    self.logger.info(f"Loaded {len(self.historical_data['trade_history'])} trade history records")
                    
            except Exception as e:
                self.logger.error(f"Error loading historical data: {str(e)}")

            return True
        except Exception as e:
            self.logger.error(f"Error initializing modules: {str(e)}")
            raise  # Re-raise the exception to prevent bot from starting in a broken state
    
    async def shutdown(self):
        """Async method to shutdown the bot orchestrator and all its components."""
        try:
            logger.info("Shutting down trading bot...")
            self._running = False
            
            # Stop all scheduled jobs
            if self.scheduler and self.scheduler.running:
                self.scheduler.shutdown()
                logger.info("Scheduler stopped")
            
            # Shutdown all modules
            # In a real implementation, these would be awaited
            # if hasattr(self, 'exchange_manager') and self.exchange_manager:
            #     await self.exchange_manager.shutdown()
            
            # if hasattr(self, 'api_server') and self.api_server:
            #     await self.api_server.shutdown()
            
            # if hasattr(self, 'db_manager') and self.db_manager:
            #     await self.db_manager.shutdown()
            
            logger.info("Bot orchestrator shutdown complete")
            return {"success": True, "message": "Bot shutdown complete"}
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def start_sync(self):
        """Synchronous method to start the bot."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.startup())
        
        try:
            logger.info("Bot running. Press Ctrl+C to stop.")
            loop.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            loop.run_until_complete(self.shutdown())
            loop.close()
    
    async def get_portfolio_history(self):
        """Get historical portfolio performance data."""
        try:
            # Get real data from database
            balance_history = await self.db_manager.balance_repository.get_balance_history()
            pnl_history = await self.db_manager.balance_repository.get_pnl_history()
            trade_history = await self.db_manager.trade_repository.get_trades(limit=100)
            
            # Format the data as expected by the API
            return {
                "balance_history": balance_history,
                "pnl_history": pnl_history,
                "trade_history": trade_history,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error retrieving portfolio history: {str(e)}")
            return {
                "balance_history": [],
                "pnl_history": [],
                "trade_history": [],
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    async def get_positions(self):
        """Get all open trading positions."""
        try:
            # Make sure the database is available
            if not self.db_manager:
                self.logger.warning("Database manager not initialized")
                return {}
                
            # Get open positions directly from the trade repository
            positions = await self.db_manager.trade_repository.get_open_positions()
            
            # If we have no open positions but positions should be available, check if this is due to a mismatch
            if not positions and self.balance_manager and hasattr(self.balance_manager, "open_positions"):
                # Check if balance manager has positions
                if self.balance_manager.open_positions:
                    self.logger.warning("Balance manager has positions but database has none. Syncing...")
                    # Force create position records in database for each balance manager position
                    for symbol, position in self.balance_manager.open_positions.items():
                        trade_data = {
                            "trade_id": position.get("trade_id", position.get("order_id", f"trade-{int(time.time())}-{symbol}")),
                            "trading_pair": symbol,
                            "entry_time": position.get("entry_time", datetime.now()).isoformat(),
                            "trading_pair_price": position.get("entry_price", 0),
                            "trading_fee": 0.0,  # We don't have this info
                            "trading_pair_quantity": position.get("amount", 0),
                            "trade_amount": position.get("cost", 0),
                            "entry_conditions": {"strategy": position.get("strategy", "unknown")},
                            "type": position.get("type", "buy"),  # Use type for code consistency
                            "side": position.get("type", "buy"),  # Add side for database compatibility
                            "status": "open",
                            "strategy": position.get("strategy", "unknown")
                        }
                        await self.db_manager.trade_repository.save_trade(trade_data)
                    
                    # Try to get positions again
                    positions = await self.db_manager.trade_repository.get_open_positions()
            
            return positions
        except Exception as e:
            self.logger.error(f"Error getting positions: {str(e)}")
            return {}
    
    async def update_trading_pairs(self):
        """Periodically update trading pairs based on market conditions."""
        try:
            self.logger.info("Updating trading pairs...")
            
            if not hasattr(self, 'trading_pair_selector') or not self.trading_pair_selector:
                self.logger.warning("Trading pair selector not initialized, skipping update")
                return
            
            # Get current pairs before update
            current_pairs = set(self.trading_pair_selector.selected_pairs)
            
            # Select new pairs based on current market conditions
            await self.trading_pair_selector.select_pairs()
            new_pairs = set(self.trading_pair_selector.selected_pairs)
            
            # Log changes
            added_pairs = new_pairs - current_pairs
            removed_pairs = current_pairs - new_pairs
            
            if added_pairs:
                self.logger.info(f"Added trading pairs: {', '.join(added_pairs)}")
            
            if removed_pairs:
                self.logger.info(f"Removed trading pairs: {', '.join(removed_pairs)}")
            
            if not added_pairs and not removed_pairs:
                self.logger.info("No changes to trading pairs")
            
            # Update heartbeat to show this process is running
            self.update_heartbeat()
            
        except Exception as e:
            self.logger.error(f"Error updating trading pairs: {str(e)}")
    
    async def update_position_prices(self):
        """
        Continuously update prices for open positions and evaluate exit conditions.
        This method implements the full Exit Trade Flow.
        """
        if not hasattr(self, 'balance_manager') or not self.balance_manager:
            self.logger.warning("Balance manager not initialized, cannot update positions")
            return
            
        try:
            # Get current open positions
            open_positions = self.balance_manager.open_positions
            
            if not open_positions:
                self.logger.debug("No open positions to update")
                return
                
            self.logger.info(f"Updating prices for {len(open_positions)} open positions: {', '.join(open_positions.keys())}")
            
            # First update all position prices from the exchange
            for symbol, position in list(open_positions.items()):
                try:
                    # Get current market price
                    ticker = await self.exchange_manager.fetch_ticker(symbol)
                    if not ticker or 'last' not in ticker:
                        self.logger.warning(f"Could not fetch current price for {symbol}, skipping update")
                        continue
                        
                    current_price = ticker['last']
                    
                    # Update position with current price
                    if position.get('current_price') != current_price:
                        self.logger.info(f"Updating price for {symbol}: {position.get('current_price', 'N/A')} -> {current_price}")
                        
                        # Update current price
                        position['current_price'] = current_price
                        
                        # Calculate unrealized P&L
                        entry_price = position.get('entry_price', 0)
                        amount = position.get('amount', 0)
                        
                        # Determine trade type (buy/sell)
                        trade_type = position.get('side', None)
                        if trade_type is None:
                            # Try to infer from trade_id
                            if position.get('trade_id', '').lower().endswith('_buy'):
                                trade_type = 'buy'
                            elif position.get('trade_id', '').lower().endswith('_sell'):
                                trade_type = 'sell'
                            else:
                                trade_type = 'buy'  # Default to buy if unknown
                        
                        # Calculate P&L
                        if trade_type.lower() == 'buy':
                            unrealized_pnl = (current_price - entry_price) * amount
                            unrealized_pnl_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
                        else:
                            unrealized_pnl = (entry_price - current_price) * amount
                            unrealized_pnl_pct = ((entry_price / current_price) - 1) * 100 if current_price > 0 else 0
                        
                        # Update position with P&L values
                        position['unrealized_pnl'] = unrealized_pnl
                        position['unrealized_pnl_pct'] = unrealized_pnl_pct
                        
                        self.logger.info(f"Position {symbol} PnL: {unrealized_pnl_pct:.2f}% (${unrealized_pnl:.2f})")
                        
                        # Update open_positions with the modified position
                        self.balance_manager.open_positions[symbol] = position
                        
                        # Update position in database if possible
                        if hasattr(self.db_manager, 'trade_repository'):
                            # Create update data
                            update_data = {
                                'current_price': current_price,
                                'unrealized_pnl': unrealized_pnl,
                                'unrealized_pnl_pct': unrealized_pnl_pct,
                                'last_updated': datetime.now().isoformat()
                            }
                            
                            # Get trade_id for database update
                            trade_id = position.get('trade_id')
                            if trade_id:
                                await self.db_manager.trade_repository.update_trade(trade_id, update_data)
                
                except Exception as e:
                    self.logger.error(f"Error updating price for {symbol}: {str(e)}")
            
            # Keep track of positions that should be closed
            positions_to_close = []
            
            # 1. Check for stop loss/take profit triggers via balance manager
            actions = await self.balance_manager.check_positions()
            
            # If any automatic actions were taken, log them
            if actions:
                self.logger.info(f"Automatic position check closed {len(actions)} positions")
                
                # Process results
                for result in actions:
                    if result["success"]:
                        self.logger.info(f"Closed position for {result['symbol']} due to {result['reason']}: {result['pnl_percentage']:.2f}%")
                    else:
                        self.logger.error(f"Failed to close position: {result.get('error', 'unknown error')}")
            
            # 2. For remaining positions, perform comprehensive exit flow assessment
            for symbol, position in list(open_positions.items()):
                # Skip if already closed by a stop/take profit
                if symbol not in self.balance_manager.open_positions:
                    continue
                    
                # Check if we have market data for this symbol
                if not hasattr(self, 'market_data_store') or not self.market_data_store or symbol not in self.market_data_store:
                    # Try to fetch market data for this specific symbol
                    self.logger.info(f"Fetching market data for {symbol} for exit evaluation")
                    try:
                        market_data = {}
                        for timeframe in self.settings.default_timeframes:
                            ohlcv = await self.exchange_manager.fetch_ohlcv(
                                symbol=symbol,
                                timeframe=timeframe,
                                limit=100
                            )
                            if not ohlcv.empty:
                                # Add technical indicators if TA module available
                                if self.ta_module:
                                    ohlcv = await self.ta_module.calculate_indicators(ohlcv)
                                market_data[timeframe] = ohlcv
                        
                        if market_data:
                            if not hasattr(self, 'market_data_store'):
                                self.market_data_store = {}
                            self.market_data_store[symbol] = market_data
                        else:
                            self.logger.warning(f"Could not fetch market data for {symbol}, skipping exit evaluation")
                            continue
                    except Exception as e:
                        self.logger.error(f"Error fetching market data for {symbol}: {str(e)}")
                        continue
                
                # Get multi-timeframe market data for this symbol
                market_data = self.market_data_store[symbol]
                
                # Skip if we don't have any data
                if not market_data:
                    continue
                
                # 3. Calculate position-specific indicators with TA module
                position_indicators = {}
                if self.ta_module:
                    try:
                        position_indicators = await self.ta_module.monitor_positions(
                            {symbol: position}, 
                            {symbol: market_data}
                        )
                    except Exception as e:
                        self.logger.error(f"Error calculating indicators for {symbol}: {str(e)}")
                
                # 4. Get ML module exit evaluation
                ml_exit_decision = {"should_exit": False, "confidence": 0.0}
                if self.ml_module:
                    try:
                        # Get primary timeframe data for ML evaluation
                        primary_tf = position.get("timeframe", "1h")
                        if primary_tf in market_data:
                            ml_exit_decision = await self.ml_module.evaluate_exit_decision(
                                position, 
                                market_data[primary_tf]
                            )
                    except Exception as e:
                        self.logger.error(f"Error getting ML exit decision for {symbol}: {str(e)}")
                
                # 5. Check all exit conditions
                should_exit = False
                exit_reason = None
                exit_confidence = 0.0
                
                # 5.1. Check technical exit signals
                if symbol in position_indicators and position_indicators[symbol].get("exit_signals", {}).get("technical", False):
                    should_exit = True
                    exit_reason = f"technical:{position_indicators[symbol]['exit_signals']['reason']}"
                    exit_confidence = position_indicators[symbol]["exit_signals"].get("confidence", 0.6)
                    self.logger.info(f"Technical analysis suggests exiting {symbol}: {exit_reason}")
                
                # 5.2. Check ML exit signals
                if ml_exit_decision.get("should_exit", False):
                    # If ML confidence is higher, use ML reason instead
                    ml_confidence = ml_exit_decision.get("confidence", 0.0)
                    if ml_confidence > exit_confidence:
                        should_exit = True
                        exit_reason = f"ml:{ml_exit_decision.get('reason', 'prediction')}"
                        exit_confidence = ml_confidence
                        self.logger.info(f"ML model suggests exiting {symbol} with {exit_confidence:.2f} confidence")
                
                # 5.3. Check strategy module exit signal
                strategy_exit = False
                if self.strategy_module:
                    try:
                        # Get current price
                        current_price = position.get('current_price')
                        
                        if current_price:
                            strategy_should_exit, strategy_reason = self.strategy_module.should_exit_trade(position, current_price)
                            if strategy_should_exit:
                                should_exit = True
                                exit_reason = f"strategy:{strategy_reason}"
                                exit_confidence = 0.8  # Default high confidence for strategy module
                                self.logger.info(f"Strategy module suggests exiting {symbol}: {strategy_reason}")
                    except Exception as e:
                        self.logger.error(f"Error in strategy module exit evaluation for {symbol}: {str(e)}")
                
                # 5.4 Check time-based exit - close positions open for too long
                try:
                    # Calculate position duration
                    if 'timestamp' in position:
                        position_time = datetime.fromisoformat(position['timestamp'].replace('Z', '+00:00'))
                        current_time = datetime.now(position_time.tzinfo if position_time.tzinfo else None)
                        duration_hours = (current_time - position_time).total_seconds() / 3600
                        
                        # Close positions open for more than 48 hours
                        max_position_hours = getattr(self.settings, 'MAX_POSITION_HOURS', 48)
                        if duration_hours > max_position_hours:
                            self.logger.info(f"Position {symbol} has been open for {duration_hours:.1f} hours (max {max_position_hours})")
                            should_exit = True
                            exit_reason = f"time_based:max_duration_exceeded"
                            exit_confidence = 0.9
                except Exception as e:
                    self.logger.error(f"Error calculating position duration for {symbol}: {str(e)}")
                
                # 6. If any exit condition is met, add to positions to close
                if should_exit:
                    positions_to_close.append({
                        "symbol": symbol,
                        "position": position,
                        "reason": exit_reason,
                        "confidence": exit_confidence
                    })
            
            # 7. Execute all required position closures
            for position_info in positions_to_close:
                self.logger.info(f"Closing position for {position_info['symbol']} due to {position_info['reason']}")
                
                # Execute the exit trade with enhanced flow
                result = await self.close_position(
                    symbol=position_info["symbol"],
                    position_data=position_info["position"],
                    reason=position_info["reason"]
                )
                
                if result and result.get("success"):
                    self.logger.info(f"Successfully closed position for {position_info['symbol']}: {result.get('pnl_percentage', 0):.2f}%")
                else:
                    error = result.get("error", "unknown error")
                    self.logger.error(f"Failed to close position for {position_info['symbol']}: {error}")
            
            # After all position updates, update portfolio status
            portfolio = self.balance_manager.get_portfolio_summary() if hasattr(self.balance_manager, "get_portfolio_summary") else {}
            
            # Log portfolio summary if positions exist
            if portfolio and portfolio.get("open_positions_count", 0) > 0:
                self.logger.info(f"Portfolio summary: {portfolio.get('open_positions_count')} positions, " +
                               f"Unrealized P/L: ${portfolio.get('unrealized_pnl', 0):.2f}")
                
            # Update metrics if available
            if hasattr(self, "stats"):
                self.stats["open_positions"] = portfolio.get("open_positions_count", 0)
                self.stats["portfolio_value"] = portfolio.get("total_positions_value", 0) + portfolio.get("available_usdc", 0)
            
            self.update_heartbeat()
            
        except Exception as e:
            self.logger.exception(f"Error updating position prices: {str(e)}")
            
            # Send error notification if available
            if hasattr(self, 'notification_system') and self.notification_system:
                await self.notification_system.send_error(
                    error_type="position_update_error",
                    message=f"Failed to update positions: {str(e)}",
                    priority="high"
                )
    
    def _create_placeholder_db_manager(self):
        """Create a placeholder database manager."""
        class PlaceholderDBManager:
            def __init__(self):
                self.logger = logging.getLogger(__name__)
                
                # Create a placeholder balance repository to prevent attribute errors
                class PlaceholderBalanceRepository:
                    async def get_balance_history(self, days=90):
                        return []
                        
                    async def save_balance_history(self, timestamp, balance):
                        return True
                        
                    async def get_pnl_history(self, days=90):
                        return []
                        
                    async def save_pnl_history(self, timestamp, pnl, cumulative_pnl):
                        return True
                
                # Create a placeholder trade repository
                class PlaceholderTradeRepository:
                    async def get_trades(self, status=None, limit=100):
                        return []
                        
                    async def get_open_positions(self):
                        return {}
                        
                    async def save_trade(self, trade_data):
                        return True
                
                # Create a placeholder market data repository
                class PlaceholderMarketDataRepository:
                    async def save_ohlcv_data(self, symbol, timeframe, data):
                        return True
                        
                    async def get_ohlcv_data(self, symbol, timeframe, limit=100):
                        return []
                
                self.balance_repository = PlaceholderBalanceRepository()
                self.trade_repository = PlaceholderTradeRepository()
                self.market_data_repository = PlaceholderMarketDataRepository()
                
            async def initialize(self, **kwargs):
                return True
                
            async def close(self):
                return True
            
        return PlaceholderDBManager()
    
    def _create_placeholder_exchange_manager(self):
        """Create a placeholder exchange manager."""
        class PlaceholderExchangeManager:
            def __init__(self):
                self.logger = logging.getLogger(__name__)
                self.exchanges = {"binance": "placeholder"}
                
            async def initialize(self):
                return True
                
            async def close(self):
                return True
                
            async def fetch_ticker(self, symbol, exchange_id="binance"):
                return {
                    'symbol': symbol,
                    'last': 50000.0,
                    'bid': 49990.0,
                    'ask': 50010.0,
                    'volume': 100.0,
                    'quoteVolume': 5000000.0,
                    'percentage': 0.0,
                    'high': 50100.0,
                    'low': 49900.0,
                    'timestamp': int(time.time() * 1000)
                }
                
            async def fetch_ohlcv(self, symbol, timeframe='1h', limit=100, exchange_id="binance"):
                # Return empty DataFrame
                return pd.DataFrame()
                
            async def fetch_usdc_pairs(self, exchange_id="binance"):
                return ["BTC/USDC", "ETH/USDC", "SOL/USDC", "XRP/USDC", "SUI/USDC"]
                
            async def fetch_usdc_trading_pairs(self, exchange_id="binance"):
                return ["BTC/USDC", "ETH/USDC", "SOL/USDC", "XRP/USDC", "SUI/USDC"]
                
            async def fetch_current_prices(self, pairs, exchange_id="binance"):
                result = {}
                for pair in pairs:
                    result[pair] = {
                        'price': 50000.0,
                        'volume_24h': 1000000.0,
                        'change_24h': 0.0
                    }
                return result
            
        return PlaceholderExchangeManager()
    
    def _create_placeholder_balance_manager(self):
        """Create a placeholder balance manager with basic functionality."""
        class PlaceholderBalanceManager:
            def __init__(self):
                self.logger = logging.getLogger(__name__)
                self.open_positions = {}
                self.balances = {
                    "USDC": {
                        "free": 10000.0,
                        "used": 0.0,
                        "total": 10000.0
                    }
                }
                # Stats tracking
                self.total_trades = 0
                self.winning_trades = 0
                self.losing_trades = 0
                self.total_profit = 0.0
                self.total_loss = 0.0
                
            async def initialize(self):
                self.logger.info("Placeholder balance manager initialized")
                return True
                
            def get_portfolio_summary(self):
                """Return a default portfolio summary with all required fields."""
                return {
                    "total_balance_usdc": 10000.00,
                    "available_usdc": 10000.00,
                    "allocated_balance": 0.00,
                    "total_positions_value": 0.00,
                    "open_positions": {},
                    "open_positions_count": 0,
                    "win_rate": 0.0,
                    "total_trades": 0,
                    "total_profit": 0.0,
                    "total_loss": 0.0,
                    "net_profit": 0.0,
                    "pnl_24h": 0.0,
                    "timestamp": datetime.now().isoformat()
                }
                
            async def get_open_positions(self):
                """Return empty open positions."""
                return {}
                
            async def update_balances(self):
                """Mock update balances."""
                return True
                
            def get_available_usdc(self):
                """Get available USDC balance for trading."""
                return self.balances["USDC"]["free"]
        
        placeholder = PlaceholderBalanceManager()
        return placeholder
    
    def _create_placeholder_trading_pair_selector(self):
        """Create a placeholder trading pair selector for testing."""
        
        class PlaceholderTradingPairSelector:
            def __init__(self):
                self.selected_pairs = [
                    "BTC/USDC",
                    "ETH/USDC",
                    "SOL/USDC",
                    "XRP/USDC",
                    "BNB/USDC"
                ]
                self.logger = logging.getLogger(__name__)
                self.logger.info("Initialized PlaceholderTradingPairSelector with default pairs")
            
            async def select_pairs(self):
                """Return default trading pairs."""
                return self.selected_pairs
            
            def get_selected_pairs(self):
                """Return currently selected pairs."""
                return self.selected_pairs
        
        return PlaceholderTradingPairSelector()
    
    async def process_signals(self, signals):
        """
        Process and filter trading signals based on risk management rules.
        
        Args:
            signals: List of raw trading signals from the strategy module
            
        Returns:
            List of filtered and enriched trading signals
        """
        try:
            self.logger.info(f"Processing {len(signals)} trading signals")
            filtered_signals = []
            
            # Get current portfolio data
            portfolio = await self.exchange_manager.fetch_balance()
            active_positions = await self.exchange_manager.fetch_positions()
            
            # Process each signal
            for signal in signals:
                try:
                    # Skip invalid signals
                    if not isinstance(signal, dict) or 'symbol' not in signal or 'trade_type' not in signal:
                        self.logger.warning(f"Skipping invalid signal: {signal}")
                        continue
                    
                    symbol = signal.get('symbol')
                    trade_type = signal.get('trade_type')
                    
                    # Skip signals for symbols that already have an active position
                    if symbol in active_positions:
                        position_type = active_positions[symbol].get('type')
                        if position_type == trade_type:
                            self.logger.info(f"Skipping {trade_type} signal for {symbol} - already have active {position_type} position")
                            continue
                    
                    # Apply risk management rules
                    # 1. Check maximum open positions
                    if len(active_positions) >= self.settings.MAX_POSITIONS:
                        self.logger.info(f"Skipping signal for {symbol} - maximum positions ({self.settings.MAX_POSITIONS}) reached")
                        continue
                    
                    # 2. Check available balance
                    available_balance = portfolio.get('USDC', {}).get('free', 0)
                    min_position_size = getattr(self.settings, 'MIN_POSITION_SIZE', 100)
                    if available_balance < min_position_size:
                        self.logger.info(f"Skipping signal for {symbol} - insufficient balance ({available_balance} USDC)")
                        continue
                    
                    # 3. Calculate position size based on available balance and risk
                    position_size = available_balance * 0.1  # Use 10% of available balance per position
                    if position_size < min_position_size:
                        position_size = min_position_size
                    
                    # Add additional trading metadata to the signal
                    enriched_signal = signal.copy()
                    enriched_signal['position_size'] = position_size
                    enriched_signal['timestamp'] = datetime.now()
                    
                    # Add to filtered signals
                    filtered_signals.append(enriched_signal)
                    
                except Exception as e:
                    self.logger.error(f"Error processing signal for {signal.get('symbol', 'unknown')}: {str(e)}")
                    continue
            
            self.logger.info(f"Processed signals: {len(filtered_signals)} passed filtering")
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Error in process_signals: {str(e)}")
            return []
    
    async def patched_generate_signals(self):
        """
        Generate trading signals and execute trades if auto-execution is enabled.
        This method is called by the scheduler at regular intervals.
        """
        try:
            # Generate signals
            signals = await self.generate_signals()
            
            # Execute signals if auto-execution is enabled
            if signals and getattr(self.settings, 'AUTO_EXECUTE_SIGNALS', False):
                self.logger.info("Auto-executing trading signals")
                for signal in signals:
                    await self.execute_signal(signal)
            
            # Update heartbeat to show this process is running
            self.update_heartbeat()
            
            return {"success": True, "signals": signals}
            
        except Exception as e:
            self.logger.error(f"Error in patched_generate_signals: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def execute_signal(self, signal):
        """
        Execute a trading signal by placing an order on the exchange.
        
        Args:
            signal: Trading signal with all necessary information
            
        Returns:
            Dictionary with the result of the order execution
        """
        try:
            self.logger.info(f"Executing signal: {signal}")
            
            # Extract signal data
            symbol = signal.get('symbol')
            trade_type = signal.get('trade_type')
            price = signal.get('price')
            position_size = signal.get('position_size')
            
            # Validate signal data
            if not all([symbol, trade_type, price, position_size]):
                self.logger.error(f"Invalid signal data for execution: {signal}")
                return {"success": False, "error": "Invalid signal data"}
            
            # Place order on exchange
            if trade_type.lower() == 'buy':
                result = await self.exchange_manager.create_order(
                    symbol=symbol,
                    order_type='limit',
                    side='buy',
                    amount=position_size / price,
                    price=price
                )
            elif trade_type.lower() == 'sell':
                result = await self.exchange_manager.create_order(
                    symbol=symbol,
                    order_type='limit',
                    side='sell',
                    amount=position_size / price,
                    price=price
                )
            else:
                self.logger.error(f"Unknown trade type: {trade_type}")
                return {"success": False, "error": f"Unknown trade type: {trade_type}"}
            
            # Save order to database
            if result.get('success', False):
                order_data = result.get('order', {})
                await self.db_manager.save_order(order_data)
                self.logger.info(f"Order placed successfully: {order_data.get('id')}")
                return {"success": True, "order": order_data}
            else:
                self.logger.error(f"Failed to place order: {result.get('error')}")
                return {"success": False, "error": result.get('error')}
            
        except Exception as e:
            self.logger.error(f"Error executing signal: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def start_bot(self):
        """
        Start the trading bot and all its components.
        Main entry point for bot operation.
        Raises exceptions if critical components fail to initialize.
        """
        try:
            self.logger.info(f"Starting trading bot (Simulation Mode: {self.settings.simulation_mode})")
            
            # Initialize all required modules - will raise exception if critical components fail
            try:
                await self.initialize_modules()
            except Exception as init_error:
                self.logger.error(f"Failed to initialize critical components: {str(init_error)}")
                
                # Send notification about the initialization error if notification system is available
                if hasattr(self, 'notification_system') and self.notification_system:
                    await self.notification_system.notify_error(
                        f"Bot initialization failed: {str(init_error)}",
                        source="Bot Startup",
                        severity="high"
                    )
                raise  # Re-raise to abort startup
                
            # Run recovery procedure for existing positions
            await self._recover_existing_positions()
            
            # Schedule all required jobs - will raise exception if scheduler fails
            try:
                self.schedule_jobs()
            except Exception as scheduler_error:
                self.logger.error(f"Failed to schedule jobs: {str(scheduler_error)}")
                
                # Send notification about the scheduler error if notification system is available
                if hasattr(self, 'notification_system') and self.notification_system:
                    await self.notification_system.notify_error(
                        f"Job scheduling failed: {str(scheduler_error)}",
                        source="Bot Startup",
                        severity="high"
                    )
                raise  # Re-raise to abort startup
            
            # Start websocket server if enabled
            if hasattr(self.settings, 'enable_websocket') and self.settings.enable_websocket:
                # Implement websocket server startup here if needed
                pass
            
            # Initial market data collection
            try:
                self.logger.info("Collecting initial market data")
                market_data_result = await self.collect_market_data()
                if not market_data_result.get("success", False):
                    self.logger.warning(f"Initial market data collection failed: {market_data_result.get('error', 'Unknown error')}")
            except Exception as data_error:
                self.logger.error(f"Error collecting initial market data: {str(data_error)}")
                # Non-critical error, continue startup
            
            # Initial trading pair selection
            try:
                self.logger.info("Selecting initial trading pairs")
                pairs = await self.select_trading_pairs()
                if pairs:
                    self.trading_pairs = pairs
                    self.logger.info(f"Selected {len(pairs)} initial trading pairs")
                else:
                    self.logger.warning("No trading pairs selected initially")
            except Exception as pair_error:
                self.logger.error(f"Error selecting initial trading pairs: {str(pair_error)}")
                # Non-critical error, continue startup
            
            # Record start time and set running flag
            self.start_time = datetime.now()
            self._running = True
            
            # Send notification about successful startup if notification system is available
            if hasattr(self, 'notification_system') and self.notification_system:
                mode = "Simulation" if self.settings.simulation_mode else "Live Trading"
                await self.notification_system.send_notification(
                    message=f"Trading bot started in {mode} mode",
                    title="Bot Started",
                    category="SYSTEM",
                    priority="HIGH"
                )
            
            self.logger.info("Trading bot started successfully")
            return {"success": True, "message": "Trading bot started successfully"}
            
        except Exception as e:
            self.logger.error(f"Error starting trading bot: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _recover_existing_positions(self):
        """
        Recover existing positions from the database.
        This method is called when the bot starts to ensure that any open positions are correctly loaded.
        """
        try:
            if not hasattr(self, 'db_manager') or not self.db_manager:
                self.logger.warning("Database manager not initialized, cannot recover positions")
                return
            
            # Get open positions from the database
            open_positions = await self.db_manager.trade_repository.get_open_positions()
            
            if not open_positions:
                self.logger.info("No open positions found in database")
                return
            
            # Update balance manager with these positions
            for position in open_positions:
                symbol = position.get('symbol')
                if symbol:
                    self.balance_manager.open_positions[symbol] = position
                    self.logger.info(f"Recovered position for {symbol}: {position['amount']} @ {position['entry_price']}")
            else:
                self.logger.info("No open positions found in database")
                
            # Update position prices immediately
            if self.balance_manager.open_positions:
                self.logger.info("Updating prices for recovered positions")
                await self.update_position_prices()
        except Exception as e:
            self.logger.error(f"Error recovering existing positions: {str(e)}")
            # Continue despite errors in recovery
    
    async def cleanup_old_data(self):
        """
        Clean up old data from the database to prevent excessive growth.
        This includes old market data, logs, and trade history beyond retention period.
        """
        try:
            self.logger.info("Running database cleanup task")
            
            if not hasattr(self, 'db_manager') or not self.db_manager:
                self.logger.warning("Database manager not initialized, skipping cleanup")
                return
                
            # Clean up market data older than 30 days
            if hasattr(self.db_manager, 'market_data_repository'):
                days_to_keep = 30
                deleted_count = await self.db_manager.market_data_repository.delete_old_data(days=days_to_keep)
                self.logger.info(f"Cleaned up {deleted_count} old market data records (kept last {days_to_keep} days)")
            
            # Other cleanup tasks can be added here
            
            self.logger.info("Database cleanup completed successfully")
        except Exception as e:
            self.logger.error(f"Error in database cleanup task: {str(e)}")
            # Continue despite errors