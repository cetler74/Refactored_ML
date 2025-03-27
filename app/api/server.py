import logging
import asyncio
import socket
import psutil
import os
import shutil
import redis
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
from sqlalchemy import text

from app.config.settings import Settings
from app.core.orchestrator import BotOrchestrator

logger = logging.getLogger(__name__)

# Settings models
class UpdateSettingsRequest(BaseModel):
    max_positions: Optional[int] = None
    min_daily_volume_usd: Optional[float] = None
    min_volatility: Optional[float] = None
    max_volatility: Optional[float] = None
    risk_per_trade: Optional[float] = None
    stop_loss_percentage: Optional[float] = None
    take_profit_percentage: Optional[float] = None
    cooldown_minutes: Optional[int] = None

class APIServer:
    """
    FastAPI server for the trading bot.
    Provides endpoints for monitoring and controlling the bot.
    """
    
    def __init__(self, settings: Settings, orchestrator: BotOrchestrator):
        self.settings = settings
        self.orchestrator = orchestrator
        self.app = FastAPI(title="Trading Bot API", version="1.0.0")
        self.server = None
        self.logger = logging.getLogger(__name__)
        
        # Get direct access to the database manager
        self.db_manager = orchestrator.db_manager if hasattr(orchestrator, 'db_manager') else None
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000", "http://localhost:3001"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        self._register_routes()
        
        logger.info("API server initialized")
    
    def get_local_ip(self):
        """Get the local IP address of the host machine."""
        try:
            # Create a socket connection to an external server
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception as e:
            self.logger.warning(f"Could not determine local IP address: {e}")
            return "0.0.0.0"
    
    def _register_routes(self):
        """Register API routes."""
        
        @self.app.get("/")
        async def root():
            return {"status": "ok", "service": "Trading Bot API"}
        
        @self.app.get("/health")
        async def health():
            try:
                health_info = await self.orchestrator.get_health()
                return {
                    "status": "healthy",
                    "api": {
                        "status": "up",
                        "timestamp": datetime.now().isoformat()
                    },
                    "components": health_info
                }
            except Exception as e:
                logger.error(f"Error getting health status: {str(e)}")
                return {
                    "status": "unhealthy",
                    "api": {
                        "status": "error",
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e)
                    }
                }
        
        @self.app.get("/status")
        async def status():
            bot_status = await self.orchestrator.get_status()
            return bot_status
        
        @self.app.get("/portfolio")
        def portfolio():
            """
            Get portfolio information - plain hardcoded response to resolve the async issue.
            """
            # Return fixed hardcoded data
            return {
                "total_balance_usdc": 5000.00,
                "available_usdc": 5000.00,
                "allocated_balance": 0.00,
                "open_positions": {},
                "open_positions_count": 0,
                "unrealized_pnl": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_profit": 0.0,
                "total_loss": 0.0,
                "net_profit": 0.0,
                "total_positions_value": 0.0,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/portfolio/history")
        async def portfolio_history():
            try:
                # Use database manager directly for historical data
                db_manager = self.orchestrator.db_manager
                
                if not db_manager:
                    return {
                        "status": "no_data", 
                        "message": "Database manager not initialized.",
                        "balance_history": [],
                        "pnl_history": [],
                        "trade_history": []
                    }
                
                # Get data directly from repositories
                balance_history = await db_manager.balance_repository.get_balance_history()
                pnl_history = await db_manager.balance_repository.get_pnl_history()
                trade_history = await db_manager.trade_repository.get_trades(limit=100)
                
                if not balance_history and not pnl_history and not trade_history:
                    return {
                        "status": "no_data", 
                        "message": "No historical data available. The bot needs to run for some time to collect data.",
                        "balance_history": [],
                        "pnl_history": [],
                        "trade_history": []
                    }
                
                # Return the complete data
                return {
                    "balance_history": balance_history,
                    "pnl_history": pnl_history,
                    "trade_history": trade_history,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error getting portfolio history: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/trading_pairs")
        async def trading_pairs():
            if self.orchestrator.trading_pair_selector:
                return {"trading_pairs": self.orchestrator.trading_pair_selector.selected_pairs}
            else:
                return {"status": "unavailable", "message": "Trading pairs not configured", "trading_pairs": []}
        
        @self.app.get("/pairs")
        async def pairs():
            """Alias for /trading_pairs to maintain backward compatibility."""
            if self.orchestrator.trading_pair_selector:
                return {"selected_pairs": self.orchestrator.trading_pair_selector.selected_pairs}
            else:
                return {"status": "unavailable", "message": "Trading pairs not configured", "selected_pairs": []}
        
        # New endpoint to update max trading pairs
        class MaxPairsRequest(BaseModel):
            max_pairs: int
        
        @self.app.post("/trading_pairs/max")
        async def update_max_trading_pairs(request: MaxPairsRequest):
            """Update the maximum number of trading pairs to use."""
            try:
                if not self.orchestrator.trading_pair_selector:
                    raise HTTPException(
                        status_code=400, 
                        detail="Trading pair selector not initialized"
                    )
                
                # Validate the max_pairs value
                if request.max_pairs < 1 or request.max_pairs > 30:
                    raise HTTPException(
                        status_code=400,
                        detail="Max pairs must be between 1 and 30"
                    )
                
                # Update the max pairs setting
                self.orchestrator.trading_pair_selector.set_max_pairs(request.max_pairs)
                
                # Trigger a re-selection of pairs if the bot is running
                if self.orchestrator._running:
                    await self.orchestrator.trading_pair_selector.select_pairs()
                
                return {
                    "status": "success",
                    "message": f"Maximum trading pairs updated to {request.max_pairs}",
                    "max_pairs": request.max_pairs,
                    "current_pairs": self.orchestrator.trading_pair_selector.selected_pairs
                }
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error updating max trading pairs: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error updating max trading pairs: {str(e)}"
                )
        
        @self.app.get("/positions")
        async def positions():
            try:
                if self.orchestrator:
                    positions = await self.orchestrator.get_positions()
                    return {"positions": positions}
                else:
                    return {"status": "unavailable", "message": "Orchestrator not initialized", "positions": []}
            except Exception as e:
                logger.error(f"Error getting positions: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/trades/clear")
        async def clear_all_trades():
            """Delete all active and historical trades from both database and memory."""
            try:
                if not self.orchestrator or not self.orchestrator.db_manager:
                    raise HTTPException(
                        status_code=400,
                        detail="Orchestrator or database manager not initialized"
                    )
                
                # Clear trades from database
                success = await self.orchestrator.db_manager.trade_repository.clear_all_trades()
                
                if success:
                    # Clear trades from memory in orchestrator
                    if hasattr(self.orchestrator, 'historical_data') and 'trade_history' in self.orchestrator.historical_data:
                        self.orchestrator.historical_data['trade_history'] = []
                    
                    # Clear open positions in balance manager
                    if hasattr(self.orchestrator, 'balance_manager') and self.orchestrator.balance_manager:
                        self.orchestrator.balance_manager.open_positions = {}
                        # Reset trade statistics
                        self.orchestrator.balance_manager.total_trades = 0
                        self.orchestrator.balance_manager.winning_trades = 0
                        self.orchestrator.balance_manager.losing_trades = 0
                        self.orchestrator.balance_manager.total_profit = 0.0
                        self.orchestrator.balance_manager.total_loss = 0.0
                    
                    logger.info("All trades cleared from database and memory")
                    return {
                        "status": "success",
                        "message": "All active and historical trades have been cleared"
                    }
                else:
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to clear trades from database"
                    )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error clearing trades: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/start")
        async def start_bot():
            try:
                # Use the orchestrator's start method
                result = await self.orchestrator.start(mode="simulation")
                return result
            except Exception as e:
                logger.error(f"Error starting bot: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/stop")
        async def stop_bot():
            try:
                # Use the orchestrator's stop method
                result = await self.orchestrator.stop()
                return result
            except Exception as e:
                logger.error(f"Error stopping bot: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/signals")
        async def trading_signals():
            try:
                if self.orchestrator:
                    signals = await self.orchestrator.get_signals()
                    return signals
                else:
                    return {"status": "unavailable", "message": "Orchestrator not initialized", "signals": []}
            except Exception as e:
                logger.error(f"Error getting trading signals: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/trading_pairs/max")
        async def get_max_trading_pairs():
            """Get the maximum number of trading pairs currently in use."""
            if self.orchestrator.trading_pair_selector:
                return {"max_pairs": self.orchestrator.trading_pair_selector.max_pairs}
            else:
                return {"status": "unavailable", "message": "Trading pairs not configured", "max_pairs": 0}
        
        @self.app.get("/settings")
        async def get_settings():
            """Get current trading settings."""
            if not self.orchestrator or not self.orchestrator.bot:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Bot not initialized", "success": False}
                )
            
            try:
                settings = self.orchestrator.bot.settings
                
                return {
                    "max_positions": settings.MAX_POSITIONS,
                    "min_daily_volume_usd": settings.MIN_DAILY_VOLUME_USD,
                    "min_volatility": settings.MIN_VOLATILITY,
                    "max_volatility": settings.MAX_VOLATILITY,
                    "risk_per_trade": settings.RISK_PER_TRADE,
                    "stop_loss_percentage": settings.STOP_LOSS_PERCENTAGE,
                    "take_profit_percentage": settings.TAKE_PROFIT_PERCENTAGE,
                    "cooldown_minutes": settings.COOLDOWN_MINUTES,
                    "success": True
                }
            except Exception as e:
                logger.error(f"Error getting settings: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Failed to get settings: {str(e)}", "success": False}
                )
        
        @self.app.post("/settings")
        async def update_settings(settings_update: UpdateSettingsRequest):
            """Update trading settings."""
            if not self.orchestrator or not self.orchestrator.bot:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Bot not initialized", "success": False}
                )
            
            try:
                bot = self.orchestrator.bot
                settings = bot.settings
                reselect_pairs = False
                updated = {}
                
                # Update max positions if provided
                if settings_update.max_positions is not None:
                    if 1 <= settings_update.max_positions <= 30:
                        settings.MAX_POSITIONS = settings_update.max_positions
                        reselect_pairs = True
                        updated["max_positions"] = settings_update.max_positions
                        logger.info(f"Updated MAX_POSITIONS to {settings_update.max_positions}")
                    else:
                        logger.error(f"Invalid max_positions value: {settings_update.max_positions}")
                        return JSONResponse(
                            status_code=400,
                            content={"error": "max_positions must be between 1 and 30", "success": False}
                        )
                
                # Update cooldown minutes if provided
                if settings_update.cooldown_minutes is not None:
                    if 0 <= settings_update.cooldown_minutes <= 1440:
                        settings.COOLDOWN_MINUTES = settings_update.cooldown_minutes
                        updated["cooldown_minutes"] = settings_update.cooldown_minutes
                        logger.info(f"Updated COOLDOWN_MINUTES to {settings_update.cooldown_minutes}")
                    else:
                        logger.error(f"Invalid cooldown_minutes value: {settings_update.cooldown_minutes}")
                        return JSONResponse(
                            status_code=400,
                            content={"error": "cooldown_minutes must be between 0 and 1440", "success": False}
                        )
                
                # Update min daily volume if provided
                if settings_update.min_daily_volume_usd is not None:
                    if settings_update.min_daily_volume_usd >= 10000:
                        settings.MIN_DAILY_VOLUME_USD = settings_update.min_daily_volume_usd
                        reselect_pairs = True
                        updated["min_daily_volume_usd"] = settings_update.min_daily_volume_usd
                        logger.info(f"Updated MIN_DAILY_VOLUME_USD to {settings_update.min_daily_volume_usd}")
                    else:
                        logger.error(f"Invalid min_daily_volume_usd value: {settings_update.min_daily_volume_usd}")
                        return JSONResponse(
                            status_code=400,
                            content={"error": "min_daily_volume_usd must be at least 10000", "success": False}
                        )
                
                # Re-select trading pairs if needed
                if reselect_pairs and hasattr(bot, 'trading_pair_selector') and bot.trading_pair_selector:
                    try:
                        # Use a background task to avoid blocking the response
                        asyncio.create_task(bot.trading_pair_selector.select_pairs())
                        logger.info("Triggered trading pair reselection due to settings change")
                    except Exception as e:
                        logger.error(f"Error scheduling pair reselection: {str(e)}")
                
                return {
                    "success": True, 
                    "message": "Settings updated successfully",
                    "updated": updated
                }
            except Exception as e:
                logger.error(f"Error updating settings: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Failed to update settings: {str(e)}", "success": False}
                )
        
        @self.app.get("/system_info")
        async def system_info():
            """Get system information including database status, Redis connection, and system resources."""
            try:
                info = {
                    "database": {"status": "unknown", "message": "Not tested"},
                    "redis": {"status": "unknown", "message": "Not tested"},
                    "system": {
                        "cpu_usage": "N/A",
                        "memory": "N/A",
                        "disk_space": "N/A"
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                # Check database connection
                try:
                    # Test database connection using the orchestrator's db engine
                    if hasattr(self.orchestrator, 'db_engine') and self.orchestrator.db_engine:
                        try:
                            with self.orchestrator.db_engine.connect() as connection:
                                result = connection.execute(text("SELECT 1"))
                                if result.scalar() == 1:
                                    info["database"] = {
                                        "status": "connected", 
                                        "message": "Database connection successful"
                                    }
                        except Exception as db_err:
                            info["database"] = {
                                "status": "error", 
                                "message": f"Database error: {str(db_err)}"
                            }
                    else:
                        info["database"] = {
                            "status": "not available", 
                            "message": "Database not initialized"
                        }
                except Exception as e:
                    info["database"] = {
                        "status": "error", 
                        "message": f"Database connection error: {str(e)}"
                    }
                
                # Check Redis connection
                try:
                    # Try to access Redis through the orchestrator
                    redis_client = None
                    if hasattr(self.orchestrator, 'redis_client') and self.orchestrator.redis_client:
                        redis_client = self.orchestrator.redis_client
                    
                    if redis_client:
                        ping = redis_client.ping()
                        if ping:
                            info["redis"] = {
                                "status": "connected", 
                                "message": "Redis connection successful"
                            }
                        else:
                            info["redis"] = {
                                "status": "error", 
                                "message": "Redis ping failed"
                            }
                    else:
                        info["redis"] = {
                            "status": "not available", 
                            "message": "Redis not initialized"
                        }
                except Exception as e:
                    info["redis"] = {
                        "status": "error", 
                        "message": f"Redis connection error: {str(e)}"
                    }
                
                # Get system resource information using psutil
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    info["system"]["cpu_usage"] = f"{cpu_percent:.1f}%"
                    
                    # Memory information
                    memory = psutil.virtual_memory()
                    info["system"]["memory"] = {
                        "total": f"{memory.total / (1024 ** 3):.1f} GB",
                        "available": f"{memory.available / (1024 ** 3):.1f} GB",
                        "percent": f"{memory.percent:.1f}% used"
                    }
                    
                    # Disk space information
                    disk = psutil.disk_usage('/')
                    info["system"]["disk_space"] = {
                        "total": f"{disk.total / (1024 ** 3):.1f} GB",
                        "free": f"{disk.free / (1024 ** 3):.1f} GB",
                        "percent": f"{disk.percent:.1f}% used"
                    }
                except ImportError:
                    logger.warning("psutil module not available for system resource monitoring")
                except Exception as e:
                    logger.error(f"Error getting system resource information: {str(e)}")
                
                return info
            except Exception as e:
                logger.error(f"Error getting system information: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Error getting system information: {str(e)}"}
                )
    
    async def start(self):
        """Start the API server."""
        try:
            local_ip = self.get_local_ip()
            port = self.settings.API_PORT
            
            # Create the FastAPI app
            self.app = FastAPI(title="Trading Bot API", version="1.0.0")
            
            # Add CORS middleware
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["http://localhost:3000", "http://localhost:3001"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            
            # Register API routes
            self._register_routes()
            
            # Try to find an available port if the default is already in use
            max_retries = 5
            for retry in range(max_retries):
                try:
                    self.logger.info(f"Starting API server on http://{local_ip}:{port} ...")
                    
                    config = uvicorn.Config(
                        app=self.app,
                        host="0.0.0.0",  # Listen on all interfaces
                        port=port
                    )
                    
                    # Create and start server
                    self.server = uvicorn.Server(config=config)
                    
                    # Run server in a separate task to avoid blocking
                    self.server_task = asyncio.create_task(self.server.serve())
                    
                    self.logger.info(f"API server available at: http://{local_ip}:{port}")
                    self.logger.info(f"API can be accessed from any device on the network using the above URL")
                    
                    return True
                except OSError as e:
                    if "address already in use" in str(e).lower() and retry < max_retries - 1:
                        self.logger.warning(f"Port {port} already in use, trying port {port + 1}")
                        port += 1
                    else:
                        self.logger.error(f"Failed to start API server: {str(e)}")
                        raise
            
            return False
        except Exception as e:
            self.logger.error(f"Failed to start API server: {str(e)}")
            return False
    
    async def shutdown(self):
        """Shutdown the API server."""
        if self.server:
            self.server.should_exit = True
            logger.info("API server shutdown initiated")
        else:
            logger.warning("API server shutdown called but server not running") 