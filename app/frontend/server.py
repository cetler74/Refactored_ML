#!/usr/bin/env python3
import logging
import os
import sys
from pathlib import Path
import traceback
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import uvicorn
import contextlib
import asyncio
import socket
import signal

# Setup logging first, before any imports that might use it
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define a global variable for the current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define a function to setup imports - this ensures proper error handling
def setup_imports():
    """Configure import paths based on the execution context"""
    # Get current directory and possible root paths
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    parent_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(parent_dir)
    
    # Try multiple approaches to find the correct import path
    possible_roots = [
        project_root,  # Standard project layout
        parent_dir,    # If app is the project root
        os.path.dirname(os.path.dirname(project_root))  # If running from a subdirectory
    ]
    
    logger.info(f"Current file: {current_file}")
    logger.info(f"Current directory: {current_dir}")
    
    # Add all possible roots to sys.path
    for root in possible_roots:
        if root not in sys.path:
            sys.path.insert(0, root)
            logger.info(f"Added {root} to sys.path")
    
    # Try importing using absolute imports
    try:
        logger.info("Attempting to import using absolute imports...")
        from app.core.orchestrator import BotOrchestrator
        from app.config.settings import Settings
        from app.frontend.ml_metrics_helper import get_enhanced_ml_metrics
        logger.info("Successfully imported using absolute imports")
        return BotOrchestrator, Settings, get_enhanced_ml_metrics
    except ImportError as e:
        logger.warning(f"Absolute imports failed: {e}")
        
        # Try importing using relative imports
        try:
            logger.info("Attempting relative imports...")
            # Add the parent directory to sys.path for relative imports
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
                
            # Now try the imports
            from core.orchestrator import BotOrchestrator
            from config.settings import Settings
            from frontend.ml_metrics_helper import get_enhanced_ml_metrics
            logger.info("Successfully imported using relative imports")
            return BotOrchestrator, Settings, get_enhanced_ml_metrics
        except ImportError as e2:
            logger.warning(f"Relative imports failed: {e2}")
            
            # Last resort - try direct imports from current directory
            try:
                logger.info("Attempting direct imports...")
                sys.path.insert(0, current_dir)
                import ml_metrics_helper
                get_enhanced_ml_metrics = ml_metrics_helper.get_enhanced_ml_metrics
                
                sys.path.insert(0, os.path.join(parent_dir, "core"))
                from orchestrator import BotOrchestrator
                
                sys.path.insert(0, os.path.join(parent_dir, "config"))
                from settings import Settings
                
                logger.info("Successfully imported using direct imports")
                return BotOrchestrator, Settings, get_enhanced_ml_metrics
            except ImportError as e3:
                logger.error(f"All import attempts failed: {e3}")
                logger.error(f"Current sys.path: {sys.path}")
                raise ImportError(f"Could not import required modules: {e}, {e2}, {e3}")

# Import the required modules
try:
    BotOrchestrator, Settings, get_enhanced_ml_metrics = setup_imports()
except ImportError as e:
    logger.critical(f"FATAL: Could not import required modules: {e}")
    sys.exit(1)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Store WebSocket connections
connected_clients: List[WebSocket] = []

# Create FastAPI app with lifespan
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app initialization and cleanup"""
    # Startup logic
    try:
        # Initialize bot
        settings = Settings()
        app.bot = BotOrchestrator(settings)
        logger.info("Bot initialized")
        
        # Start the bot properly
        await app.bot.start(mode="simulation")
        logger.info("Bot started in simulation mode")
        
        # Start periodic data broadcast task
        data_broadcast_task = asyncio.create_task(periodic_data_broadcast())
        yield
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        app.bot = None
        yield
    finally:
        # Shutdown logic
        if hasattr(app, 'bot') and app.bot:
            try:
                await app.bot.stop()
            except Exception as e:
                logger.error(f"Error stopping bot: {e}")
        
        # Cancel background tasks
        if 'data_broadcast_task' in locals() and not data_broadcast_task.done():
            data_broadcast_task.cancel()
            
        # Close all WebSocket connections
        for client in connected_clients:
            try:
                await client.close()
            except:
                pass
        connected_clients.clear()

# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Add CORS middleware - in production, restrict origins
# In development mode, allow all origins
is_dev_mode = os.environ.get("ENV", "development") == "development"
origins = ["*"] if is_dev_mode else [
    "https://yourdomain.com",  # Replace with actual frontend domain
    "https://api.yourdomain.com",  # Replace with API domain if different
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize app state
app.bot = None

async def broadcast_data(data, message_type="update"):
    """Broadcast data to all connected WebSocket clients"""
    if not connected_clients:
        return
    
    message = {
        "type": message_type,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }
    
    for client in connected_clients:
        try:
            await client.send_json(message)
        except Exception as e:
            logger.error(f"Error broadcasting to client: {e}")
            if client in connected_clients:
                connected_clients.remove(client)

async def get_portfolio_data() -> Dict[str, Any]:
    """Get portfolio data from bot, with proper error handling"""
    try:
        if not app.bot:
            logger.warning("Bot not initialized, returning empty portfolio data")
            return {
                "status": "unavailable",
                "message": "Trading bot is not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        # Let the bot handle simulation/real data internally
        portfolio = await app.bot.get_portfolio()
        
        # Debug log to see the structure
        logger.info(f"Portfolio data structure: {json.dumps(portfolio, default=str)}")
        
        # Convert field names to match frontend expectations if needed
        if "total_balance_usdc" in portfolio and "total_balance" not in portfolio:
            portfolio["total_balance"] = portfolio["total_balance_usdc"]
        
        if "available_usdc" in portfolio and "available_balance" not in portfolio:
            portfolio["available_balance"] = portfolio["available_usdc"]
            
        if "open_positions" in portfolio and "positions" not in portfolio:
            portfolio["positions"] = portfolio["open_positions"]
        
        # Additional debug: Log final keys
        logger.info(f"Portfolio data keys after transformation: {sorted(portfolio.keys())}")
        logger.info(f"Final portfolio data structure for frontend: {json.dumps({k: type(v).__name__ for k, v in portfolio.items()}, default=str)}")
            
        return portfolio
        
    except Exception as e:
        logger.error(f"Error getting portfolio data: {str(e)}")
        traceback.print_exc()
        # Return structured error information
        return {
            "status": "error",
            "message": f"Failed to retrieve portfolio data: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "error_type": type(e).__name__
        }

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/ml/metrics")
async def get_ml_metrics_endpoint():
    """Get ML metrics endpoint"""
    try:
        return get_enhanced_ml_metrics()
    except Exception as e:
        logger.error(f"Error getting ML metrics: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Failed to retrieve ML metrics: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "error_type": type(e).__name__
            }
        )

# Alias the mock function to make intent clear
from app.frontend.ml_metrics_helper import get_enhanced_ml_metrics as get_mock_ml_metrics

# --- New function to get REAL ML metrics ---
async def get_current_ml_metrics() -> Dict[str, Any]:
    """Attempts to retrieve real-time metrics from the ML module."""
    now_iso = datetime.now().isoformat()
    
    # Default response if ML is unavailable
    ml_unavailable_response = {
        "status": "unavailable",
        "message": "ML module is disabled or not initialized.",
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1_score": 0,
        "timestamp": now_iso,
        "models": [],
        "training_status": {"in_progress": False, "current_operation": "Idle"},
        "model_stats": {"trained": 0, "total": 0},
        "prediction_stats": {"total": 0, "success_rate": 0},
        "model_health": {"status": "unknown", "message": "ML module inactive."},
        "last_training_cycle": None,
        "next_training_cycle": None,
        "training_frequency": "N/A",
        "predictions": [],
        "feature_importance": {}
    }

    if not app.bot or not hasattr(app.bot, 'ml_module') or not app.bot.ml_module:
        logger.warning("ML module not available in BotOrchestrator.")
        return ml_unavailable_response

    # Check if the ML module has the required get_metrics method
    if not hasattr(app.bot.ml_module, 'get_metrics'):
        logger.error("ML module exists but lacks the 'get_metrics' method.")
        # Return unavailable status, as we can't get real data
        response = ml_unavailable_response.copy()
        response["status"] = "error"
        response["message"] = "ML module is missing the 'get_metrics' method."
        response["model_health"]["status"] = "error"
        response["model_health"]["message"] = "Interface mismatch: 'get_metrics' not found."
        return response

    try:
        # Call the actual method on the ML module instance
        metrics = await app.bot.ml_module.get_metrics()
        if not metrics:
             logger.warning("ML module get_metrics() returned empty data.")
             # Fallback to unavailable status if method returns nothing useful
             response = ml_unavailable_response.copy()
             response["status"] = "nodata"
             response["message"] = "ML module reported no metrics data available."
             return response
             
        # Ensure basic structure compatibility (add more checks as needed)
        metrics.setdefault("status", "available")
        metrics.setdefault("timestamp", now_iso)
        logger.info("Successfully retrieved real ML metrics.")
        return metrics
    except Exception as e:
        logger.error(f"Error retrieving metrics from ML module: {e}", exc_info=True)
        # Return error status
        response = ml_unavailable_response.copy()
        response["status"] = "error"
        response["message"] = f"Failed to retrieve metrics from ML module: {str(e)}"
        response["model_health"]["status"] = "error"
        response["model_health"]["message"] = "Error during metrics retrieval."
        return response

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    connected_clients.append(websocket)
    
    logger.info(f"WebSocket client connected. Total clients: {len(connected_clients)}")
    
    try:
        # Send initial data
        try:
            # --- Use the new function for ML metrics ---
            ml_metrics = await get_current_ml_metrics() 
            await websocket.send_json({
                "type": "ml_metrics",
                "data": ml_metrics,
                "timestamp": datetime.now().isoformat()
            })
            
            # Send portfolio data
            portfolio = await get_portfolio_data()
            logger.info(f"DEBUG - Portfolio data being sent to client: {json.dumps(portfolio, default=str)}")
            await websocket.send_json({
                "type": "portfolio",
                "data": portfolio,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error sending initial data: {e}")
            # Send error message to client
            await websocket.send_json({
                "type": "error",
                "message": f"Failed to load initial data: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
        
        # Keep connection alive and process messages
        while True:
            # Wait for messages (we don't actually process them in this example)
            data = await websocket.receive_text()
            # If we get here, client is still connected
            
            # In a real implementation, process commands from client
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
        if websocket in connected_clients:
            connected_clients.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in connected_clients:
            connected_clients.remove(websocket)

# Serve static files
@app.get("/")
async def get_root():
    """Serve the React app"""
    index_path = os.path.join(CURRENT_DIR, 'react-app/build/index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        logger.warning(f"Index file not found at {index_path}")
        return {"status": "error", "message": "Frontend files not built"}

# Serve static files from the React build directory
# Check if directory exists before mounting
static_dir = os.path.join(CURRENT_DIR, "react-app/build/static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"Mounted static files directory: {static_dir}")
else:
    logger.warning(f"Static files directory not found: {static_dir}")

# Serve other static files from React build
@app.get("/{path:path}")
async def serve_react_assets(path: str):
    file_path = os.path.join(CURRENT_DIR, f"react-app/build/{path}")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    index_path = os.path.join(CURRENT_DIR, 'react-app/build/index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(
        status_code=404,
        content={"status": "error", "message": "File not found"}
    )

# Background tasks
async def periodic_data_broadcast():
    """Periodically broadcast data to all connected clients"""
    BROADCAST_INTERVAL_SECONDS = 15
    
    while True:
        try:
            if connected_clients:
                # --- Use the new function for ML metrics ---
                ml_metrics = await get_current_ml_metrics() 
                portfolio = await get_portfolio_data()
                
                # Broadcast to all clients
                await broadcast_data(ml_metrics, "ml_metrics")
                await broadcast_data(portfolio, "portfolio")
                
                logger.debug(f"Broadcasted data to {len(connected_clients)} clients")
        except Exception as e:
            logger.error(f"Error in periodic broadcast: {e}")
        
        await asyncio.sleep(BROADCAST_INTERVAL_SECONDS)

# Run the server
if __name__ == "__main__":
    # Get port from settings or use default
    settings = Settings()
    port = getattr(settings, "FRONTEND_PORT", 3002)
    
    print(f"Starting FastAPI server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port) 