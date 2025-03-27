#!/usr/bin/env python3
"""
Trading Bot Launcher Script

This script provides a simple CLI to launch different components of the trading bot.
"""

import argparse
import asyncio
import logging
from logging.handlers import RotatingFileHandler
import sys
import socket
import os
import redis
import sqlalchemy
from sqlalchemy import create_engine, text
from pathlib import Path
import subprocess
import signal
import time
import traceback
from datetime import datetime

from app.config.settings import get_settings, AppMode, Settings
from app.exchange.manager import ExchangeManager
from app.trading.bot import TradingBot
from app.api.server import APIServer
from app.core.orchestrator import BotOrchestrator

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging with file output
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# File handler - 5MB max file size, keep 5 backup files
file_handler = RotatingFileHandler(
    'logs/trading_bot.log',
    maxBytes=5*1024*1024,
    backupCount=5
)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Trading Bot CLI')
    parser.add_argument(
        '--component',
        choices=['all', 'bot', 'api', 'frontend'],
        default='all',
        help='Component to run'
    )
    parser.add_argument(
        '--mode',
        choices=['live', 'simulation'],
        default='simulation',
        help='Trading mode'
    )
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--frontend-only', action='store_true', help='Run only the frontend server')
    parser.add_argument('--backend-only', action='store_true', help='Run only the backend trading bot')
    return parser.parse_args()

def get_local_ip():
    """Get the local IP address of the host machine."""
    try:
        # Create a socket connection to an external server
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        logger.warning(f"Could not determine local IP address: {e}")
        return "localhost"

async def check_database_connection(settings):
    """Check if database is accessible."""
    try:
        # Create a database URL from settings
        db_url = f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        engine = create_engine(db_url)
        
        # Try to connect and run a simple query
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            connection.commit()
            row = result.fetchone()
            if row and row[0] == 1:
                return True, None
    except Exception as e:
        logger.warning(f"Database connection failed: {str(e)}")
        return False, str(e)
    
    return False, "Unknown error"

async def check_redis_connection(settings):
    """Check if Redis is accessible."""
    try:
        # Create Redis client
        r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD or None,  # Use None for empty password
            socket_timeout=2  # Short timeout for check
        )
        
        # Try to ping Redis
        if r.ping():
            return True, None
        else:
            return False, "Redis ping failed"
    except Exception as e:
        logger.warning(f"Redis connection failed: {str(e)}")
        return False, str(e)

async def start_bot(settings, exchange_manager, orchestrator):
    """Start the trading bot component."""
    # Initialize modules first
    await orchestrator.initialize_modules()
    
    # Call the async start method with mode parameter
    result = await orchestrator.start(mode=settings.APP_MODE.lower())
    logger.info("Bot initialized and started via orchestrator")
    return orchestrator

async def start_api(settings, orchestrator):
    """Start the API server component."""
    try:
        # Import here to avoid circular imports
        from app.api.server import APIServer
        
        # Use the shared orchestrator
        api_server = APIServer(settings, orchestrator)
        await api_server.start()
        
        return api_server
    except Exception as e:
        logger.error(f"Error starting API server: {e}")
        raise

async def start_frontend():
    """Start the frontend server as a subprocess"""
    frontend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app/frontend")
    script_path = os.path.join(frontend_path, "start_frontend.sh")
    
    if not os.path.exists(script_path):
        logger.error(f"Frontend script not found: {script_path}")
        return None
    
    try:
        # Make sure the script is executable
        os.chmod(script_path, 0o755)
        
        # Start the frontend server
        logger.info("Starting frontend server...")
        process = subprocess.Popen(
            [script_path],
            cwd=frontend_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Log the output in a separate task
        async def log_output():
            for line in iter(process.stdout.readline, ""):
                logger.info(f"Frontend: {line.strip()}")
        
        asyncio.create_task(log_output())
        
        return process
    except Exception as e:
        logger.error(f"Error starting frontend: {e}")
        return None

async def print_system_info(components, mode, settings):
    """Print basic system information at startup."""
    local_ip = get_local_ip()
    api_port = settings.API_PORT
    frontend_port = settings.FRONTEND_PORT
    
    print("\n" + "=" * 80)
    print(f"🚀 STARTING TRADING BOT IN {mode.upper()} MODE 🚀")
    print("=" * 80)
    
    print(f"\n▶️ Starting components: {', '.join(components if components != ['all'] else ['bot', 'api', 'frontend'])}")
    print(f"▶️ Network IP: {local_ip}")
    
    if "api" in components or "all" in components:
        print(f"▶️ API will be available at: http://{local_ip}:{api_port} and http://localhost:{api_port}")
    
    if "frontend" in components or "all" in components:
        print(f"▶️ Dashboard will be available at: http://{local_ip}:{frontend_port} and http://localhost:{frontend_port}")
    
    print("\n" + "=" * 80)
    
    # Log this information as well
    logger.info(f"Starting trading bot in {mode} mode with components: {components}")
    logger.info(f"Local IP: {local_ip}")
    if "api" in components or "all" in components:
        logger.info(f"API will be available at: http://{local_ip}:{api_port}")
    if "frontend" in components or "all" in components:
        logger.info(f"Dashboard will be available at: http://{local_ip}:{frontend_port}")

def check_node_installation():
    """Check if Node.js and npm are installed."""
    try:
        subprocess.run(['node', '--version'], capture_output=True, check=True)
        subprocess.run(['npm', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def build_react_app():
    """Build the React frontend application."""
    react_app_dir = Path('app/frontend/react-app')
    if not react_app_dir.exists():
        logger.error("React app directory not found")
        return False

    try:
        # Install dependencies
        logger.info("Installing React dependencies...")
        subprocess.run(['npm', 'install'], cwd=react_app_dir, check=True)

        # Build the app
        logger.info("Building React application...")
        subprocess.run(['npm', 'run', 'build'], cwd=react_app_dir, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error building React app: {str(e)}")
        return False

def start_trading_bot(mode="simulation"):
    """Start the trading bot application."""
    try:
        # Load settings with the correct mode
        settings = get_settings()
        settings.APP_MODE = mode
        
        # Check Node.js installation
        if not check_node_installation():
            logger.error("Node.js and npm are required but not installed. Please install them first.")
            sys.exit(1)

        # Build React app
        if not build_react_app():
            logger.error("Failed to build React application")
            sys.exit(1)

        # Start FastAPI server asynchronously
        import asyncio
        
        async def start_server():
            try:
                from app.frontend.server import FrontendServer
                
                # Create and start the frontend server
                frontend_server = FrontendServer(settings)
                await frontend_server.start()
                
                # Keep the server running
                while True:
                    await asyncio.sleep(1)
            except Exception as e:
                logger.exception(f"Error in frontend server: {e}")
                
        # Run the server in the event loop
        logger.info(f"Starting frontend server on port {settings.FRONTEND_PORT}...")
        loop = asyncio.get_event_loop()
        
        try:
            loop.run_until_complete(start_server())
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            loop.close()
            logger.info("Server shutdown")

    except Exception as e:
        logger.exception(f"Error starting trading bot: {str(e)}")
        sys.exit(1)

async def main():
    try:
        args = parse_args()
        settings = get_settings()
        components = [args.component]
        
        if args.component == 'all':
            components = ['bot', 'api', 'frontend']
        
        # Show initial startup information
        await print_system_info(components, args.mode, settings)
            
        logger.info(f"Starting trading bot with components: {components} in mode: {args.mode}")
        
        # Initialize exchange manager with the specified mode
        exchange_manager = ExchangeManager(settings, args.mode)
        logger.info(f"Exchange manager initialized in {args.mode} mode")
        
        # Create a shared orchestrator that will be used by both bot and API
        orchestrator = BotOrchestrator(settings)
        orchestrator.exchange_manager = exchange_manager
        logger.info("Created shared orchestrator instance")
        
        # Initialize components based on arguments
        started_components = []
        
        if args.component in ['all', 'bot']:
            logger.info(f"Starting trading bot in {args.mode} mode")
            bot = await start_bot(settings, exchange_manager, orchestrator)
            started_components.append("Trading Bot")
            print(f"✅ Trading Bot started successfully in {args.mode} mode")
            
        if args.component in ['all', 'api']:
            logger.info(f"Starting API server")
            api_server = await start_api(settings, orchestrator)
            started_components.append("API Server")
            print(f"✅ API Server started successfully on port {settings.API_PORT}")
            
        if args.component in ['all', 'frontend']:
            logger.info(f"Starting frontend server")
            frontend_process = await start_frontend()
            started_components.append("Frontend Dashboard")
            print(f"✅ Frontend Dashboard started successfully on port {settings.FRONTEND_PORT}")
        
        # Try to check services status but don't let it block startup
        try:
            db_status, db_error = await check_database_connection(settings)
            redis_status, redis_error = await check_redis_connection(settings)
            
            # Display service status clearly
            print("\n" + "=" * 80)
            print("📊 SERVICES STATUS")
            print("=" * 80)
            print(f"• Database: {'✅ Connected' if db_status else '❌ Not Connected - ' + db_error}")
            print(f"• Redis:    {'✅ Connected' if redis_status else '❌ Not Connected - ' + redis_error}")
            print(f"• Logging:  ✅ Writing to logs/trading_bot.log")
            print("=" * 80)
            
            # Log the service status
            logger.info(f"Database connection: {'Success' if db_status else 'Failed - ' + db_error}")
            logger.info(f"Redis connection: {'Success' if redis_status else 'Failed - ' + redis_error}")
            logger.info("Logging is configured and working")
        except Exception as e:
            logger.error(f"Error checking services: {e}")
            print(f"❌ Error checking services: {e}")
            
        # Display final access information
        print("\n" + "=" * 80)
        print("🔗 ACCESS INFORMATION")
        print("=" * 80)
        local_ip = get_local_ip()
        
        if "api" in components or "all" in components:
            api_url = f"http://{local_ip}:{settings.API_PORT}"
            print(f"📡 API Server:")
            print(f"   • Local:   http://localhost:{settings.API_PORT}")
            print(f"   • Network: {api_url}")
            print(f"   • Health:  {api_url}/health")
        
        if "frontend" in components or "all" in components:
            frontend_url = f"http://{local_ip}:{settings.FRONTEND_PORT}"
            print(f"\n🖥️ Dashboard:")
            print(f"   • Local:   http://localhost:{settings.FRONTEND_PORT}")
            print(f"   • Network: {frontend_url}")
        
        print("\n" + "=" * 80)
        print("✨ TRADING BOT IS RUNNING ✨")
        print("=" * 80)
        print("• Press Ctrl+C to stop the bot")
        print("=" * 80 + "\n")
        
        # Log the final success message
        logger.info(f"Trading bot is running with components: {', '.join(started_components)}")
        
        # Keep the main task running
        while True:
            # Check if frontend process is still running
            if frontend_process and frontend_process.poll() is not None:
                logger.error("Frontend process exited unexpectedly")
                frontend_process = None
                
                # Restart frontend if it crashed
                if not args.backend_only:
                    logger.info("Attempting to restart frontend...")
                    frontend_process = await start_frontend()
            
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        print("\n👋 Trading bot is shutting down. Thank you for using our system! 👋\n")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error starting components: {e}")
        raise

if __name__ == '__main__':
    asyncio.run(main()) 