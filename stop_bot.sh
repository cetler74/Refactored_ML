#!/bin/bash

# Stop Trading Bot Script
echo "🛑 Stopping Trading Bot..."

# Check for running bot processes
BOT_PIDS=$(pgrep -f "python run.py")
if [ -z "$BOT_PIDS" ]; then
    echo "ℹ️ No running bot processes found."
else
    echo "🔍 Found running bot processes: $BOT_PIDS"
    
    # Send SIGTERM to allow clean shutdown
    echo "⏳ Sending termination signal..."
    kill $BOT_PIDS
    
    # Wait a moment for graceful shutdown
    sleep 2
    
    # Check if any processes are still running
    REMAINING=$(pgrep -f "python run.py")
    if [ ! -z "$REMAINING" ]; then
        echo "⚠️ Some processes didn't terminate gracefully, forcing shutdown..."
        kill -9 $REMAINING
    fi
    
    echo "✅ Bot processes terminated."
fi

# Check for React server processes
REACT_PIDS=$(pgrep -f "serve_react.py")
if [ -z "$REACT_PIDS" ]; then
    echo "ℹ️ No running React server processes found."
else
    echo "🔍 Found running React server processes: $REACT_PIDS"
    kill $REACT_PIDS
    echo "✅ React server processes terminated."
fi

# Find and kill any remaining uvicorn/Flask processes
WEB_PIDS=$(pgrep -f "uvicorn|flask")
if [ ! -z "$WEB_PIDS" ]; then
    echo "🔍 Found additional web server processes: $WEB_PIDS"
    kill $WEB_PIDS
    echo "✅ Web server processes terminated."
fi

echo "✅ All trading bot processes stopped" 