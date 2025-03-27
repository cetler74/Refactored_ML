#!/bin/bash

# Start Trading Bot Launch Script
echo "🚀 Starting Trading Bot..."

# Function to check if port is in use
is_port_in_use() {
    lsof -i:$1 >/dev/null 2>&1
    return $?
}

# Function to kill process using port
kill_process_on_port() {
    local port=$1
    local pid=$(lsof -ti:$port)
    if [ ! -z "$pid" ]; then
        echo "⚠️ Found process $pid using port $port. Killing..."
        kill -9 $pid >/dev/null 2>&1
        sleep 1
    fi
}

# Define ports
API_PORT=8080
FRONTEND_PORT=3001

# Check for and kill existing bot processes
if pgrep -f "python run.py" > /dev/null; then
    echo "⚠️ Bot already running. Stopping previous instance..."
    pkill -f "python run.py"
    sleep 2
fi

# Check for and kill processes on required ports
if is_port_in_use $API_PORT; then
    echo "⚠️ Port $API_PORT is already in use. Freeing port..."
    kill_process_on_port $API_PORT
fi

if is_port_in_use $FRONTEND_PORT; then
    echo "⚠️ Port $FRONTEND_PORT is already in use. Freeing port..."
    kill_process_on_port $FRONTEND_PORT
fi

# Start the bot in simulation mode with API server only (frontend will be handled separately)
echo "⏳ Starting bot and API server in simulation mode..."
python run.py --component api --mode simulation &
BOT_PID=$!

# Wait for API server to be ready
echo "⏳ Waiting for API server to start..."
MAX_RETRIES=10
RETRY_COUNT=0
while ! curl -s http://localhost:$API_PORT/health >/dev/null && [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    sleep 1
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo "❌ Failed to start API server after $MAX_RETRIES attempts"
        echo "   Check logs for details"
    fi
done

# Start the React server on port 3001 to serve the dashboard
echo "⏳ Starting React server..."
./serve_react.py --port $FRONTEND_PORT --api http://localhost:$API_PORT &
REACT_PID=$!

# Wait a moment and check if services are running
sleep 2
if ! ps -p $BOT_PID > /dev/null; then
    echo "❌ Bot process failed to start or exited. Check logs for details."
    # Kill the React server if bot failed
    if ps -p $REACT_PID > /dev/null; then
        kill $REACT_PID 2>/dev/null
    fi
    exit 1
fi

if ! ps -p $REACT_PID > /dev/null; then
    echo "❌ React server failed to start or exited. Check logs for details."
    exit 1
fi

# Services are running, print success message
echo "✅ Trading Bot started successfully with PIDs: $BOT_PID (main), $REACT_PID (React)"
echo "📊 Dashboard: http://localhost:$FRONTEND_PORT"
echo "🔌 API: http://localhost:$API_PORT"
echo ""
echo "🛑 To stop all services, press Ctrl+C or run: kill $BOT_PID $REACT_PID"

# Setup handler for clean shutdown
trap "kill $BOT_PID $REACT_PID 2>/dev/null; echo '🛑 Stopped all services.'; exit 0" INT TERM

# Keep script running to handle signals
wait 