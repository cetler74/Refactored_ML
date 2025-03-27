#!/bin/bash
set -e  # Exit on error

echo "==== Starting Trading Bot Frontend ===="

# Change to the script directory
cd "$(dirname "$0")"
FRONTEND_DIR="$(pwd)"
echo "Frontend directory: $FRONTEND_DIR"

# Ensure node and npm are installed
if ! command -v node &> /dev/null || ! command -v npm &> /dev/null; then
    echo "Error: Node.js and npm are required but not installed"
    echo "Please install Node.js and npm, then try again"
    exit 1
fi

NODE_VERSION=$(node -v)
NPM_VERSION=$(npm -v)
echo "Using Node.js $NODE_VERSION and npm $NPM_VERSION"

# Check if react-app directory exists
if [ ! -d "react-app" ]; then
    echo "Error: react-app directory not found"
    exit 1
fi

cd react-app

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing React dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies"
        exit 1
    fi
    echo "Dependencies installed successfully"
else
    echo "Dependencies already installed"
fi

# Build the React app
echo "Building React app..."
npm run build
if [ $? -ne 0 ]; then
    echo "Error: Failed to build React app"
    exit 1
fi
echo "React app built successfully"

# Return to frontend directory
cd "$FRONTEND_DIR"

# Check if build directory exists
if [ ! -d "react-app/build" ]; then
    echo "Error: Build directory not found after build process"
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed"
    echo "Please install Python 3, then try again"
    exit 1
fi

# Start the server
echo "Starting frontend server..."
python3 server.py 