#!/usr/bin/env python
"""
Simple Flask server to serve the React static files and proxy API requests
"""

from flask import Flask, send_from_directory, request, Response, jsonify
import os
import socket
from pathlib import Path
import requests
from flask_cors import CORS
import logging
import traceback
import argparse
import subprocess
import time
import json
import psutil
import platform
from datetime import datetime, timedelta
import random
import numpy as np

app = Flask(__name__, static_folder='app/frontend/react-app/build')
CORS(app)  # Enable CORS for all routes

# API server URL (default, can be overridden by command line args)
API_URL = "http://localhost:8080"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a settings file path for temporary storage
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_bot_settings.json")

# Default bot settings
DEFAULT_SETTINGS = {
    "max_trading_pairs": 5,
    "trading_strategy": "MACD_RSI",
    "risk_level": "medium",
    "quote_currency": "USDC",
    "trade_amount_per_position": 100,
    "use_stop_loss": True,
    "stop_loss_percentage": 5,
    "take_profit_percentage": 10,
    "trading_timeframe": "1h",
    "simulation_mode": True
}

# Sample data for portfolio performance
def generate_portfolio_data():
    """Generate sample portfolio performance data"""
    now = datetime.now()
    
    # Generate daily data points for the last 30 days
    daily_data = []
    balance = 10000.0  # Starting balance
    
    for i in range(30):
        date = (now - timedelta(days=29-i)).strftime("%Y-%m-%d")
        # Random daily change between -3% and +5%
        daily_change = random.uniform(-0.03, 0.05)
        balance = balance * (1 + daily_change)
        
        daily_data.append({
            "date": date,
            "balance": round(balance, 2),
            "profit_loss": round(daily_change * 100, 2)
        })
    
    # Current portfolio stats
    total_profit = round(balance - 10000, 2)
    profit_percentage = round((balance / 10000 - 1) * 100, 2)
    
    return {
        "current_balance": round(balance, 2),
        "starting_balance": 10000,
        "total_profit_loss": total_profit,
        "profit_loss_percentage": profit_percentage,
        "daily_performance": daily_data,
        "holdings": [
            {"asset": "BTC", "amount": 0.012, "value_usdc": round(balance * 0.4, 2)},
            {"asset": "ETH", "amount": 0.5, "value_usdc": round(balance * 0.3, 2)},
            {"asset": "SOL", "amount": 3.2, "value_usdc": round(balance * 0.15, 2)},
            {"asset": "USDC", "amount": round(balance * 0.15, 2), "value_usdc": round(balance * 0.15, 2)},
        ]
    }

# Sample data for active and closed trades
def generate_trades_data():
    """Generate sample trades data"""
    now = datetime.now()
    
    # Active trades
    active_trades = []
    for i in range(3):
        entry_time = now - timedelta(hours=random.randint(1, 48))
        entry_price = random.uniform(80000, 90000) if i == 0 else random.uniform(1900, 2100) if i == 1 else random.uniform(100, 140)
        current_price = entry_price * (1 + random.uniform(-0.05, 0.08))
        profit_loss = (current_price - entry_price) / entry_price * 100
        
        active_trades.append({
            "id": f"trade_{i+1}",
            "trading_pair": "BTC/USDC" if i == 0 else "ETH/USDC" if i == 1 else "SOL/USDC",
            "trade_type": "buy" if random.random() > 0.3 else "sell",
            "entry_time": entry_time.isoformat(),
            "entry_price": round(entry_price, 2),
            "current_price": round(current_price, 2),
            "quantity": round(random.uniform(0.001, 0.1), 6) if i < 2 else round(random.uniform(0.5, 2), 2),
            "profit_loss": round(profit_loss, 2),
            "profit_loss_usdc": round((current_price - entry_price) * (0.01 if i < 2 else 1), 2),
            "stop_loss": round(entry_price * 0.95, 2),
            "take_profit": round(entry_price * 1.1, 2)
        })
    
    # Closed trades - last 10
    closed_trades = []
    for i in range(10):
        close_time = now - timedelta(days=random.randint(1, 20))
        entry_time = close_time - timedelta(hours=random.randint(4, 72))
        entry_price = random.uniform(75000, 95000) if i % 3 == 0 else random.uniform(1800, 2200) if i % 3 == 1 else random.uniform(90, 150)
        exit_price = entry_price * (1 + random.uniform(-0.1, 0.15))
        profit_loss = (exit_price - entry_price) / entry_price * 100
        
        pair = "BTC/USDC" if i % 3 == 0 else "ETH/USDC" if i % 3 == 1 else "SOL/USDC"
        qty = round(random.uniform(0.001, 0.05), 6) if pair.startswith("BTC") else round(random.uniform(0.01, 0.5), 4) if pair.startswith("ETH") else round(random.uniform(0.5, 5), 2)
        
        closed_trades.append({
            "id": f"closed_{i+1}",
            "trading_pair": pair,
            "trade_type": "buy" if random.random() > 0.3 else "sell",
            "entry_time": entry_time.isoformat(),
            "exit_time": close_time.isoformat(),
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "quantity": qty,
            "profit_loss": round(profit_loss, 2),
            "profit_loss_usdc": round((exit_price - entry_price) * qty, 2),
            "reason": random.choice(["take_profit", "stop_loss", "manual", "signal"])
        })
    
    # Overall stats
    win_count = sum(1 for trade in closed_trades if trade["profit_loss"] > 0)
    loss_count = len(closed_trades) - win_count
    win_rate = round((win_count / len(closed_trades)) * 100, 1) if closed_trades else 0
    
    total_profit = sum(trade["profit_loss_usdc"] for trade in closed_trades)
    avg_profit = round(sum(trade["profit_loss"] for trade in closed_trades) / len(closed_trades), 2) if closed_trades else 0
    
    return {
        "active_trades": active_trades,
        "closed_trades": closed_trades,
        "stats": {
            "total_trades": len(closed_trades),
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": f"{win_rate}%",
            "total_profit_loss": round(total_profit, 2),
            "average_profit_loss": f"{avg_profit}%",
            "largest_win": round(max([trade["profit_loss"] for trade in closed_trades], default=0), 2),
            "largest_loss": round(min([trade["profit_loss"] for trade in closed_trades], default=0), 2),
            "average_hold_time": "19.4 hours"
        }
    }

# Sample data for ML model performance
def generate_ml_metrics():
    """Generate sample ML model metrics"""
    # Accuracy over time (last 30 days)
    accuracy_trend = []
    now = datetime.now()
    
    baseline_accuracy = 60  # baseline accuracy percentage
    for i in range(30):
        date = (now - timedelta(days=29-i)).strftime("%Y-%m-%d")
        # Random daily fluctuation +/- 8%
        daily_accuracy = min(95, max(40, baseline_accuracy + random.uniform(-8, 8)))
        baseline_accuracy = daily_accuracy  # Update baseline for next day
        
        accuracy_trend.append({
            "date": date,
            "accuracy": round(daily_accuracy, 1)
        })
    
    # Recent predictions vs actual
    recent_predictions = []
    for i in range(10):
        prediction_time = now - timedelta(hours=i*8)
        predicted_direction = "up" if random.random() > 0.4 else "down"
        actual_direction = predicted_direction if random.random() > 0.3 else ("down" if predicted_direction == "up" else "up")
        
        recent_predictions.append({
            "timestamp": prediction_time.isoformat(),
            "trading_pair": random.choice(["BTC/USDC", "ETH/USDC", "SOL/USDC"]),
            "timeframe": random.choice(["1h", "4h", "1d"]),
            "predicted_direction": predicted_direction,
            "confidence": round(random.uniform(0.6, 0.95), 2),
            "actual_direction": actual_direction,
            "accurate": predicted_direction == actual_direction
        })
    
    # Metrics by trading pair
    pairs_metrics = []
    for pair in ["BTC/USDC", "ETH/USDC", "SOL/USDC", "XRP/USDC", "BNB/USDC"]:
        pairs_metrics.append({
            "trading_pair": pair,
            "accuracy": round(random.uniform(55, 85), 1),
            "profit_factor": round(random.uniform(1.1, 2.5), 2),
            "sharpe_ratio": round(random.uniform(0.8, 2.8), 2),
            "prediction_count": random.randint(50, 200)
        })
    
    return {
        "overall_accuracy": round(sum(p["accuracy"] for p in accuracy_trend[-7:]) / 7, 1),  # Last 7 days
        "accuracy_trend": accuracy_trend,
        "recent_predictions": recent_predictions,
        "metrics_by_pair": pairs_metrics,
        "model_name": "Ensemble (LSTM + XGBoost)",
        "last_training": (now - timedelta(days=random.randint(1, 7))).isoformat(),
        "features_count": random.randint(15, 30)
    }

# Sample data for risk metrics
def generate_risk_metrics():
    """Generate sample risk metrics"""
    # Portfolio risk metrics
    portfolio_risk = {
        "var_95": round(random.uniform(3, 8), 2),  # 95% VaR (percentage)
        "var_99": round(random.uniform(7, 12), 2),  # 99% VaR (percentage)
        "sharpe_ratio": round(random.uniform(0.8, 2.5), 2),
        "sortino_ratio": round(random.uniform(1.0, 3.0), 2),
        "max_drawdown": round(random.uniform(10, 25), 2),  # Maximum historical drawdown percentage
        "volatility": round(random.uniform(15, 35), 2),  # Annualized volatility percentage
        "risk_adjusted_return": round(random.uniform(8, 18), 2),  # Annualized risk-adjusted return
    }
    
    # Risk allocation by asset
    risk_allocation = []
    assets = ["BTC", "ETH", "SOL", "XRP", "BNB", "USDC"]
    weights = [0.3, 0.25, 0.15, 0.1, 0.1, 0.1]  # Sum to 1.0
    
    for asset, weight in zip(assets, weights):
        risk_allocation.append({
            "asset": asset,
            "allocation": round(weight * 100, 1),  # Convert to percentage
            "risk_contribution": round(weight * random.uniform(0.8, 1.2) * 100, 1),  # Risk contribution isn't always equal to allocation
            "var_contribution": round(weight * random.uniform(0.9, 1.1) * 100, 1)
        })
    
    # Risk metrics over time
    risk_trend = []
    now = datetime.now()
    var_baseline = 5.0  # Starting VaR value
    
    for i in range(30):
        date = (now - timedelta(days=29-i)).strftime("%Y-%m-%d")
        # Random daily fluctuation in VaR
        daily_var = max(1.0, var_baseline + random.uniform(-0.8, 0.8))
        var_baseline = daily_var  # Update baseline for next day
        
        risk_trend.append({
            "date": date,
            "var_95": round(daily_var, 2),
            "volatility": round(daily_var * random.uniform(2.5, 3.5), 2)
        })
    
    return {
        "portfolio_risk": portfolio_risk,
        "risk_allocation": risk_allocation,
        "risk_trend": risk_trend,
        "current_exposure": {
            "long_exposure": round(random.uniform(30, 70), 1),  # Percentage of portfolio in long positions
            "short_exposure": round(random.uniform(0, 20), 1),  # Percentage in short
            "neutral": round(random.uniform(20, 40), 1)  # Remaining in stable assets/cash
        }
    }

# Add portfolio endpoint
@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    try:
        # First try to get data from the real API
        try:
            resp = requests.get(f"{API_URL}/portfolio", timeout=1)
            if resp.status_code == 200:
                return Response(resp.content, resp.status_code, headers=dict(resp.headers))
        except:
            # If real API fails, provide fallback data
            pass
            
        return jsonify(generate_portfolio_data())
    except Exception as e:
        logger.error(f"Error getting portfolio data: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Failed to load portfolio data: {str(e)}"}), 500

# Add trades endpoint
@app.route('/api/trades', methods=['GET'])
def get_trades():
    try:
        # First try to get data from the real API
        try:
            resp = requests.get(f"{API_URL}/trades", timeout=1)
            if resp.status_code == 200:
                return Response(resp.content, resp.status_code, headers=dict(resp.headers))
        except:
            # If real API fails, provide fallback data
            pass
            
        return jsonify(generate_trades_data())
    except Exception as e:
        logger.error(f"Error getting trades data: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Failed to load trades data: {str(e)}"}), 500

# Add ML metrics endpoint
@app.route('/api/ml_metrics', methods=['GET'])
def get_ml_metrics():
    try:
        # First try to get data from the real API
        try:
            resp = requests.get(f"{API_URL}/ml_metrics", timeout=1)
            if resp.status_code == 200:
                return Response(resp.content, resp.status_code, headers=dict(resp.headers))
        except:
            # If real API fails, provide fallback data
            pass
            
        return jsonify(generate_ml_metrics())
    except Exception as e:
        logger.error(f"Error getting ML metrics data: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Failed to load ML metrics data: {str(e)}"}), 500

# Add risk metrics endpoint
@app.route('/api/risk_metrics', methods=['GET'])
def get_risk_metrics():
    try:
        # First try to get data from the real API
        try:
            resp = requests.get(f"{API_URL}/risk_metrics", timeout=1)
            if resp.status_code == 200:
                return Response(resp.content, resp.status_code, headers=dict(resp.headers))
        except:
            # If real API fails, provide fallback data
            pass
            
        return jsonify(generate_risk_metrics())
    except Exception as e:
        logger.error(f"Error getting risk metrics data: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Failed to load risk metrics data: {str(e)}"}), 500

# Add handler for bot settings
@app.route('/api/settings', methods=['GET'])
def get_settings():
    try:
        # First try to get settings from the real API
        try:
            resp = requests.get(f"{API_URL}/settings", timeout=1)
            if resp.status_code == 200:
                return Response(resp.content, resp.status_code, headers=dict(resp.headers))
        except:
            # If real API fails, provide fallback data
            pass
            
        # Check if we have saved settings
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
        else:
            # Use default settings if no file exists
            settings = DEFAULT_SETTINGS
            
        return jsonify(settings)
    except Exception as e:
        logger.error(f"Error getting bot settings: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Failed to load settings: {str(e)}"}), 500

# Add handler for saving bot settings
@app.route('/api/settings', methods=['POST', 'PUT'])
def save_settings():
    try:
        # First try to forward to the real API
        try:
            method = request.method
            resp = requests.request(
                method=method,
                url=f"{API_URL}/settings",
                headers={key: value for key, value in request.headers if key != 'Host'},
                data=request.get_data(),
                cookies=request.cookies,
                allow_redirects=False,
                timeout=5
            )
            if resp.status_code < 400:  # Any successful response
                return Response(resp.content, resp.status_code, headers=dict(resp.headers))
        except:
            # If real API fails, handle locally
            pass
            
        # Get the settings from the request
        new_settings = request.get_json()
        
        # Validate settings (basic validation)
        if not isinstance(new_settings, dict):
            return jsonify({"error": "Invalid settings format"}), 400
            
        # Merge with defaults to ensure all required fields
        settings = DEFAULT_SETTINGS.copy()
        settings.update(new_settings)
        
        # Save to temporary file
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
            
        logger.info(f"Saved bot settings: {settings}")
        return jsonify({"message": "Settings saved successfully", "settings": settings})
    except Exception as e:
        logger.error(f"Error saving bot settings: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Failed to save settings: {str(e)}"}), 500

# Add a direct handler for system_info to prevent 404
@app.route('/api/system_info', methods=['GET'])
def system_info():
    try:
        # Check if the real API is available
        try:
            resp = requests.get(f"{API_URL}/system_info", timeout=1)
            if resp.status_code == 200:
                return Response(resp.content, resp.status_code, headers=dict(resp.headers))
        except:
            # If real API fails, provide fallback data
            pass
            
        # Fallback: Generate basic system info
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        info = {
            "status": "running",
            "version": "1.0.0",
            "uptime": "Unknown",  # Would need bot's actual uptime
            "system": {
                "platform": platform.system(),
                "version": platform.version(),
                "cpu_usage": f"{cpu_percent}%",
                "memory_usage": f"{memory.percent}%",
                "memory_available": f"{memory.available / (1024 * 1024):.1f} MB",
                "disk_usage": f"{disk.percent}%",
                "disk_available": f"{disk.free / (1024 * 1024 * 1024):.1f} GB"
            },
            "trading": {
                "active_pairs": ["BTC/USDC", "ETH/USDC", "SOL/USDC", "XRP/USDC", "BNB/USDC"],
                "mode": "simulation",
                "last_signal": datetime.now().isoformat()
            }
        }
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error generating system info: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500

# Proxy API requests to the API server
@app.route('/api/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy_api(path):
    # Skip proxying endpoints we handle directly
    if path == 'system_info':
        return system_info()
    elif path == 'settings' and request.method == 'GET':
        return get_settings()
    elif path == 'settings' and request.method in ['POST', 'PUT']:
        return save_settings()
    elif path == 'portfolio':
        return get_portfolio()
    elif path == 'trades':
        return get_trades()
    elif path == 'ml_metrics':
        return get_ml_metrics()
    elif path == 'risk_metrics':
        return get_risk_metrics()
        
    target_url = f"{API_URL}/{path}"
    logger.info(f"Proxying request to: {target_url}")
    
    try:
        # Forward the request to the API server
        headers = {key: value for key, value in request.headers if key != 'Host'}
        logger.info(f"Request method: {request.method}")
        
        resp = requests.request(
            method=request.method,
            url=target_url,
            headers=headers,
            data=request.get_data(),
            cookies=request.cookies,
            allow_redirects=False,
            timeout=10  # Add timeout to prevent hanging requests
        )
        
        logger.info(f"Response status: {resp.status_code}")
        
        # Create a Flask response object
        response = Response(resp.content, resp.status_code)
        
        # Add response headers
        for key, value in resp.headers.items():
            if key.lower() != 'content-length':
                response.headers[key] = value
        
        return response
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error to API at {target_url}")
        return Response("API server unavailable", 503)
    except Exception as e:
        logger.error(f"Error proxying request to {target_url}: {str(e)}")
        logger.error(traceback.format_exc())
        return Response(f"Proxy error: {str(e)}", 500)

# Special endpoint to restart the trading bot
@app.route('/restart', methods=['POST'])
def restart_bot():
    try:
        logger.info("Received request to restart the bot")
        
        # Check if the restart_script parameter is provided
        restart_script = request.args.get('script', './start_bot.sh')
        
        # Kill existing bot processes
        try:
            subprocess.run(['pkill', '-f', 'python run.py'], check=False)
            logger.info("Killed existing bot processes")
            time.sleep(2)  # Give processes time to terminate
        except Exception as e:
            logger.error(f"Error killing existing processes: {str(e)}")
        
        # Execute the restart script
        if os.path.exists(restart_script) and os.access(restart_script, os.X_OK):
            subprocess.Popen([restart_script], start_new_session=True)
            return jsonify({"status": "success", "message": "Bot restart initiated"})
        else:
            logger.error(f"Restart script {restart_script} not found or not executable")
            return jsonify({"status": "error", "message": f"Restart script {restart_script} not found or not executable"}), 400
            
    except Exception as e:
        logger.error(f"Error restarting bot: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

def find_available_port(start_port, max_retries=5):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_retries):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('0.0.0.0', port))
            sock.close()
            return port
        except OSError:
            continue
    return start_port + max_retries  # Return the last port even if it might be in use

def get_local_ip():
    """Get the local IP address of the machine"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Serve React app with API proxy')
    parser.add_argument('--port', type=int, default=3001, help='Port to serve React app')
    parser.add_argument('--api', type=str, default='http://localhost:8080', help='API server URL')
    args = parser.parse_args()
    
    # Set API URL from command line
    API_URL = args.api
    
    # Check if build directory exists
    build_path = Path('app/frontend/react-app/build')
    if not build_path.exists() or not (build_path / 'index.html').exists():
        print(f"ERROR: React build directory not found at {build_path} or missing index.html")
        exit(1)
    
    # Find an available port
    port = find_available_port(args.port)
    local_ip = get_local_ip()
    
    print(f"Serving React app from {build_path}")
    print(f"API proxy configured to forward requests to {API_URL}")
    print(f"React app available at:")
    print(f"  - http://localhost:{port}")
    print(f"  - http://{local_ip}:{port}")
    print(f"Restart endpoint available at: http://localhost:{port}/restart")
    
    app.run(debug=True, use_reloader=True, port=port, host='0.0.0.0') 