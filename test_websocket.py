#!/usr/bin/env python3
"""
Test script for WebSocket connection to the trading bot dashboard
"""

import asyncio
import websockets
import json
import argparse
import sys

async def test_websocket(port=3001):
    """Test connection to the WebSocket server"""
    uri = f"ws://localhost:{port}/ws"
    print(f"Attempting to connect to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print(f"✅ Successfully connected to WebSocket at {uri}")
            print("Waiting for data (5 seconds)...")
            
            # Set a timeout to wait for messages
            try:
                for _ in range(3):  # Try to get 3 messages
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        data = json.loads(message)
                        print("\nReceived data (full structure):")
                        print(json.dumps(data, indent=2))
                        
                        # Extract ML metrics if available
                        if "ml_metrics" in data:
                            ml_metrics = data["ml_metrics"]
                            print(f"ML Status: {ml_metrics.get('status', 'unknown')}")
                            if "training_status" in ml_metrics:
                                training = ml_metrics["training_status"]
                                print(f"Training Operation: {training.get('current_operation', 'None')}")
                                print(f"In Progress: {training.get('in_progress', False)}")
                                print(f"Progress: {training.get('progress', 0)}%")
                                
                            if "models" in ml_metrics:
                                print("\nModels:")
                                for model in ml_metrics["models"]:
                                    print(f"  - {model.get('symbol', 'Unknown')}: {model.get('status', 'unknown')}")
                    except asyncio.TimeoutError:
                        print("No data received within timeout")
                        break
            except Exception as e:
                print(f"Error receiving data: {e}")
    except ConnectionRefusedError:
        print(f"❌ Connection refused. Is the server running on port {port}?")
        return False
    except asyncio.TimeoutError:
        print(f"❌ Connection timed out. Server at {uri} is not responding.")
        return False
    except Exception as e:
        print(f"❌ Error connecting to WebSocket: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Test WebSocket connection to trading bot dashboard")
    parser.add_argument("-p", "--port", type=int, default=3001, help="WebSocket server port (default: 3001)")
    args = parser.parse_args()
    
    print("Testing WebSocket connection to trading bot dashboard")
    result = asyncio.run(test_websocket(args.port))
    
    if result:
        print("\n✅ WebSocket test completed successfully")
        return 0
    else:
        print("\n❌ WebSocket test failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 