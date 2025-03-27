#!/usr/bin/env python
"""
Dashboard Test Script

This script tests the connections to the API server and WebSocket server.
"""

import asyncio
import websockets
import requests
import json

async def test_websocket():
    """Test WebSocket connection."""
    try:
        uri = "ws://localhost:8080/ws"
        async with websockets.connect(uri) as websocket:
            print(f"✅ WebSocket connection successful to {uri}")
            
            # Wait for some data
            print("Waiting for data from WebSocket...")
            for _ in range(3):  # Try to get 3 messages
                try:
                    data = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    print(f"Received data: {data[:100]}..." if len(data) > 100 else f"Received data: {data}")
                except asyncio.TimeoutError:
                    print("No data received within timeout")
                    break
    except Exception as e:
        print(f"❌ WebSocket connection failed: {e}")

def test_api():
    """Test API endpoints."""
    endpoints = [
        "http://localhost:8080/health",
        "http://localhost:8080/portfolio",
        "http://localhost:8080/ml/metrics"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint)
            print(f"✅ API endpoint {endpoint}: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        except Exception as e:
            print(f"❌ API endpoint {endpoint} failed: {e}")

async def main():
    """Run all tests."""
    print("=== Testing API Endpoints ===")
    test_api()
    
    print("\n=== Testing WebSocket Connection ===")
    await test_websocket()

if __name__ == "__main__":
    asyncio.run(main()) 