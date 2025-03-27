import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Mock ML metrics for dashboard testing
def get_enhanced_ml_metrics() -> Dict[str, Any]:
    """
    Generate enhanced ML metrics with more details for the dashboard.
    This is used when actual ML module isn't available.
    """
    now = datetime.now()
    
    # Sample trading pairs
    active_pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT"]
    
    # Generate model info for active pairs
    models_info = []
    for pair in active_pairs:
        # Create mock model data
        model_status = "trained" if pair in ["BTC/USDT", "ETH/USDT"] else "not_trained"
        model_accuracy = 0.75 if model_status == "trained" else 0.0
        
        models_info.append({
            "symbol": pair,
            "model_type": "ensemble",
            "status": model_status,
            "accuracy": model_accuracy,
            "last_training": (now - timedelta(days=1)).isoformat(),
            "next_training": (now + timedelta(hours=23)).isoformat(),
            "samples": 1240 if model_status == "trained" else 0
        })
    
    # Generate mock training status
    training_status = {
        "in_progress": False,
        "current_operation": "Idle",
        "progress": 0,
        "eta": "Unknown"
    }
    
    # Generate mock model stats
    model_stats = {
        "trained": 2,  # BTC and ETH
        "total": len(active_pairs)
    }
    
    # Generate mock prediction stats
    prediction_stats = {
        "total": 150,
        "success_rate": 68.5,
        "avg_confidence": 0.72,
        "signals_generated": 42,
        "data_points": 1240,
        "avg_prediction_time": 125  # ms
    }
    
    # Generate mock model health
    model_health = {
        "status": "healthy",
        "message": "Models are up to date"
    }
    
    # Mock predictions data for charts
    predictions = []
    for i in range(10):
        prediction_time = now - timedelta(hours=i*8)
        pair = active_pairs[i % len(active_pairs)]
        predictions.append({
            "timestamp": prediction_time.isoformat(),
            "symbol": pair,
            "prediction": "buy" if i % 2 == 0 else "sell",
            "confidence": 0.65 + (i % 3) * 0.1,
            "accuracy": 1 if i % 3 != 1 else 0  # 2/3 correct predictions
        })
    
    # Mock feature importance
    feature_importance = {
        "close_delta": 0.25,
        "volume": 0.18,
        "rsi_14": 0.15,
        "macd": 0.12,
        "ema_crossover": 0.10,
        "bollinger_bands": 0.08,
        "adx": 0.07,
        "obv": 0.05
    }
    
    return {
        "status": "available",
        "accuracy": 0.75,
        "precision": 0.73,
        "recall": 0.70,
        "f1_score": 0.72,
        "timestamp": now.isoformat(),
        "models": models_info,
        "training_status": training_status,
        "model_stats": model_stats,
        "prediction_stats": prediction_stats,
        "model_health": model_health,
        "last_training_cycle": (now - timedelta(days=1)).isoformat(),
        "next_training_cycle": (now + timedelta(hours=23)).isoformat(),
        "training_frequency": "Every 24 hours",
        "predictions": predictions,
        "feature_importance": feature_importance
    }

if __name__ == "__main__":
    # Test the function
    metrics = get_enhanced_ml_metrics()
    print(json.dumps(metrics, indent=2)) 