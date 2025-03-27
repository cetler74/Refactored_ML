import logging
import numpy as np
import pandas as pd
import torch
import os
import joblib
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)

class OnlineLearner:
    """
    Implements online learning capabilities for ML models.
    Continuously updates models with new data and manages model lifecycle.
    """
    
    def __init__(self, 
                 model_save_path: str,
                 model_registry_path: str,
                 update_interval_minutes: int = 15,
                 history_window_size: int = 5000,
                 min_samples_for_update: int = 100):
        """
        Initialize the online learner.
        
        Args:
            model_save_path: Path to save models
            model_registry_path: Path to model registry
            update_interval_minutes: How often to update models (in minutes)
            history_window_size: Maximum number of samples to keep in history
            min_samples_for_update: Minimum number of new samples required for update
        """
        self.model_save_path = model_save_path
        self.model_registry_path = model_registry_path
        self.update_interval_minutes = update_interval_minutes
        self.history_window_size = history_window_size
        self.min_samples_for_update = min_samples_for_update
        
        # Create directories if they don't exist
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(model_registry_path, exist_ok=True)
        
        # Initialize data stores
        self.feature_history = deque(maxlen=history_window_size)
        self.target_history = deque(maxlen=history_window_size)
        
        # Tracking variables
        self.last_update_time = None
        self.sample_count_since_update = 0
        self.model_versions = {}
        
        logger.info(f"Initialized online learner with update interval: {update_interval_minutes} minutes")
        
    def add_samples(self, features: pd.DataFrame, targets: pd.Series) -> int:
        """
        Add new samples to the history.
        
        Args:
            features: DataFrame with feature data
            targets: Series with target data
            
        Returns:
            Number of samples added
        """
        if features.empty or targets.empty:
            logger.warning("Empty data provided to online learner")
            return 0
        
        if len(features) != len(targets):
            logger.error(f"Feature and target lengths don't match: {len(features)} vs {len(targets)}")
            return 0
        
        # Add samples to history
        for i in range(len(features)):
            feature_row = features.iloc[i].to_dict()
            target_value = targets.iloc[i]
            
            self.feature_history.append(feature_row)
            self.target_history.append(target_value)
        
        # Update counters
        sample_count = len(features)
        self.sample_count_since_update += sample_count
        
        logger.info(f"Added {sample_count} samples to online learner (total: {len(self.feature_history)})")
        return sample_count
    
    def should_update_model(self) -> bool:
        """
        Check if the model should be updated based on time and data criteria.
        
        Returns:
            True if the model should be updated, False otherwise
        """
        # Check if we have enough new samples
        if self.sample_count_since_update < self.min_samples_for_update:
            logger.debug(f"Not enough new samples for update: {self.sample_count_since_update}/{self.min_samples_for_update}")
            return False
        
        # Check if it's time for an update
        current_time = datetime.now()
        
        if self.last_update_time is None:
            logger.info("No previous update, performing initial update")
            return True
        
        time_since_update = (current_time - self.last_update_time).total_seconds() / 60
        
        if time_since_update >= self.update_interval_minutes:
            logger.info(f"Update interval reached: {time_since_update:.1f} minutes since last update")
            return True
        
        return False
    
    def get_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get the current training data from history.
        
        Returns:
            Tuple of (features DataFrame, targets Series)
        """
        if not self.feature_history or not self.target_history:
            logger.warning("No data in history, returning empty DataFrames")
            return pd.DataFrame(), pd.Series()
        
        # Convert to DataFrame/Series
        features_df = pd.DataFrame(list(self.feature_history))
        targets_series = pd.Series(list(self.target_history))
        
        logger.info(f"Retrieved {len(features_df)} samples for training")
        return features_df, targets_series
    
    async def update_model(self, train_func, model_name: str) -> Dict[str, Any]:
        """
        Update the model with current data.
        
        Args:
            train_func: Function to train the model (takes features_df, targets_series)
            model_name: Name of the model
            
        Returns:
            Dictionary with update results
        """
        try:
            # Get training data
            features_df, targets_series = self.get_training_data()
            
            if features_df.empty or targets_series.empty:
                logger.warning("No data available for model update")
                return {"success": False, "error": "No data available"}
            
            # Train the model
            start_time = time.time()
            result = await train_func(features_df, targets_series)
            training_time = time.time() - start_time
            
            if not result.get("success", False):
                logger.error(f"Error updating model: {result.get('error', 'Unknown error')}")
                return result
            
            # Update tracking variables
            self.last_update_time = datetime.now()
            self.sample_count_since_update = 0
            
            # Update model version registry
            version = result.get("version", self.last_update_time.strftime("%Y%m%d%H%M%S"))
            self.model_versions[model_name] = {
                "version": version,
                "timestamp": self.last_update_time.isoformat(),
                "performance": result.get("performance", {}),
                "sample_count": len(features_df),
                "training_time": training_time,
                "path": result.get("model_path", "")
            }
            
            # Save model version registry
            registry_path = os.path.join(self.model_registry_path, f"{model_name}_registry.joblib")
            joblib.dump(self.model_versions, registry_path)
            
            logger.info(f"Updated {model_name} model (version: {version}) with {len(features_df)} samples in {training_time:.2f}s")
            
            return {
                "success": True,
                "model_name": model_name,
                "version": version,
                "sample_count": len(features_df),
                "training_time": training_time,
                "performance": result.get("performance", {})
            }
        
        except Exception as e:
            logger.exception(f"Error in model update: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        if model_name not in self.model_versions:
            return {"error": f"Model {model_name} not found"}
        
        return self.model_versions[model_name]
    
    def get_all_model_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all models.
        
        Returns:
            Dictionary mapping model names to their information
        """
        return self.model_versions
    
    def get_latest_version(self, model_name: str) -> Optional[str]:
        """
        Get the latest version of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Version string or None if not found
        """
        if model_name not in self.model_versions:
            return None
        
        return self.model_versions[model_name]["version"]


class ModelPerformanceTracker:
    """
    Tracks model performance metrics over time.
    Helps identify model drift and performance degradation.
    """
    
    def __init__(self, model_dir: str = None, metrics_history_size: int = 100):
        """
        Initialize the model performance tracker.
        
        Args:
            model_dir: Directory to store metrics history
            metrics_history_size: Maximum number of metric points to keep
        """
        self.metrics_history_size = metrics_history_size
        self.metrics_history = {}
        self.model_dir = model_dir
        
        # Load metrics history if it exists
        if model_dir:
            metrics_path = os.path.join(model_dir, "metrics_history.joblib")
            if os.path.exists(metrics_path):
                try:
                    self.load_metrics_history(metrics_path)
                except Exception as e:
                    logger.warning(f"Could not load metrics history: {str(e)}")
    
    def add_performance_metrics(self, 
                              model_name: str, 
                              metrics: Dict[str, float],
                              timestamp: Optional[datetime] = None) -> None:
        """
        Add performance metrics for a model.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of metric names to values
            timestamp: Timestamp of the metrics (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Initialize model entry if needed
        if model_name not in self.metrics_history:
            self.metrics_history[model_name] = {
                metric: deque(maxlen=self.metrics_history_size) 
                for metric in metrics
            }
            self.metrics_history[model_name]["timestamps"] = deque(maxlen=self.metrics_history_size)
        
        # Add metrics
        for metric, value in metrics.items():
            if metric in self.metrics_history[model_name]:
                self.metrics_history[model_name][metric].append(value)
            else:
                self.metrics_history[model_name][metric] = deque([value], maxlen=self.metrics_history_size)
        
        # Add timestamp
        self.metrics_history[model_name]["timestamps"].append(timestamp)
        
        logger.debug(f"Added performance metrics for {model_name}: {metrics}")
    
    def cleanup_inactive_metrics(self, active_models: List[str]) -> None:
        """
        Remove metrics history for models that are no longer active.
        
        OPTIMIZATION: This method removes the metrics history for trading pairs that are
        no longer actively being traded, which reduces memory usage and ensures that the
        performance metrics only reflect currently relevant trading pairs.
        
        Args:
            active_models: List of active model names to keep
        """
        # Get models to remove (those not in active_models)
        models_to_remove = [name for name in self.metrics_history.keys() 
                          if name not in active_models and not name.startswith("ensemble")]
        
        # Remove inactive models' metrics
        for model_name in models_to_remove:
            if model_name in self.metrics_history:
                del self.metrics_history[model_name]
                logger.info(f"Removed metrics history for inactive model: {model_name}")
        
        # Log summary
        if models_to_remove:
            logger.info(f"Cleaned up metrics for {len(models_to_remove)} inactive models. " 
                       f"Keeping {len(active_models)} active models plus ensemble models.")
    
    def get_latest_metrics(self, model_name: str = "ensemble") -> Dict[str, float]:
        """
        Get the latest performance metrics for a model.
        
        Args:
            model_name: Name of the model (defaults to "ensemble")
            
        Returns:
            Dictionary with the latest metrics
        """
        # Return default metrics if the model doesn't exist
        if model_name not in self.metrics_history:
            logger.warning(f"No metrics history found for model {model_name}")
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            }
        
        # Get the latest metrics for all available metrics
        latest_metrics = {}
        for metric, values in self.metrics_history[model_name].items():
            if metric != "timestamps" and len(values) > 0:
                latest_metrics[metric] = values[-1]
        
        # Ensure we have standard metrics even if they aren't in history
        result = {
            "accuracy": latest_metrics.get("forest_accuracy", latest_metrics.get("accuracy", 0.0)),
            "precision": latest_metrics.get("precision", 0.0),
            "recall": latest_metrics.get("recall", 0.0),
            "f1_score": latest_metrics.get("forest_f1", latest_metrics.get("f1_score", 0.0))
        }
        
        return result
    
    def detect_model_drift(self, 
                         model_name: str, 
                         metric: str, 
                         threshold: float = 0.1,
                         window_size: int = 10) -> Tuple[bool, float]:
        """
        Detect if a model has drifted based on a performance metric.
        
        Args:
            model_name: Name of the model
            metric: Name of the metric to check
            threshold: Threshold for drift detection
            window_size: Number of recent points to consider
            
        Returns:
            Tuple of (drift detected, drift magnitude)
        """
        if model_name not in self.metrics_history:
            logger.warning(f"No metrics history for model {model_name}")
            return False, 0.0
        
        if metric not in self.metrics_history[model_name]:
            logger.warning(f"Metric {metric} not found for model {model_name}")
            return False, 0.0
        
        # Get metric history
        metric_history = list(self.metrics_history[model_name][metric])
        
        if len(metric_history) < window_size + 1:
            logger.info(f"Not enough history for drift detection ({len(metric_history)}/{window_size+1})")
            return False, 0.0
        
        # Calculate baseline (average of points before the window)
        baseline_points = metric_history[:-window_size]
        baseline = np.mean(baseline_points)
        
        # Calculate recent average
        recent_points = metric_history[-window_size:]
        recent_avg = np.mean(recent_points)
        
        # Calculate drift
        if baseline == 0:
            drift = 0.0
        else:
            drift = (recent_avg - baseline) / baseline
        
        # Check if drift exceeds threshold
        drift_detected = abs(drift) > threshold
        
        if drift_detected:
            logger.warning(f"Model drift detected for {model_name} ({metric}): {drift:.4f} > {threshold}")
        
        return drift_detected, drift
    
    def get_metric_history(self, 
                         model_name: str, 
                         metric: str) -> Tuple[List[datetime], List[float]]:
        """
        Get the history of a metric for a model.
        
        Args:
            model_name: Name of the model
            metric: Name of the metric
            
        Returns:
            Tuple of (timestamps, values)
        """
        if model_name not in self.metrics_history:
            return [], []
        
        if metric not in self.metrics_history[model_name]:
            return [], []
        
        timestamps = list(self.metrics_history[model_name]["timestamps"])
        values = list(self.metrics_history[model_name][metric])
        
        return timestamps, values
    
    def get_all_metrics(self, model_name: str) -> Dict[str, Tuple[List[datetime], List[float]]]:
        """
        Get all metrics for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary mapping metric names to (timestamps, values) tuples
        """
        if model_name not in self.metrics_history:
            return {}
        
        result = {}
        timestamps = list(self.metrics_history[model_name]["timestamps"])
        
        for metric in self.metrics_history[model_name]:
            if metric != "timestamps":
                values = list(self.metrics_history[model_name][metric])
                result[metric] = (timestamps, values)
        
        return result
    
    def save_metrics_history(self, path: str) -> bool:
        """
        Save metrics history to a file.
        
        Args:
            path: Path to save the metrics history
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert deques to lists for serialization
            serializable_history = {}
            
            for model_name, metrics in self.metrics_history.items():
                serializable_history[model_name] = {}
                
                for metric, values in metrics.items():
                    if metric == "timestamps":
                        serializable_history[model_name][metric] = [t.isoformat() for t in values]
                    else:
                        serializable_history[model_name][metric] = list(values)
            
            joblib.dump(serializable_history, path)
            logger.info(f"Saved metrics history to {path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving metrics history: {str(e)}")
            return False
    
    def load_metrics_history(self, path: str) -> bool:
        """
        Load metrics history from a file.
        
        Args:
            path: Path to load the metrics history from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(path):
                logger.warning(f"Metrics history file {path} not found")
                return False
                
            serialized_history = joblib.load(path)
            
            # Convert lists back to deques
            for model_name, metrics in serialized_history.items():
                self.metrics_history[model_name] = {}
                
                for metric, values in metrics.items():
                    if metric == "timestamps":
                        self.metrics_history[model_name][metric] = deque(
                            [datetime.fromisoformat(t) for t in values],
                            maxlen=self.metrics_history_size
                        )
                    else:
                        self.metrics_history[model_name][metric] = deque(
                            values, maxlen=self.metrics_history_size
                        )
            
            logger.info(f"Loaded metrics history from {path}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading metrics history: {str(e)}")
            return False 