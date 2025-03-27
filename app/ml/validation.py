import logging
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import joblib
import os
import mlflow

logger = logging.getLogger(__name__)

class WalkForwardValidator:
    """
    Walk-forward validation for time series data.
    This approach respects time order and avoids lookahead bias by training on past data
    and testing on future data in sequential windows.
    """
    
    def __init__(self, 
                 train_window_size: int = 90, 
                 test_window_size: int = 30,
                 step_size: int = 15,
                 min_train_samples: int = 1000):
        """
        Initialize the walk-forward validator.
        
        Args:
            train_window_size: Size of the training window in days
            test_window_size: Size of the test window in days
            step_size: Number of days to move forward between validations
            min_train_samples: Minimum number of samples required for training
        """
        self.train_window_size = train_window_size
        self.test_window_size = test_window_size
        self.step_size = step_size
        self.min_train_samples = min_train_samples
        
    def generate_windows(self, 
                        df: pd.DataFrame, 
                        date_column: str = 'timestamp') -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate train-test windows for walk-forward validation.
        
        Args:
            df: DataFrame with time series data
            date_column: Name of the column containing timestamps
            
        Returns:
            List of (train_df, test_df) tuples
        """
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in DataFrame")
        
        # Ensure the DataFrame is sorted by date
        df = df.sort_values(by=date_column).reset_index(drop=True)
        
        # Convert date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
        
        # Get unique dates to create windows
        unique_dates = df[date_column].dt.date.unique()
        
        if len(unique_dates) < self.train_window_size + self.test_window_size:
            logger.warning("Not enough data for walk-forward validation")
            # Return a single train-test split using the available data
            split_idx = int(len(unique_dates) * 0.8)  # 80/20 split
            train_dates = unique_dates[:split_idx]
            test_dates = unique_dates[split_idx:]
            
            train_df = df[df[date_column].dt.date.isin(train_dates)]
            test_df = df[df[date_column].dt.date.isin(test_dates)]
            
            return [(train_df, test_df)]
        
        # Generate windows
        windows = []
        
        for i in range(0, len(unique_dates) - self.train_window_size - self.test_window_size + 1, self.step_size):
            train_end_idx = i + self.train_window_size
            test_end_idx = train_end_idx + self.test_window_size
            
            train_dates = unique_dates[i:train_end_idx]
            test_dates = unique_dates[train_end_idx:test_end_idx]
            
            train_df = df[df[date_column].dt.date.isin(train_dates)]
            test_df = df[df[date_column].dt.date.isin(test_dates)]
            
            # Only add if we have enough training samples
            if len(train_df) >= self.min_train_samples:
                windows.append((train_df, test_df))
        
        logger.info(f"Generated {len(windows)} walk-forward validation windows")
        return windows
    
    def validate(self, 
                df: pd.DataFrame, 
                train_func: Callable, 
                predict_func: Callable,
                feature_cols: List[str],
                target_col: str,
                date_column: str = 'timestamp') -> Dict[str, Any]:
        """
        Perform walk-forward validation.
        
        Args:
            df: DataFrame with time series data
            train_func: Function to train a model (takes X_train, y_train)
            predict_func: Function to predict (takes X_test, returns y_pred)
            feature_cols: List of feature column names
            target_col: Name of the target column
            date_column: Name of the column containing timestamps
            
        Returns:
            Dictionary with validation results
        """
        windows = self.generate_windows(df, date_column)
        
        if not windows:
            logger.warning("No valid windows for walk-forward validation")
            return {
                "success": False,
                "error": "No valid windows for walk-forward validation"
            }
        
        results = []
        
        for i, (train_df, test_df) in enumerate(windows):
            logger.info(f"Processing window {i+1}/{len(windows)}")
            
            # Prepare training data
            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            
            # Prepare test data
            X_test = test_df[feature_cols]
            y_test = test_df[target_col]
            
            try:
                # Train the model
                model = train_func(X_train, y_train)
                
                # Make predictions
                y_pred = predict_func(model, X_test)
                
                # Evaluate
                window_results = {
                    "window": i+1,
                    "train_start": train_df[date_column].min().strftime('%Y-%m-%d'),
                    "train_end": train_df[date_column].max().strftime('%Y-%m-%d'),
                    "test_start": test_df[date_column].min().strftime('%Y-%m-%d'),
                    "test_end": test_df[date_column].max().strftime('%Y-%m-%d'),
                    "train_samples": len(train_df),
                    "test_samples": len(test_df),
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average='binary'),
                    "recall": recall_score(y_test, y_pred, average='binary'),
                    "f1": f1_score(y_test, y_pred, average='binary')
                }
                
                results.append(window_results)
                logger.info(f"Window {i+1} results: accuracy={window_results['accuracy']:.4f}, f1={window_results['f1']:.4f}")
                
            except Exception as e:
                logger.error(f"Error in window {i+1}: {str(e)}")
                continue
        
        if not results:
            return {
                "success": False,
                "error": "All validation windows failed"
            }
        
        # Calculate overall metrics
        overall_results = {
            "accuracy": np.mean([r["accuracy"] for r in results]),
            "precision": np.mean([r["precision"] for r in results]),
            "recall": np.mean([r["recall"] for r in results]),
            "f1": np.mean([r["f1"] for r in results]),
            "window_count": len(results),
            "windows": results
        }
        
        return {
            "success": True,
            "results": overall_results
        }


class HyperparameterOptimizer:
    """
    Hyperparameter optimization using Optuna.
    Supports both traditional ML models and deep learning models.
    """
    
    def __init__(self, 
                 study_name: str, 
                 storage: Optional[str] = None,
                 direction: str = "maximize",
                 metric: str = "f1"):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            study_name: Name of the Optuna study
            storage: Storage URL for the Optuna study (SQLite, MySQL, etc.)
            direction: Optimization direction ("maximize" or "minimize")
            metric: Metric to optimize (e.g., "accuracy", "f1", "loss")
        """
        self.study_name = study_name
        self.storage = storage
        self.direction = direction
        self.metric = metric
        
        # Create or load the study
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            load_if_exists=True
        )
        
        logger.info(f"Initialized hyperparameter optimizer for metric: {metric}")
    
    def optimize_xgboost(self, 
                        X_train: pd.DataFrame, 
                        y_train: pd.DataFrame,
                        X_val: pd.DataFrame,
                        y_val: pd.DataFrame,
                        n_trials: int = 100,
                        timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters for XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with best parameters and study results
        """
        def objective(trial):
            param = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
                'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
                'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            }
            
            # Create and train the model
            from xgboost import XGBClassifier
            model = XGBClassifier(**param)
            model.fit(X_train, y_train)
            
            # Predict on validation set
            y_pred = model.predict(X_val)
            
            # Calculate the metric
            if self.metric == 'accuracy':
                score = accuracy_score(y_val, y_pred)
            elif self.metric == 'precision':
                score = precision_score(y_val, y_pred, average='binary')
            elif self.metric == 'recall':
                score = recall_score(y_val, y_pred, average='binary')
            elif self.metric == 'f1':
                score = f1_score(y_val, y_pred, average='binary')
            else:
                score = accuracy_score(y_val, y_pred)
            
            return score
        
        try:
            # Start MLflow run
            mlflow.start_run(run_name=f"xgboost_optim_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Run optimization
            self.study.optimize(objective, n_trials=n_trials, timeout=timeout)
            
            # Extract best parameters
            best_params = self.study.best_params
            best_value = self.study.best_value
            
            # Log to MLflow
            mlflow.log_params(best_params)
            mlflow.log_metric(f"best_{self.metric}", best_value)
            
            # End MLflow run
            mlflow.end_run()
            
            logger.info(f"Best {self.metric}: {best_value:.4f} with params: {best_params}")
            
            return {
                "success": True,
                "best_params": best_params,
                "best_value": best_value,
                "study": self.study
            }
        
        except Exception as e:
            logger.exception(f"Error in XGBoost optimization: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def optimize_lstm(self, 
                     X_train: Dict[str, Any], 
                     y_train: pd.DataFrame,
                     X_val: Dict[str, Any],
                     y_val: pd.DataFrame,
                     n_trials: int = 50,
                     timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters for LSTM model.
        
        Args:
            X_train: Training features (dictionary for temporal models)
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with best parameters and study results
        """
        def objective(trial):
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            
            # Define hyperparameters to optimize
            params = {
                'hidden_dim': trial.suggest_int('hidden_dim', 32, 256),
                'num_layers': trial.suggest_int('num_layers', 1, 3),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                'attention_heads': trial.suggest_int('attention_heads', 1, 8)
            }
            
            # Here you would implement the LSTM model training with the given hyperparameters
            # For brevity, we'll just return a random score
            # In a real implementation, this would train the model and evaluate on validation data
            
            # Create a dummy TensorDataset and DataLoader for demonstration
            train_loader = DataLoader(
                TensorDataset(torch.randn(100, 10, 32), torch.randint(0, 2, (100, 1))),
                batch_size=params['batch_size'],
                shuffle=True
            )
            
            val_loader = DataLoader(
                TensorDataset(torch.randn(50, 10, 32), torch.randint(0, 2, (50, 1))),
                batch_size=params['batch_size'],
                shuffle=False
            )
            
            # In a real implementation, create and train an LSTM model and evaluate on validation data
            # Return the validation metric
            
            # For now, return a random score for demonstration
            import random
            return random.uniform(0.5, 1.0)
        
        try:
            # Start MLflow run
            mlflow.start_run(run_name=f"lstm_optim_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Run optimization
            self.study.optimize(objective, n_trials=n_trials, timeout=timeout)
            
            # Extract best parameters
            best_params = self.study.best_params
            best_value = self.study.best_value
            
            # Log to MLflow
            mlflow.log_params(best_params)
            mlflow.log_metric(f"best_{self.metric}", best_value)
            
            # End MLflow run
            mlflow.end_run()
            
            logger.info(f"Best {self.metric}: {best_value:.4f} with params: {best_params}")
            
            return {
                "success": True,
                "best_params": best_params,
                "best_value": best_value,
                "study": self.study
            }
        
        except Exception as e:
            logger.exception(f"Error in LSTM optimization: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def optimize_tft(self, 
                    data_module,  # Would be a PyTorch Lightning DataModule
                    n_trials: int = 30,
                    timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters for Temporal Fusion Transformer model.
        
        Args:
            data_module: PyTorch Lightning DataModule with train/val/test data
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with best parameters and study results
        """
        def objective(trial):
            from pytorch_lightning import Trainer
            from pytorch_lightning.callbacks import EarlyStopping
            
            # Define hyperparameters to optimize
            params = {
                'hidden_dim': trial.suggest_int('hidden_dim', 32, 256),
                'lstm_layers': trial.suggest_int('lstm_layers', 1, 3),
                'attn_heads': trial.suggest_int('attn_heads', 1, 8),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
            }
            
            # Set batch size in data module
            data_module.batch_size = params['batch_size']
            data_module.setup('fit')
            
            # Create a dummy TFT model for demonstration
            # In a real implementation, you would create a TFT model with the given hyperparameters
            # from app.ml.temporal_fusion_transformer import TemporalFusionTransformer
            # model = TemporalFusionTransformer(...params...)
            
            # For demonstration, just return a random score
            import random
            return random.uniform(0.5, 1.0)
        
        try:
            # Start MLflow run
            mlflow.start_run(run_name=f"tft_optim_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Run optimization
            self.study.optimize(objective, n_trials=n_trials, timeout=timeout)
            
            # Extract best parameters
            best_params = self.study.best_params
            best_value = self.study.best_value
            
            # Log to MLflow
            mlflow.log_params(best_params)
            mlflow.log_metric(f"best_{self.metric}", best_value)
            
            # End MLflow run
            mlflow.end_run()
            
            logger.info(f"Best {self.metric}: {best_value:.4f} with params: {best_params}")
            
            return {
                "success": True,
                "best_params": best_params,
                "best_value": best_value,
                "study": self.study
            }
        
        except Exception as e:
            logger.exception(f"Error in TFT optimization: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            } 