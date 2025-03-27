# model.py
import logging
import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import mlflow
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder # Added LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, Dataset # Use Dataset
import shap
from matplotlib import pyplot as plt
import io
import base64
import time
import asyncio
import json
import pickle
import talib
import random

# --- Import TFT and other custom modules ---
from app.ml.temporal_fusion_transformer import TemporalFusionTransformer # <<< Import the actual TFT
# Assuming other helpers are correctly located
from app.ml.validation import WalkForwardValidator, HyperparameterOptimizer
from app.ml.online_learning import OnlineLearner, ModelPerformanceTracker
from app.ml.cross_correlation import CrossPairCorrelationAnalyzer
# from app.config.settings import Settings # Assuming settings provides config

logger = logging.getLogger(__name__)

# --- LSTM Model and Lightning Module (Keep as before) ---
class LSTMModel(nn.Module):
    # ... (Implementation from previous response) ...
    pass
class TimeSeriesLightningModule(pl.LightningModule):
    # ... (Implementation from previous response) ...
    pass

# --- TFT Dataset Class (Moved here or keep in separate datasets.py) ---
class TFTDataset(Dataset):
    def __init__(self, data: Dict[str, Any], targets: np.ndarray):
        """
        data: Dictionary containing keys like 'static', 'encoder_cat', 'encoder_real', etc.
              Values are dictionaries mapping feature names to numpy arrays.
        targets: Numpy array of target values [num_samples].
        """
        self.data = data
        self.targets = targets
        # Determine length from a required time-varying input (e.g., encoder_real)
        self.length = 0
        if data.get('encoder_real'):
             first_real_key = next(iter(data['encoder_real']), None)
             if first_real_key: self.length = len(data['encoder_real'][first_real_key])
        elif data.get('encoder_cat'): # Fallback to categoricals
             first_cat_key = next(iter(data['encoder_cat']), None)
             if first_cat_key: self.length = len(data['encoder_cat'][first_cat_key])

        if self.length == 0:
            logger.warning("Could not determine dataset length from TFT input data.")
            self.length = len(targets) # Fallback, might be wrong if data is malformed

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item_data = {}
        # Helper to get item from nested dicts, converting to tensors
        def get_nested_item(nested_dict):
            if nested_dict is None: return None
            # Handle potential dtype issues during tensor conversion
            tensor_dict = {}
            for k, v in nested_dict.items():
                try:
                     # Determine dtype
                     if np.issubdtype(v.dtype, np.floating): dtype = torch.float32
                     elif np.issubdtype(v.dtype, np.integer): dtype = torch.long # Use long for categoricals/IDs
                     else: dtype = torch.float32 # Default fallback
                     tensor_dict[k] = torch.tensor(v[idx], dtype=dtype)
                except IndexError:
                     logger.error(f"IndexError accessing feature '{k}' at index {idx}. Array length: {len(v)}. Dataset length: {self.length}")
                     # Handle error - maybe return None or raise? Returning zeros for now.
                     # Need to know the expected shape. Assume feature dim is 1 if error.
                     time_dim = self.data.get('encoder_real',{}).get(k, np.array([])).shape[1] if v.ndim > 1 else 0
                     feat_dim = 1
                     shape = (time_dim, feat_dim) if time_dim > 0 else (feat_dim,)
                     tensor_dict[k] = torch.zeros(shape, dtype=dtype) # Placeholder on error
                except Exception as e:
                     logger.error(f"Error converting feature '{k}' at index {idx} to tensor: {e}")
                     time_dim = self.data.get('encoder_real',{}).get(k, np.array([])).shape[1] if v.ndim > 1 else 0
                     feat_dim = 1
                     shape = (time_dim, feat_dim) if time_dim > 0 else (feat_dim,)
                     tensor_dict[k] = torch.zeros(shape, dtype=dtype) # Placeholder on error

            return tensor_dict

        # Populate item_data using the helper
        item_data['static_inputs'] = get_nested_item(self.data.get('static'))
        # Map prepared data keys to expected TFT forward keys
        item_data['encoder_known_categoricals'] = get_nested_item(self.data.get('encoder_cat')) # Assuming encoder_cat holds known cats
        item_data['encoder_known_reals'] = get_nested_item(self.data.get('encoder_real'))      # Assuming encoder_real holds known reals
        item_data['encoder_observed_categoricals'] = None # Populate if you have observed-only cats
        item_data['encoder_observed_reals'] = None      # Populate if you have observed-only reals
        item_data['decoder_known_categoricals'] = get_nested_item(self.data.get('decoder_cat'))
        item_data['decoder_known_reals'] = get_nested_item(self.data.get('decoder_real'))
        # item_data['encoder_lengths'] = ... (If using packing)
        # item_data['decoder_lengths'] = ... (If using packing)

        # Target tensor
        try:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
        except IndexError:
             logger.error(f"IndexError accessing target at index {idx}. Target length: {len(self.targets)}. Dataset length: {self.length}")
             target = torch.tensor(0.0, dtype=torch.float32) # Placeholder target on error
        except Exception as e:
             logger.error(f"Error converting target at index {idx} to tensor: {e}")
             target = torch.tensor(0.0, dtype=torch.float32)

        return item_data, target


# --- Updated TFT Lightning Module ---
class TFTLightningModule(pl.LightningModule):
    """PyTorch Lightning module wrapper for the Temporal Fusion Transformer model."""
    def __init__(self,
                 tft_model_instance: TemporalFusionTransformer, # Pass the initialized model
                 learning_rate: float = 0.001,
                 target_idx: int = -1, # Index of target step in decoder output (e.g., -1 for last)
                 ):
        super().__init__()
        # Store hparams for saving/loading, ignore the model instance itself
        # Need to save parameters needed to re-initialize tft_model_instance on load
        self.save_hyperparameters('learning_rate', 'target_idx')
        self.model = tft_model_instance
        # Criterion for binary classification (assumes model outputs logits)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # x contains keys like 'static_inputs', 'encoder_known_categoricals', etc.
        # Pass dictionary directly to the TFT model's forward method
        return self.model(x) # Use direct call, model expects the dict

    def step(self, batch, batch_idx, step_type):
        x, y = batch # x is the dictionary of features, y is the target tensor [batch] or [batch, 1]

        try:
            # Pass input dictionary 'x' to the model
            predictions_all_steps, _ = self(x) # Get predictions, ignore interpretability weights here
            # predictions_all_steps shape: [batch_size, decoder_length, num_outputs]

            # Select the prediction for the relevant time step (target_idx) and output dim (0)
            pred_target_step = predictions_all_steps[:, self.hparams.target_idx, 0] # Shape: [batch_size]

            # Ensure target 'y' has the correct shape [batch_size] and type
            target = y.squeeze().float()

            # Calculate loss
            loss = self.criterion(pred_target_step, target)

            # Calculate metrics (use sigmoid for probabilities)
            probs = torch.sigmoid(pred_target_step)
            preds = (probs > 0.5).long() # Convert to 0/1 predictions
            target_int = target.long() # Ensure target is Long for metrics

            # Handle potential issues with metrics calculation (e.g., single class in batch)
            try:
                 acc = accuracy_score(target_int.cpu().numpy(), preds.cpu().numpy())
            except ValueError: acc = 0.5 # Default if accuracy cannot be calculated
            try:
                 f1 = f1_score(target_int.cpu().numpy(), preds.cpu().numpy(), zero_division=0)
            except ValueError: f1 = 0.0 # Default if F1 cannot be calculated


            log_dict = {
                f'{step_type}_loss': loss,
                f'{step_type}_acc': acc,
                f'{step_type}_f1': f1
            }
            self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, batch_size=target.size(0))
            return loss

        except Exception as e:
             logger.error(f"Error during {step_type} step {batch_idx}: {e}", exc_info=True)
             # Return a dummy loss or handle appropriately
             return torch.tensor(0.0, requires_grad=True, device=self.device) # Return zero loss on error?


    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        # PyTorch Lightning automatically handles evaluation mode during validation
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test")

    def configure_optimizers(self):
        # Use Adam optimizer with the specified learning rate
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # Optional: Add learning rate scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        return optimizer


# --- Updated MLModule ---
class MLModule:
    # ... (Keep __init__ and helper methods like _save/load_tft_config, _get_device, tft_config_for_init, _load_models from previous response) ...
    def __init__(self, settings: Any, db_manager=None):
        # ... (init logic from previous response, including TFT config handling) ...
        pass

    def _save_tft_config(self):
        # ... (implementation from previous response) ...
        pass

    def _load_tft_config(self):
        # ... (implementation from previous response) ...
        pass

    def _load_models(self):
        # ... (implementation from previous response, using tft_config_for_init) ...
        pass

    def _get_device(self):
        # ... (implementation from previous response) ...
        pass

    def tft_config_for_init(self) -> Dict:
        # ... (implementation from previous response) ...
        pass


    # --- Updated prepare_training_data ---
    async def prepare_training_data(self, market_data_dict: Dict[str, pd.DataFrame]) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
        """
        Prepare feature and target data specifically for the Temporal Fusion Transformer.
        Updates self.tft_config with feature dimensions.
        """
        # ... (Use the detailed implementation from the previous response) ...
        # Key parts:
        # 1. Define feature categories (static, enc_cat, enc_real, dec_cat, dec_real) - USER MUST CUSTOMIZE
        # 2. Loop through symbols in market_data_dict
        # 3. Perform feature engineering (time features, indicators, custom features)
        # 4. Apply scaling robustly (using saved scalers or fitting appropriately)
        # 5. Define the binary target based on future price movement
        # 6. Handle NaNs carefully after shifts and indicator calculations
        # 7. Create sequences (encoder/decoder slices) for each valid starting point
        # 8. Populate the `all_samples` dictionary with numpy arrays for each feature category
        # 9. Convert `all_samples` lists into stacked numpy arrays in `final_data_dict`
        # 10. Update `self.tft_config` with the *actual determined dimensions* of the features
        # 11. Save the updated `self.tft_config`
        # 12. Return `final_data_dict`, `targets_array`
        # --- Placeholder for the detailed logic ---
        logger.info("Preparing training data for Temporal Fusion Transformer...")
        # [ ***** PASTE THE DETAILED IMPLEMENTATION FROM THE PREVIOUS RESPONSE HERE ***** ]
        # Example Snippet:
        # ... loop through symbols ...
        #     ... feature eng ...
        #     ... scaling ...
        #     ... target def ...
        #     ... NaN handling ...
        #     ... loop through slices (i) ...
        #         ... populate all_samples[category][feature_name].append(slice_data) ...
        # ... convert lists in all_samples to numpy arrays in final_data_dict ...
        # ... update self.tft_config with determined dimensions ...
        # ... self._save_tft_config() ...
        # return final_data_dict, targets_array
        # --- End Placeholder ---
        # Returning dummy data for now until the full logic is pasted
        logger.warning("prepare_training_data for TFT needs full implementation pasted.")
        return None, None # Requires full implementation

    # --- LSTM sequence helper (unchanged) ---
    def _create_sequences(self, features: np.ndarray, targets: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        # ... (Implementation from previous response) ...
        pass

    # --- Load flat data helper (unchanged) ---
    def _load_and_filter_training_data(self, active_pairs: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        # ... (Implementation from previous response) ...
        pass

    # --- Final Model Training Methods ---
    async def _train_final_forest_model(self, features_df: pd.DataFrame, targets_df: pd.DataFrame) -> Dict[str, Any]:
        # ... (Implementation from previous response) ...
        pass
    async def _train_final_lstm_model(self, features_df: pd.DataFrame, targets_df: pd.DataFrame) -> Dict[str, Any]:
        # ... (Implementation from previous response) ...
        pass
    async def _train_final_tft_model(self, data_dict: Dict[str, Any], targets_array: np.ndarray) -> Dict[str, Any]:
        # ... (Use the implementation from the previous response, ensuring TFTDataset and TFTLightningModule are used) ...
        pass

    # --- Main Train Method ---
    async def train_models(self, market_data_dict: Dict[str, pd.DataFrame], active_pairs: List[str] = None) -> Dict[str, Any]:
        # ... (Use the implementation from the previous response which calls the correct prep and train methods) ...
        pass

    # --- Prediction Methods ---
    async def make_predictions(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        # ... (Use the implementation from the previous response which includes TFT prediction logic and _prepare_single_tft_input) ...
        pass
    def _prepare_single_tft_input(self, df_hist: pd.DataFrame) -> Optional[Dict[str, Any]]:
        # ... (Use the implementation from the previous response - CRITICAL FOR TFT PREDICTION) ...
        pass

    # --- Other Methods ---
    # Keep all other methods (feedback loop, SHAP, evaluation, etc.)
    # Ensure feature names used in feedback/evaluation align with training.
    async def analyze_cross_pair_correlations(self, market_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        # ... (Implementation from previous response) ...
        pass
    async def incorporate_trade_results(self, trade_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        # ... (Implementation from previous response) ...
        pass
    async def _update_training_dataset(self, feedback_data: List[Dict[str, Any]]) -> None:
        # ... (Implementation from previous response) ...
        pass
    async def _should_retrain_models(self, num_new_records: int) -> bool:
        # ... (Implementation from previous response) ...
        pass
    async def _retrain_models_with_feedback(self) -> None:
        # ... (Implementation from previous response) ...
        pass
    async def _train_feedback_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        # ... (Implementation from previous response) ...
        pass
    def _log_feedback_model_mlflow(self, model, metrics: Dict, feature_names: List[str]):
        # ... (Implementation from previous response) ...
        pass
    def _calculate_shap_values(self, X: pd.DataFrame, feature_names: List[str]) -> Optional[Dict[str, Any]]:
        # ... (Implementation from previous response - Note: SHAP for TFT is much more complex) ...
        # SHAP currently only works well for the tree-based model here.
        pass
    async def explain_prediction(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        # ... (Implementation from previous response - Note: Based on Forest SHAP) ...
        pass
    async def evaluate_exit_decision(self, position_data: Dict[str, Any], market_data: pd.DataFrame = None) -> Dict[str, Any]:
        # ... (Implementation from previous response) ...
        pass
    def _evaluate_exit_with_default_logic(self, position_data: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        # ... (Implementation from previous response) ...
        pass
    def custom_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        # ... (Implementation from previous response) ...
        pass
    def _validate_model_data(self, features_df: pd.DataFrame, targets_df: pd.DataFrame) -> Dict[str, Any]:
        # ... (Implementation from previous response - basic check) ...
        pass

# --- End of MLModule class ---