import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class DynamicTimeframeSelector:
    """
    Dynamically selects the optimal timeframe for trading based on market conditions.
    Uses XGBoost to classify which timeframe is most effective for current conditions.
    """
    
    def __init__(self, timeframes: List[str] = None):
        self.timeframes = timeframes or ["1m", "5m", "15m", "1h"]
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.last_training_time = None
        self.min_samples_for_training = 100
        self.training_data = pd.DataFrame()
        logger.info(f"Initialized DynamicTimeframeSelector with timeframes: {self.timeframes}")
    
    def prepare_training_features(self, 
                                market_data: Dict[str, Dict[str, pd.DataFrame]],
                                performance_history: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare features for training the timeframe selection model.
        
        Args:
            market_data: Dictionary mapping symbol -> {timeframe -> DataFrame}
            performance_history: Performance metrics by timeframe (optional)
            
        Returns:
            DataFrame with features and target timeframe labels
        """
        if not market_data:
            return pd.DataFrame()
        
        # Extract features from different timeframes for each symbol
        all_features = []
        
        for symbol, timeframes_data in market_data.items():
            # Calculate common metrics across timeframes
            symbol_features = {}
            
            # Get volatility metrics for each timeframe
            for tf, df in timeframes_data.items():
                if df.empty or len(df) < 20:
                    continue
                
                # Calculate volatility metrics
                symbol_features[f"{tf}_volatility"] = df['natr'].iloc[-1] if 'natr' in df.columns else np.nan
                symbol_features[f"{tf}_volume_surge"] = (df['volume'].iloc[-1] / df['volume'].iloc[-20:].mean()) if 'volume' in df.columns else np.nan
                symbol_features[f"{tf}_rsi"] = df['rsi_14'].iloc[-1] if 'rsi_14' in df.columns else np.nan
                symbol_features[f"{tf}_trend_strength"] = df['adx'].iloc[-1] if 'adx' in df.columns else np.nan
                
                # Calculate price momentum
                if 'close' in df.columns and len(df) >= 5:
                    symbol_features[f"{tf}_momentum"] = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1) * 100
                
                # Calculate Bollinger Band metrics
                if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'close']):
                    symbol_features[f"{tf}_bb_width"] = (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1]) / df['close'].iloc[-1]
                    symbol_features[f"{tf}_bb_position"] = (df['close'].iloc[-1] - df['bb_lower'].iloc[-1]) / (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1])
            
            # Add symbol as a feature
            symbol_features['symbol'] = symbol
            
            # Add timestamp
            symbol_features['timestamp'] = datetime.now()
            
            # If we have performance history, add the best performing timeframe as target
            if performance_history is not None and not performance_history.empty:
                # Filter performance for this symbol
                symbol_perf = performance_history[performance_history['symbol'] == symbol]
                
                if not symbol_perf.empty:
                    # Get the timeframe with best performance
                    best_tf = symbol_perf.iloc[symbol_perf['performance'].argmax()]['timeframe']
                    symbol_features['best_timeframe'] = best_tf
                    
                    # Add this to our training data
                    all_features.append(symbol_features)
            
            # If we're just collecting data without a target yet
            elif 'best_timeframe' in symbol_features:
                all_features.append(symbol_features)
        
        if not all_features:
            return pd.DataFrame()
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        return features_df
    
    def train_model(self, training_data: pd.DataFrame) -> bool:
        """
        Train the XGBoost model for timeframe selection.
        
        Args:
            training_data: DataFrame with features and target timeframe labels
            
        Returns:
            Boolean indicating success
        """
        if training_data.empty or 'best_timeframe' not in training_data.columns:
            logger.warning("Cannot train timeframe selector: no training data or missing target")
            return False
        
        # Drop rows with missing target
        training_data = training_data.dropna(subset=['best_timeframe'])
        
        if len(training_data) < self.min_samples_for_training:
            logger.info(f"Not enough training samples ({len(training_data)}/{self.min_samples_for_training})")
            
            # Append to existing training data
            self.training_data = pd.concat([self.training_data, training_data], ignore_index=True)
            self.training_data = self.training_data.drop_duplicates(subset=['symbol', 'timestamp'])
            
            if len(self.training_data) >= self.min_samples_for_training:
                logger.info(f"Accumulated enough samples ({len(self.training_data)}), training model")
                return self.train_model(self.training_data)
            
            return False
        
        try:
            # Get feature columns (exclude symbol, timestamp, and target)
            numeric_cols = training_data.select_dtypes(include=[np.number]).columns.tolist()
            self.feature_columns = [col for col in numeric_cols if col not in ['best_timeframe']]
            
            # Prepare features and target
            X = training_data[self.feature_columns]
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Normalize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Encode the target
            target_mapping = {tf: i for i, tf in enumerate(self.timeframes)}
            y = training_data['best_timeframe'].map(target_mapping)
            
            # Split into train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Initialize and train XGBoost classifier
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                objective='multi:softprob',
                num_class=len(self.timeframes),
                random_state=42
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate the model
            accuracy = self.model.score(X_test, y_test)
            logger.info(f"Timeframe selector model trained with accuracy: {accuracy:.4f}")
            
            # Record training time
            self.last_training_time = datetime.now()
            
            # Get feature importances
            feature_importances = dict(zip(self.feature_columns, self.model.feature_importances_))
            top_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"Top features for timeframe selection: {top_features}")
            
            return True
        
        except Exception as e:
            logger.exception(f"Error training timeframe selector model: {str(e)}")
            return False
    
    def select_optimal_timeframe(self, 
                                symbol: str, 
                                market_data: Dict[str, pd.DataFrame],
                                ml_confidence: Optional[Dict[str, float]] = None) -> str:
        """
        Select the optimal timeframe for trading based on current market conditions.
        
        Args:
            symbol: Trading symbol
            market_data: Dictionary mapping timeframes to DataFrames
            ml_confidence: Dictionary mapping timeframes to ML confidence scores (optional)
            
        Returns:
            Selected timeframe string
        """
        # If model not trained, return the longest timeframe as default
        if self.model is None or self.feature_columns is None:
            logger.info(f"Model not trained, using default timeframe: {self.timeframes[-1]}")
            return self.timeframes[-1]
        
        try:
            # Extract features for prediction
            features = {}
            
            # Get volatility metrics for each timeframe
            for tf, df in market_data.items():
                if df.empty or len(df) < 20:
                    continue
                
                # Calculate the same features used during training
                features[f"{tf}_volatility"] = df['natr'].iloc[-1] if 'natr' in df.columns else np.nan
                features[f"{tf}_volume_surge"] = (df['volume'].iloc[-1] / df['volume'].iloc[-20:].mean()) if 'volume' in df.columns else np.nan
                features[f"{tf}_rsi"] = df['rsi_14'].iloc[-1] if 'rsi_14' in df.columns else np.nan
                features[f"{tf}_trend_strength"] = df['adx'].iloc[-1] if 'adx' in df.columns else np.nan
                
                if 'close' in df.columns and len(df) >= 5:
                    features[f"{tf}_momentum"] = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1) * 100
                
                if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'close']):
                    features[f"{tf}_bb_width"] = (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1]) / df['close'].iloc[-1]
                    features[f"{tf}_bb_position"] = (df['close'].iloc[-1] - df['bb_lower'].iloc[-1]) / (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1])
            
            # Add ML confidence scores if available
            if ml_confidence:
                for tf, confidence in ml_confidence.items():
                    features[f"{tf}_ml_confidence"] = confidence
            
            # Create feature vector
            X = pd.DataFrame([features])
            
            # Ensure all required features are present
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = np.nan
            
            # Reorder columns to match the training order
            X = X[self.feature_columns]
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Normalize features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            timeframe_probs = self.model.predict_proba(X_scaled)[0]
            selected_index = np.argmax(timeframe_probs)
            selected_timeframe = self.timeframes[selected_index]
            
            logger.info(f"Selected optimal timeframe for {symbol}: {selected_timeframe} (confidence: {timeframe_probs[selected_index]:.2f})")
            
            # Include the probabilities for all timeframes in debug
            probs_by_tf = {tf: timeframe_probs[i] for i, tf in enumerate(self.timeframes)}
            logger.debug(f"Timeframe probabilities for {symbol}: {probs_by_tf}")
            
            return selected_timeframe
        
        except Exception as e:
            logger.exception(f"Error selecting optimal timeframe: {str(e)}")
            # Fall back to the longest timeframe as default
            return self.timeframes[-1]
    
    def update_performance(self, 
                          symbol: str, 
                          timeframe: str, 
                          performance_metric: float) -> None:
        """
        Update the performance record for a timeframe to improve future selections.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe that was used
            performance_metric: Performance measure (profit/loss, win rate, etc.)
        """
        if not hasattr(self, 'performance_history'):
            # Initialize performance history
            self.performance_history = pd.DataFrame(columns=['symbol', 'timeframe', 'performance', 'timestamp'])
        
        # Add new performance record
        new_record = pd.DataFrame([{
            'symbol': symbol,
            'timeframe': timeframe,
            'performance': performance_metric,
            'timestamp': datetime.now()
        }])
        
        # Append to performance history
        self.performance_history = pd.concat([self.performance_history, new_record], ignore_index=True)
        
        # Limit history size
        max_history_per_symbol = 100
        self.performance_history = self.performance_history.groupby('symbol').apply(
            lambda x: x.nlargest(max_history_per_symbol, 'timestamp')
        ).reset_index(drop=True)
        
        logger.debug(f"Updated performance history for {symbol} {timeframe}: {performance_metric:.4f}") 