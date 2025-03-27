import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class CrossPairCorrelationAnalyzer:
    """
    Analyzes correlations between different trading pairs.
    Useful for identifying leading indicators and market patterns across assets.
    """
    
    def __init__(self, lookback_periods: int = 30, correlation_threshold: float = 0.7):
        """
        Initialize the cross-pair correlation analyzer.
        
        Args:
            lookback_periods: Number of periods to use for rolling correlation
            correlation_threshold: Threshold for significant correlation
        """
        self.lookback_periods = lookback_periods
        self.correlation_threshold = correlation_threshold
        self.pair_correlations = {}
        self.leading_indicators = {}
        self.scaler = StandardScaler()
        
    def calculate_correlations(self, market_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Calculate pairwise correlations between all trading pairs.
        
        Args:
            market_data_dict: Dictionary of DataFrames with price data by symbol
            
        Returns:
            Dictionary of correlation matrices by metric
        """
        if not market_data_dict:
            logger.warning("No market data provided for correlation analysis")
            return {}
        
        # Extract close prices for each symbol
        price_data = {}
        for symbol, df in market_data_dict.items():
            if df.empty:
                continue
                
            # Make sure we have the required columns
            if 'close' not in df.columns:
                logger.warning(f"No close price data for {symbol}, skipping")
                continue
                
            # Resample to ensure same timestamps if needed
            price_data[symbol] = df['close']
        
        # Create a DataFrame with all price data
        if not price_data:
            logger.warning("No valid price data for correlation analysis")
            return {}
            
        all_prices_df = pd.DataFrame(price_data)
        
        # Calculate returns
        returns_df = all_prices_df.pct_change().dropna()
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        # Calculate rolling correlations
        rolling_correlations = {}
        for symbol1 in returns_df.columns:
            for symbol2 in returns_df.columns:
                if symbol1 != symbol2:
                    # Calculate rolling correlation
                    symbol_pair = f"{symbol1}_vs_{symbol2}"
                    rolling_corr = returns_df[symbol1].rolling(self.lookback_periods).corr(returns_df[symbol2])
                    rolling_correlations[symbol_pair] = rolling_corr
        
        # Store results
        self.pair_correlations = {
            'full_matrix': correlation_matrix,
            'rolling': pd.DataFrame(rolling_correlations)
        }
        
        logger.info(f"Calculated correlations for {len(price_data)} trading pairs")
        return self.pair_correlations
    
    def find_leading_indicators(self, 
                              market_data_dict: Dict[str, pd.DataFrame], 
                              max_lag: int = 10) -> Dict[str, List[Tuple[str, int, float]]]:
        """
        Find leading indicators by calculating lagged correlations.
        
        Args:
            market_data_dict: Dictionary of DataFrames with price data by symbol
            max_lag: Maximum lag to consider in periods
            
        Returns:
            Dictionary mapping symbols to their leading indicators (symbol, lag, correlation)
        """
        if not market_data_dict:
            logger.warning("No market data provided for leading indicator analysis")
            return {}
        
        # Extract returns for each symbol
        returns_data = {}
        for symbol, df in market_data_dict.items():
            if df.empty:
                continue
                
            # Make sure we have the required columns
            if 'close' not in df.columns:
                logger.warning(f"No close price data for {symbol}, skipping")
                continue
                
            # Calculate returns
            returns_data[symbol] = df['close'].pct_change().dropna()
        
        # Find leading indicators
        leading_indicators = {}
        
        for target_symbol, target_returns in returns_data.items():
            symbol_leaders = []
            
            for source_symbol, source_returns in returns_data.items():
                if source_symbol == target_symbol:
                    continue
                
                # Align data
                min_length = min(len(target_returns), len(source_returns))
                if min_length <= max_lag:
                    continue
                    
                target_aligned = target_returns.iloc[-min_length:].values
                source_aligned = source_returns.iloc[-min_length:].values
                
                # Calculate lagged correlations
                best_lag = 0
                best_corr = 0
                
                for lag in range(1, max_lag + 1):
                    # Shift source by lag
                    lagged_source = source_aligned[:-lag]
                    lagged_target = target_aligned[lag:]
                    
                    if len(lagged_source) < 10:  # Need enough data points
                        continue
                        
                    # Calculate correlation
                    corr = np.corrcoef(lagged_source, lagged_target)[0, 1]
                    
                    # Check if this is the best correlation
                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag = lag
                
                # If correlation is significant, add to leading indicators
                if abs(best_corr) >= self.correlation_threshold:
                    symbol_leaders.append((source_symbol, best_lag, best_corr))
            
            # Sort by absolute correlation (strongest first)
            symbol_leaders.sort(key=lambda x: abs(x[2]), reverse=True)
            leading_indicators[target_symbol] = symbol_leaders
        
        self.leading_indicators = leading_indicators
        
        logger.info(f"Found leading indicators for {len(leading_indicators)} symbols")
        return leading_indicators
    
    def get_cross_pair_features(self, 
                              market_data_dict: Dict[str, pd.DataFrame],
                              target_symbol: str) -> pd.DataFrame:
        """
        Generate cross-pair features for a target symbol based on correlations.
        
        Args:
            market_data_dict: Dictionary of DataFrames with price data by symbol
            target_symbol: Symbol to generate features for
            
        Returns:
            DataFrame with cross-pair features
        """
        if target_symbol not in market_data_dict:
            logger.warning(f"Target symbol {target_symbol} not found in market data")
            return pd.DataFrame()
        
        # Find leading indicators first if not already calculated
        if not self.leading_indicators:
            self.find_leading_indicators(market_data_dict)
        
        # Get leading indicators for the target symbol
        target_leaders = self.leading_indicators.get(target_symbol, [])
        
        if not target_leaders:
            logger.info(f"No significant leading indicators found for {target_symbol}")
            return pd.DataFrame()
        
        # Create features from leading indicators
        features = {}
        target_df = market_data_dict[target_symbol]
        
        for leader_symbol, lag, corr in target_leaders:
            if leader_symbol not in market_data_dict:
                continue
                
            leader_df = market_data_dict[leader_symbol]
            
            # Extract key metrics
            for metric in ['close', 'volume', 'high', 'low']:
                if metric not in leader_df.columns or metric not in target_df.columns:
                    continue
                    
                # Get values and calculate returns
                leader_values = leader_df[metric].values
                leader_returns = np.diff(leader_values) / leader_values[:-1]
                
                # Ensure we have enough data
                if len(leader_returns) <= lag:
                    continue
                    
                # Create lagged feature
                feature_name = f"{leader_symbol}_{metric}_lag{lag}"
                
                # Align with target data (shift by lag)
                lagged_values = np.roll(leader_returns, lag)
                lagged_values[:lag] = 0  # Fill initial values
                
                # Add to features
                features[feature_name] = lagged_values
                
                # Add correlation sign as a feature weight
                features[f"{feature_name}_corr_weight"] = np.ones_like(lagged_values) * corr
        
        # Convert to DataFrame and align with target data
        if not features:
            logger.warning(f"No cross-pair features created for {target_symbol}")
            return pd.DataFrame()
            
        features_df = pd.DataFrame(features)
        
        # Add target symbol's index
        features_df.index = target_df.index[:len(features_df)]
        
        logger.info(f"Generated {features_df.shape[1]} cross-pair features for {target_symbol}")
        return features_df


class CrossPairTransformer(nn.Module):
    """
    Transformer model for learning cross-pair relationships.
    This model uses self-attention to capture interactions between different trading pairs.
    """
    
    def __init__(self, 
                 num_pairs: int,
                 feature_dim: int,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Initialize the cross-pair transformer.
        
        Args:
            num_pairs: Number of trading pairs
            feature_dim: Dimension of features for each pair
            d_model: Hidden dimension of the transformer
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super(CrossPairTransformer, self).__init__()
        
        # Feature projection
        self.feature_projection = nn.Linear(feature_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, num_pairs, d_model))
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, num_pairs, feature_dim]
            
        Returns:
            Tensor of shape [batch_size, num_pairs, 1]
        """
        # Project features
        x = self.feature_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoder
        
        # Rearrange for transformer: [num_pairs, batch_size, d_model]
        x = x.permute(1, 0, 2)
        
        # Apply transformer
        x = self.transformer_encoder(x)
        
        # Rearrange back: [batch_size, num_pairs, d_model]
        x = x.permute(1, 0, 2)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Project to output
        x = self.output_projection(x)
        
        return x


def prepare_cross_pair_data(market_data_dict: Dict[str, pd.DataFrame], 
                          lookback: int = 30) -> Tuple[torch.Tensor, List[str]]:
    """
    Prepare data for cross-pair transformer model.
    
    Args:
        market_data_dict: Dictionary of DataFrames with price data by symbol
        lookback: Number of periods to include in features
        
    Returns:
        Tuple of (features tensor, list of symbols)
    """
    if not market_data_dict:
        logger.warning("No market data provided for cross-pair data preparation")
        return torch.tensor([]), []
    
    # Get list of symbols
    symbols = list(market_data_dict.keys())
    
    # Create feature matrices for each symbol
    features = []
    valid_symbols = []
    
    for symbol in symbols:
        df = market_data_dict[symbol]
        
        if df.empty or len(df) < lookback:
            logger.warning(f"Not enough data for {symbol}, skipping")
            continue
            
        # Extract key features
        try:
            # Price features
            returns = df['close'].pct_change().fillna(0).values
            normalized_close = (df['close'] - df['close'].mean()) / df['close'].std()
            
            # Volatility features
            if all(col in df.columns for col in ['high', 'low']):
                volatility = ((df['high'] - df['low']) / df['close']).values
            else:
                volatility = np.zeros_like(returns)
                
            # Volume features
            if 'volume' in df.columns:
                volume_change = df['volume'].pct_change().fillna(0).values
            else:
                volume_change = np.zeros_like(returns)
                
            # Create feature array
            symbol_features = np.column_stack([
                returns[-lookback:],
                normalized_close[-lookback:],
                volatility[-lookback:],
                volume_change[-lookback:]
            ])
            
            features.append(symbol_features)
            valid_symbols.append(symbol)
            
        except Exception as e:
            logger.error(f"Error preparing data for {symbol}: {str(e)}")
            continue
    
    if not features:
        logger.warning("No valid features created for cross-pair analysis")
        return torch.tensor([]), []
    
    # Convert to tensor
    features_tensor = torch.tensor(np.array(features), dtype=torch.float32)
    
    logger.info(f"Prepared cross-pair data for {len(valid_symbols)} symbols with shape {features_tensor.shape}")
    return features_tensor, valid_symbols 