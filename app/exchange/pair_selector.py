import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import talib
import xgboost as xgb
from statsmodels.tsa.stattools import coint
from scipy.stats import spearmanr
import redis
import pickle
import os
import joblib
from sklearn.preprocessing import StandardScaler

from app.config.settings import Settings
from app.exchange.manager import ExchangeManager

logger = logging.getLogger(__name__)

class TradingPairSelector:
    """
    Selects optimal USDC trading pairs based on criteria such as:
    - Liquidity (minimum volume)
    - Volatility (within min/max range)
    - Excluding stablecoin pairs
    - Respecting cooldown periods
    - ML-based scoring with XGBoost
    - Cointegration testing for portfolio diversification
    """
    
    def __init__(self, settings: Settings, exchange_manager: ExchangeManager, db_manager=None, redis_client=None):
        self.settings = settings
        self.exchange_manager = exchange_manager
        self.db_manager = db_manager
        self.redis_client = redis_client
        
        # Use configured stablecoins list from settings
        if hasattr(settings, 'binance_config') and hasattr(settings.binance_config, 'stablecoins'):
            self.stablecoins = settings.binance_config.stablecoins
        else:
            # Fallback to default list
            self.stablecoins = ["USDT", "DAI", "BUSD", "TUSD", "USDC", "UST", "USDP", "USDN", "GUSD"]
        
        self.selected_pairs = []
        
        # XGBoost model for pair ranking
        self.xgb_model = None
        self.feature_scaler = None
        self.model_trained = False
        
        # Path for ML model storage
        self.model_dir = getattr(settings, 'MODEL_DIR', 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Cointegration test threshold
        self.coint_pvalue_threshold = getattr(settings, 'COINT_PVALUE_THRESHOLD', 0.05)
        
        # Cache for market data
        self.market_data_cache = {}
        self.market_data_expiry = {}
        self.cache_ttl = 300  # 5 minutes in seconds
        
        # Use configured exchange from settings if available
        self.exchange_id = "binance"  # Default to Binance
        logger.info("Trading pair selector initialized")
        
        # Connect to Redis if not provided
        if self.redis_client is None:
            try:
                redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
                self.redis_client = redis.from_url(redis_url)
                logger.info("Connected to Redis for pair selector")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {str(e)}")
                self.redis_client = None
                
        # Load ML model if it exists
        self._load_ml_model()
    
    def _load_ml_model(self):
        """
        Load the XGBoost model and feature scaler from disk if they exist.
        """
        model_path = os.path.join(self.model_dir, 'xgb_pair_selector.json')
        scaler_path = os.path.join(self.model_dir, 'pair_feature_scaler.pkl')
        
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.xgb_model = xgb.Booster(model_file=model_path)
                self.feature_scaler = joblib.load(scaler_path)
                self.model_trained = True
                logger.info("Loaded XGBoost pair selection model from disk")
            else:
                logger.info("No existing pair selection model found, will train on first use")
        except Exception as e:
            logger.error(f"Error loading pair selection model: {str(e)}")
            self.model_trained = False
    
    def _save_ml_model(self):
        """
        Save the XGBoost model and feature scaler to disk.
        """
        if not self.model_trained or self.xgb_model is None:
            return
            
        try:
            model_path = os.path.join(self.model_dir, 'xgb_pair_selector.json')
            scaler_path = os.path.join(self.model_dir, 'pair_feature_scaler.pkl')
            
            self.xgb_model.save_model(model_path)
            joblib.dump(self.feature_scaler, scaler_path)
            logger.info("Saved XGBoost pair selection model to disk")
        except Exception as e:
            logger.error(f"Error saving pair selection model: {str(e)}")
    
    async def select_pairs(self) -> List[str]:
        """
        Main method to select trading pairs based on configured criteria.
        Returns a list of selected trading pairs.
        """
        try:
            # Step 1: Get all available USDC pairs from Binance
            all_usdc_pairs = await self.exchange_manager.fetch_usdc_trading_pairs(self.exchange_id)
            logger.info(f"Found {len(all_usdc_pairs)} USDC pairs on {self.exchange_id}")
            
            if not all_usdc_pairs:
                logger.warning(f"No USDC trading pairs found on {self.exchange_id}")
                return []
            
            # Step 2: Filter out stablecoin pairs
            filtered_pairs = self._filter_stablecoin_pairs(all_usdc_pairs)
            logger.info(f"After stablecoin filtering: {len(filtered_pairs)} pairs remaining")
            
            # Step 3: Filter by activity and cooldown
            active_pairs = await self._filter_by_activity_and_cooldown(filtered_pairs)
            logger.info(f"After activity & cooldown filtering: {len(active_pairs)} pairs remaining")
            
            # Step 4: Get market data for active pairs
            market_data = await self._get_market_data_for_pairs(active_pairs)
            
            # Step A: Use rule-based ranking as fallback or first stage
            rule_ranked_pairs = await self._rule_based_ranking(active_pairs, market_data)
            
            # Step B: Use ML-based ranking if possible (requires enough data)
            if len(rule_ranked_pairs) >= 5:  # Need enough pairs for ML
                # This will also train the model if needed
                ml_ranked_pairs = await self._ml_based_ranking(rule_ranked_pairs, market_data)
                ranked_pairs = ml_ranked_pairs
                logger.info("Used ML-based ranking for pair selection")
            else:
                ranked_pairs = rule_ranked_pairs
                logger.info("Used rule-based ranking (not enough data for ML)")
            
            # Step 5: Apply cointegration testing for diversification
            diversified_pairs = await self._diversify_portfolio(ranked_pairs, market_data)
            
            # Step 6: Select top pairs (limited by max_positions setting)
            # Get max_positions from settings or use default
            try:
                # Check for MAX_POSITIONS setting
                if hasattr(self.settings, 'MAX_POSITIONS'):
                    max_pairs = self.settings.MAX_POSITIONS
                    logger.info(f"Using MAX_POSITIONS from settings: {max_pairs}")
                else:
                    max_pairs = 5  # Default to 5 pairs
                    logger.warning(f"MAX_POSITIONS not found in settings, using default: {max_pairs}")
            except AttributeError:
                max_pairs = 5  # Default to 5 pairs
                logger.warning(f"MAX_POSITIONS not found in settings, using default: {max_pairs}")
                
            # Make sure max_pairs is at least 1
            if max_pairs <= 0:
                max_pairs = 5
                logger.warning(f"Invalid MAX_POSITIONS value ({max_pairs}), using default: 5")
            
            max_pairs = min(max_pairs, len(diversified_pairs))
            self.selected_pairs = diversified_pairs[:max_pairs]
            
            # Log selected pairs
            logger.info(f"Selected {len(self.selected_pairs)} trading pairs (max allowed: {max_pairs})")
            logger.info(f"Trading pairs selected: {', '.join(self.selected_pairs)}")
            
            return self.selected_pairs
        
        except Exception as e:
            logger.exception(f"Error selecting trading pairs: {str(e)}")
            return []
    
    def _filter_stablecoin_pairs(self, pairs: List[str]) -> List[str]:
        """
        Filter out pairs where the base asset is a stablecoin.
        Ensures all pairs are properly formatted with / separator.
        """
        filtered_pairs = []
        
        for pair in pairs:
            # Ensure the pair has the correct format with / separator
            if '/' not in pair:
                # Try to detect the quote currency and add separator
                for stablecoin in self.stablecoins:
                    if pair.endswith(stablecoin):
                        base = pair[:-len(stablecoin)]
                        pair = f"{base}/{stablecoin}"
                        logger.debug(f"Reformatted pair to {pair}")
                        break
                # If still no separator, skip this pair
                if '/' not in pair:
                    logger.debug(f"Skipping pair with invalid format: {pair}")
                    continue
                
            # Extract base asset (assuming format like "BTC/USDC")
            base_asset = pair.split('/')[0]
            
            # Check if base is a stablecoin
            if base_asset in self.stablecoins:
                logger.debug(f"Filtered out stablecoin pair: {pair}")
                continue
            
            filtered_pairs.append(pair)
        
        return filtered_pairs
    
    async def _filter_by_activity_and_cooldown(self, pairs: List[str]) -> List[str]:
        """
        Filter pairs by minimum volume and exclude those in cooldown period.
        """
        active_pairs = []
        tickers = await self.exchange_manager.fetch_tickers(self.exchange_id)
        
        if not tickers or self.exchange_id not in tickers:
            logger.error(f"Failed to fetch tickers from {self.exchange_id}")
            return []
        
        tickers = tickers[self.exchange_id]
        
        # Get minimum volume threshold, use default if not in settings
        try:
            min_volume_threshold = self.settings.min_daily_volume_usd
        except AttributeError:
            min_volume_threshold = 1_000_000.0  # Default to 1M USD
            logger.warning(f"min_daily_volume_usd not found in settings, using default: {min_volume_threshold}")
        
        for pair in pairs:
            # Skip if pair is in cooldown
            if await self._is_in_cooldown(pair):
                logger.debug(f"{pair} is in cooldown period, skipping")
                continue
            
            # Check volume
            if pair in tickers:
                volume_usd = tickers[pair].get('quoteVolume', 0)
                if volume_usd >= min_volume_threshold:
                    active_pairs.append(pair)
                else:
                    logger.debug(f"{pair} volume ({volume_usd:.2f} USDC) below minimum threshold")
            else:
                logger.debug(f"{pair} not found in tickers data")
        
        return active_pairs
    
    async def _is_in_cooldown(self, pair: str) -> bool:
        """
        Check if a pair is in cooldown period using either Redis or DB manager.
        """
        # First try Redis
        if self.redis_client:
            try:
                cooldown_key = f"trading_bot:symbol_cooldown:{pair}"
                return bool(self.redis_client.exists(cooldown_key))
            except Exception as e:
                logger.warning(f"Redis cooldown check failed: {str(e)}")
        
        # Fallback to DB manager
        if self.db_manager and hasattr(self.db_manager, 'has_cooldown'):
            return await self.db_manager.has_cooldown(pair)
            
        return False
    
    async def _rule_based_ranking(self, pairs: List[str], market_data: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Rank pairs based on rule-based metrics (volatility, volume, technical indicators).
        Returns ordered list of pairs.
        """
        if not pairs:
            return []
        
        # Get volatility range from settings or use defaults
        try:
            min_volatility = self.settings.MIN_VOLATILITY
        except AttributeError:
            min_volatility = 0.05  # Default 5%
            logger.warning(f"MIN_VOLATILITY not found in settings, using default: {min_volatility}")
            
        try:
            max_volatility = self.settings.MAX_VOLATILITY
        except AttributeError:
            max_volatility = 0.25  # Default 25%
            logger.warning(f"MAX_VOLATILITY not found in settings, using default: {max_volatility}")
        
        # Collect metrics for each pair
        pair_metrics = []
        
        for pair in pairs:
            try:
                # Skip if no market data available
                if pair not in market_data or market_data[pair].empty:
                    continue
                
                ohlcv = market_data[pair]
                
                # Calculate metrics
                volatility = self._calculate_volatility(ohlcv)
                
                # Skip if volatility is outside acceptable range
                if (volatility < min_volatility or 
                    volatility > max_volatility):
                    logger.debug(f"{pair} volatility ({volatility:.2f}) outside range")
                    continue
                
                # Get volume and calculate TA metrics
                volume = ohlcv['volume'].sum()
                ta_score = self._calculate_ta_score(ohlcv)
                
                # Try to get minimum daily volume or use default
                try:
                    min_daily_volume = self.settings.min_daily_volume_usd
                except AttributeError:
                    min_daily_volume = 1_000_000.0  # Default to 1M USD
                
                # Calculate a scoring metric (40% volume, 40% volatility, 20% TA)
                normalized_volume = min(1.0, volume / (2 * min_daily_volume))
                normalized_volatility = volatility / max_volatility
                
                score = (0.4 * normalized_volume + 
                         0.4 * normalized_volatility + 
                         0.2 * ta_score)
                
                pair_metrics.append({
                    'pair': pair,
                    'volatility': volatility,
                    'volume': volume,
                    'ta_score': ta_score,
                    'score': score
                })
                
            except Exception as e:
                logger.error(f"Error calculating metrics for {pair}: {str(e)}")
        
        # Rank pairs by score (descending)
        ranked_metrics = sorted(pair_metrics, key=lambda x: x['score'], reverse=True)
        
        # Extract just the pair names in ranked order
        ranked_pairs = [metric['pair'] for metric in ranked_metrics]
        
        return ranked_pairs
    
    async def _ml_based_ranking(self, pairs: List[str], market_data: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Rank pairs using XGBoost model based on historical performance.
        If model isn't trained yet, it will train and then predict.
        """
        # Prepare feature dataframe
        features_df = await self._prepare_ml_features(pairs, market_data)
        
        if features_df.empty:
            logger.warning("Could not prepare features for ML ranking")
            return pairs  # Return rule-based ranking
        
        # Check if we need to train the model
        if not self.model_trained or self.xgb_model is None:
            logger.info("Training XGBoost model for pair selection")
            await self._train_xgb_model(features_df)
        
        # Make predictions with trained model
        try:
            # Ensure we have a feature scaler
            if self.feature_scaler is None:
                logger.warning("No feature scaler available for ML ranking")
                return pairs
                
            # Create X matrix without pair column
            X = features_df.drop(['pair'], axis=1)
            
            # Scale features
            X_scaled = self.feature_scaler.transform(X)
            
            # Convert to DMatrix for prediction
            dmatrix = xgb.DMatrix(X_scaled)
            
            # Predict scores
            scores = self.xgb_model.predict(dmatrix)
            
            # Add scores back to dataframe
            features_df['ml_score'] = scores
            
            # Sort by score
            ranked_df = features_df.sort_values('ml_score', ascending=False)
            
            # Get ranked pairs
            ranked_pairs = ranked_df['pair'].tolist()
            
            logger.info(f"ML model ranked {len(ranked_pairs)} pairs")
            return ranked_pairs
            
        except Exception as e:
            logger.error(f"Error in ML-based ranking: {str(e)}")
            return pairs  # Fall back to rule-based ranking
    
    async def _train_xgb_model(self, features_df: pd.DataFrame):
        """
        Train XGBoost model to rank trading pairs based on features.
        """
        try:
            # We need to create a target variable since we're doing supervised learning
            # As a proxy, use recent price performance over a lookback period
            
            # Get historical performance data for pairs in features_df
            pairs = features_df['pair'].unique().tolist()
            
            # Create target variable: percentage price change over next day
            target_data = {}
            
            for pair in pairs:
                # Get additional data for future price movements
                future_data = await self._get_future_performance(pair, days=7)
                if future_data is not None:
                    target_data[pair] = future_data
            
            if not target_data:
                logger.warning("No target data available for ML training")
                return
            
            # Join target data to features
            features_df['target'] = features_df['pair'].map(target_data)
            
            # Remove rows with missing target
            features_df = features_df.dropna(subset=['target'])
            
            if len(features_df) < 5:
                logger.warning("Not enough data for ML training")
                return
                
            # Split features and target
            X = features_df.drop(['pair', 'target'], axis=1)
            y = features_df['target']
            
            # Scale features
            self.feature_scaler = StandardScaler()
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Train XGBoost model
            dtrain = xgb.DMatrix(X_scaled, label=y)
            
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'eta': 0.1,
                'max_depth': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
            
            self.xgb_model = xgb.train(
                params,
                dtrain,
                num_boost_round=50
            )
            
            self.model_trained = True
            
            # Save the model
            self._save_ml_model()
            
            logger.info("XGBoost pair selection model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training XGBoost model: {str(e)}")
            self.model_trained = False
    
    async def _get_future_performance(self, pair: str, days: int = 7) -> Optional[float]:
        """
        Get future performance of a pair over a specified number of days.
        This is used as the target variable for ML training.
        """
        try:
            # Get historical data
            ohlcv = await self.exchange_manager.fetch_ohlcv(
                pair, timeframe='1d', limit=days+1, exchange_id=self.exchange_id
            )
            
            if ohlcv.empty or len(ohlcv) < 2:
                return None
                
            # Calculate return over the period
            start_price = ohlcv['close'].iloc[0]
            end_price = ohlcv['close'].iloc[-1]
            
            return (end_price / start_price) - 1
            
        except Exception as e:
            logger.debug(f"Could not get future performance for {pair}: {str(e)}")
            return None
    
    async def _prepare_ml_features(self, pairs: List[str], market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Prepare features for the ML model from market data.
        """
        features = []
        
        for pair in pairs:
            if pair not in market_data or market_data[pair].empty:
                continue
                
            ohlcv = market_data[pair]
            
            try:
                # Calculate all features
                row = {'pair': pair}
                
                # 1. Volatility features
                row['volatility'] = self._calculate_volatility(ohlcv)
                row['atr_pct'] = self._calculate_atr_percentage(ohlcv)
                
                # 2. Volume features
                row['volume'] = ohlcv['volume'].sum()
                row['volume_mean'] = ohlcv['volume'].mean()
                row['volume_std'] = ohlcv['volume'].std()
                
                # 3. Momentum features
                row['rsi'] = self._calculate_rsi(ohlcv)
                row['macd'] = self._calculate_macd(ohlcv)
                
                # 4. Trend features
                row['adx'] = self._calculate_adx(ohlcv)
                row['trend_strength'] = self._calculate_trend_strength(ohlcv)
                
                # 5. Price pattern features
                row['distance_from_ma'] = self._calculate_ma_distance(ohlcv)
                row['bb_position'] = self._calculate_bbands_position(ohlcv)
                
                features.append(row)
                
            except Exception as e:
                logger.debug(f"Error calculating features for {pair}: {str(e)}")
                continue
        
        if not features:
            return pd.DataFrame()
            
        return pd.DataFrame(features)
    
    async def _diversify_portfolio(self, ranked_pairs: List[str], market_data: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Apply cointegration and correlation tests to ensure portfolio diversification.
        """
        if len(ranked_pairs) <= 1:
            return ranked_pairs
        
        # Get max positions setting or use default
        try:
            max_positions = self.settings.MAX_POSITIONS
        except AttributeError:
            max_positions = 5  # Default to 5 positions
            logger.warning(f"MAX_POSITIONS not found in settings, using default: {max_positions}")
            
        # Make sure max_positions is at least 1
        if max_positions <= 0:
            max_positions = 5
            logger.warning(f"Invalid MAX_POSITIONS value ({max_positions}), using default: 5")
            
        # Start with the top-ranked pair
        diversified = [ranked_pairs[0]]
        
        # Test each remaining pair against already selected pairs
        for pair in ranked_pairs[1:]:
            if len(diversified) >= max_positions:
                break
                
            # Skip if pair has no market data
            if pair not in market_data or market_data[pair].empty:
                continue
                
            # Check if this pair is sufficiently different from already selected pairs
            is_diversifying = await self._is_diversifying(pair, diversified, market_data)
            
            if is_diversifying:
                diversified.append(pair)
                logger.debug(f"Added {pair} to diversified portfolio")
            else:
                logger.debug(f"Skipped {pair} - too correlated with existing selections")
        
        # If we don't have enough pairs, add more from ranked list regardless of correlation
        if len(diversified) < min(max_positions, len(ranked_pairs)):
            remaining = [p for p in ranked_pairs if p not in diversified]
            needed = min(max_positions, len(ranked_pairs)) - len(diversified)
            diversified.extend(remaining[:needed])
        
        return diversified
    
    async def _is_diversifying(self, candidate: str, selected: List[str], market_data: Dict[str, pd.DataFrame]) -> bool:
        """
        Check if a candidate pair diversifies the selected portfolio.
        """
        candidate_data = market_data.get(candidate)
        if candidate_data is None or candidate_data.empty:
            return False
            
        candidate_returns = candidate_data['close'].pct_change().dropna()
        
        for selected_pair in selected:
            selected_data = market_data.get(selected_pair)
            if selected_data is None or selected_data.empty:
                continue
                
            selected_returns = selected_data['close'].pct_change().dropna()
            
            # Ensure we have matching length series
            min_len = min(len(candidate_returns), len(selected_returns))
            if min_len < 10:  # Need at least 10 data points
                continue
                
            c_returns = candidate_returns.iloc[-min_len:].values
            s_returns = selected_returns.iloc[-min_len:].values
            
            # Calculate correlation
            correlation = spearmanr(c_returns, s_returns)[0]
            if not np.isnan(correlation) and abs(correlation) > 0.7:
                logger.debug(f"{candidate} has high correlation ({correlation:.2f}) with {selected_pair}")
                return False
            
            # Test for cointegration if enough data
            if min_len >= 30:
                c_prices = candidate_data['close'].iloc[-min_len:].values
                s_prices = selected_data['close'].iloc[-min_len:].values
                
                _, pvalue, _ = coint(c_prices, s_prices)
                if not np.isnan(pvalue) and pvalue < self.coint_pvalue_threshold:
                    logger.debug(f"{candidate} is cointegrated (p={pvalue:.4f}) with {selected_pair}")
                    return False
        
        # If we got here, the pair is sufficiently different from all selected pairs
        return True
    
    async def _get_market_data_for_pairs(self, pairs: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple pairs, using cache when available.
        """
        market_data = {}
        fetch_pairs = []
        now = datetime.now()
        
        # Check which pairs need fresh data
        for pair in pairs:
            if pair in self.market_data_cache and pair in self.market_data_expiry:
                if now < self.market_data_expiry[pair]:
                    market_data[pair] = self.market_data_cache[pair]
                    continue
            
            fetch_pairs.append(pair)
        
        # Fetch fresh data for remaining pairs
        for pair in fetch_pairs:
            try:
                # Get OHLCV data
                ohlcv = await self.exchange_manager.fetch_ohlcv(
                    pair, timeframe='1h', limit=24, exchange_id=self.exchange_id
                )
                
                if ohlcv.empty:
                    continue
                
                # Calculate additional technical indicators
                ohlcv = self._add_technical_indicators(ohlcv)
                
                # Store in result and cache
                market_data[pair] = ohlcv
                self.market_data_cache[pair] = ohlcv
                self.market_data_expiry[pair] = now + timedelta(seconds=self.cache_ttl)
                
            except Exception as e:
                logger.error(f"Error fetching market data for {pair}: {str(e)}")
        
        return market_data
    
    def _calculate_volatility(self, ohlcv: pd.DataFrame) -> float:
        """
        Calculate the volatility of a trading pair based on OHLCV data.
        Returns volatility as a percentage.
        """
        # Calculate returns
        if 'close' not in ohlcv.columns or len(ohlcv) < 2:
            return 0.0
        
        # Calculate hourly returns
        returns = ohlcv['close'].pct_change().dropna()
        
        # Calculate annualized volatility (hourly to annual)
        volatility = returns.std() * (24 * 365) ** 0.5 * 100
        
        return volatility
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to OHLCV data using TA-Lib.
        """
        if df.empty or len(df) < 14:
            return df
            
        try:
            # Make a copy to avoid modifying the original
            df = df.copy()
            
            # Add RSI
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            
            # Add MACD
            macd, macdsignal, macdhist = talib.MACD(
                df['close'], fastperiod=12, slowperiod=26, signalperiod=9
            )
            df['macd'] = macd
            df['macdsignal'] = macdsignal
            df['macdhist'] = macdhist
            
            # Add Bollinger Bands
            upperband, middleband, lowerband = talib.BBANDS(
                df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
            df['bb_upper'] = upperband
            df['bb_middle'] = middleband
            df['bb_lower'] = lowerband
            
            # Add ADX
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Add ATR
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Add Moving Averages
            df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
            df['ema_9'] = talib.EMA(df['close'], timeperiod=9)
            
            return df
            
        except Exception as e:
            logger.warning(f"Error adding technical indicators: {str(e)}")
            return df
    
    def _calculate_ta_score(self, df: pd.DataFrame) -> float:
        """
        Calculate a composite technical analysis score (0-1).
        """
        if df.empty or len(df) < 20:
            return 0.5  # Neutral score
            
        score = 0.5  # Start with neutral
        count = 0
        
        try:
            latest = df.iloc[-1]
            
            # RSI - favor middle values (not overbought/oversold)
            if 'rsi' in latest and not np.isnan(latest['rsi']):
                rsi = latest['rsi']
                # Highest score at RSI 50, decreasing toward extremes
                rsi_score = 1.0 - (abs(rsi - 50) / 50)
                score += rsi_score
                count += 1
            
            # MACD - favor crosses or strong momentum
            if all(x in latest for x in ['macd', 'macdsignal']) and not np.isnan(latest['macd']):
                # MACD crossing or about to cross signal line is bullish
                macd_diff = abs(latest['macd'] - latest['macdsignal'])
                if macd_diff < 0.01 * latest['close']:  # Close to crossing
                    macd_score = 0.8
                else:
                    macd_score = 0.5
                score += macd_score
                count += 1
            
            # Bollinger Bands - favor middle position
            if all(x in latest for x in ['bb_upper', 'bb_lower', 'close']) and not np.isnan(latest['bb_upper']):
                bb_range = latest['bb_upper'] - latest['bb_lower']
                if bb_range > 0:
                    # Position within bands (0=lower, 1=upper)
                    bb_pos = (latest['close'] - latest['bb_lower']) / bb_range
                    # Score highest at middle of bands (0.5)
                    bb_score = 1.0 - abs(bb_pos - 0.5) * 2
                    score += bb_score
                    count += 1
            
            # Average the scores
            if count > 0:
                return score / count
            return 0.5
            
        except Exception as e:
            logger.debug(f"Error calculating TA score: {str(e)}")
            return 0.5
    
    def _calculate_atr_percentage(self, df: pd.DataFrame) -> float:
        """
        Calculate ATR as percentage of price.
        """
        if 'atr' not in df.columns or 'close' not in df.columns or df.empty:
            return 0.0
            
        try:
            latest = df.iloc[-1]
            if latest['close'] > 0:
                return (latest['atr'] / latest['close']) * 100
            return 0.0
        except:
            return 0.0
    
    def _calculate_rsi(self, df: pd.DataFrame) -> float:
        """
        Get the latest RSI value.
        """
        if 'rsi' not in df.columns or df.empty:
            return 50.0  # Neutral
            
        latest = df.iloc[-1]
        return latest['rsi'] if not np.isnan(latest['rsi']) else 50.0
    
    def _calculate_macd(self, df: pd.DataFrame) -> float:
        """
        Get the latest MACD histogram value, normalized.
        """
        if 'macdhist' not in df.columns or 'close' not in df.columns or df.empty:
            return 0.0
            
        latest = df.iloc[-1]
        if latest['close'] > 0:
            return (latest['macdhist'] / latest['close']) * 100
        return 0.0
    
    def _calculate_adx(self, df: pd.DataFrame) -> float:
        """
        Get the latest ADX value.
        """
        if 'adx' not in df.columns or df.empty:
            return 0.0
            
        latest = df.iloc[-1]
        return latest['adx'] if not np.isnan(latest['adx']) else 0.0
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """
        Calculate trend strength based on price movement.
        """
        if 'close' not in df.columns or len(df) < 10:
            return 0.0
            
        # Calculate linear regression slope
        try:
            y = df['close'].values
            x = np.arange(len(y))
            
            slope, _, r_value, _, _ = np.polyfit(x, y, 1, full=True)[0:5]
            
            # Normalize slope by price level
            if df['close'].mean() > 0:
                norm_slope = slope / df['close'].mean()
                
                # Combine direction and strength (r-squared)
                return norm_slope * (r_value ** 2) * 100
            return 0.0
        except:
            return 0.0
    
    def _calculate_ma_distance(self, df: pd.DataFrame) -> float:
        """
        Calculate distance from price to moving average.
        """
        if 'close' not in df.columns or 'sma_20' not in df.columns or df.empty:
            return 0.0
            
        latest = df.iloc[-1]
        if latest['sma_20'] > 0:
            return ((latest['close'] / latest['sma_20']) - 1) * 100
        return 0.0
    
    def _calculate_bbands_position(self, df: pd.DataFrame) -> float:
        """
        Calculate position within Bollinger Bands (0=lower, 1=upper).
        """
        if not all(x in df.columns for x in ['close', 'bb_upper', 'bb_lower']) or df.empty:
            return 0.5
            
        latest = df.iloc[-1]
        band_range = latest['bb_upper'] - latest['bb_lower']
        
        if band_range > 0:
            position = (latest['close'] - latest['bb_lower']) / band_range
            return position
        return 0.5
    
    async def set_pair_cooldown(self, pair: str, hours: int = 24) -> bool:
        """
        Set a cooldown period for a pair after a trade.
        """
        if not pair:
            return False
            
        try:
            # Try Redis first if available
            if self.redis_client:
                cooldown_key = f"trading_bot:symbol_cooldown:{pair}"
                seconds = hours * 3600
                self.redis_client.setex(cooldown_key, seconds, 1)
                logger.info(f"Set cooldown for {pair} for {hours} hours in Redis")
                return True
            
            # Fallback to DB manager
            if self.db_manager and hasattr(self.db_manager, 'set_cooldown'):
                result = await self.db_manager.set_cooldown(pair, hours)
                logger.info(f"Set cooldown for {pair} for {hours} hours in database")
                return result
                
            logger.warning(f"Could not set cooldown for {pair} - no Redis or DB manager available")
            return False
            
        except Exception as e:
            logger.error(f"Error setting cooldown for {pair}: {str(e)}")
            return False
    
    async def get_pair_details(self, pair: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific trading pair.
        """
        try:
            # Get market data (will use cache if available)
            market_data = await self._get_market_data_for_pairs([pair])
            
            if pair not in market_data or market_data[pair].empty:
                return {
                    'pair': pair,
                    'status': 'no_data',
                    'error': 'No OHLCV data available'
                }
            
            ohlcv = market_data[pair]
            
            # Calculate key metrics
            volatility = self._calculate_volatility(ohlcv)
            volume = ohlcv['volume'].sum()
            current_price = ohlcv['close'].iloc[-1]
            
            # Get additional TA metrics
            latest = ohlcv.iloc[-1]
            ta_metrics = {}
            
            for indicator in ['rsi', 'macd', 'adx', 'atr']:
                if indicator in latest:
                    ta_metrics[indicator] = latest[indicator]
            
            # Check cooldown status
            in_cooldown = await self._is_in_cooldown(pair)
            
            return {
                'pair': pair,
                'status': 'active',
                'price': current_price,
                'volume_24h': volume,
                'volatility': volatility,
                'ta_metrics': ta_metrics,
                'in_cooldown': in_cooldown
            }
        
        except Exception as e:
            logger.error(f"Error getting details for {pair}: {str(e)}")
            return {
                'pair': pair,
                'status': 'error',
                'error': str(e)
            }

    def set_max_pairs(self, max_pairs: int) -> None:
        """
        Set the maximum number of trading pairs to use.
        
        Args:
            max_pairs: Maximum number of pairs to select
        """
        if max_pairs < 1:
            logger.warning(f"Invalid max_pairs value ({max_pairs}), using 1 instead")
            max_pairs = 1
        elif max_pairs > 30:
            logger.warning(f"Invalid max_pairs value ({max_pairs}), using 30 instead")
            max_pairs = 30
            
        try:
            # Update settings object
            self.settings.MAX_POSITIONS = max_pairs
            logger.info(f"Updated max trading pairs to {max_pairs}")
            
            # If we already have selected pairs, trim them if needed
            if hasattr(self, 'selected_pairs') and self.selected_pairs:
                if len(self.selected_pairs) > max_pairs:
                    self.selected_pairs = self.selected_pairs[:max_pairs]
                    logger.info(f"Trimmed selected pairs to {max_pairs}")
        except Exception as e:
            logger.error(f"Error setting max pairs: {str(e)}") 