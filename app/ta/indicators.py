import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
import pandas_ta as ta
from numba import jit, cuda
import talib
from concurrent.futures import ThreadPoolExecutor
import math

# Fix for pandas-ta compatibility with newer NumPy versions
# Some versions of pandas-ta try to import NaN directly from numpy
if not hasattr(np, 'NaN'):
    np.NaN = float('nan')

logger = logging.getLogger(__name__)

class TechnicalAnalysisModule:
    """
    Calculates technical indicators for market data across multiple timeframes.
    """
    
    def __init__(self, settings=None):
        self.settings = settings
        self.timeframes = settings.default_timeframes if settings else ["1m", "5m", "15m", "1h"]
        self.use_gpu = getattr(settings, 'USE_GPU_ACCELERATION', False)
        self.use_talib = getattr(settings, 'USE_TALIB', True)
        self.max_workers = getattr(settings, 'TA_MAX_WORKERS', 4)
        logger.info(f"Technical Analysis module initialized (GPU: {self.use_gpu}, TA-Lib: {self.use_talib})")
    
    async def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for a dataframe of OHLCV data.
        
        Args:
            data: Pandas DataFrame with OHLCV data
                 (must have columns: 'open', 'high', 'low', 'close', 'volume')
        
        Returns:
            DataFrame with original data plus indicator columns
        """
        if data.empty:
            logger.warning("Empty dataframe provided, cannot calculate indicators")
            return data
        
        if not all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            logger.error("Data missing required OHLCV columns")
            return data
        
        try:
            # Create a copy to avoid modifying the original dataframe
            df = data.copy()
            
            # Calculate trend indicators
            self._add_moving_averages(df)
            self._add_macd(df)
            self._add_rsi(df)
            self._add_stochastic(df)
            self._add_bollinger_bands(df)
            self._add_adx(df)
            
            # Calculate volume indicators
            self._add_obv(df)
            self._add_volume_profile(df)
            
            # Calculate volatility indicators
            self._add_atr(df)
            
            # Calculate additional crypto-specific indicators
            self._add_mayer_multiple(df)
            self._add_crypto_patterns(df)
            
            return df
        
        except Exception as e:
            logger.exception(f"Error calculating indicators: {str(e)}")
            return data
    
    def _add_moving_averages(self, df: pd.DataFrame) -> None:
        """Add various moving averages to the dataframe."""
        # Use TA-Lib if enabled, otherwise use pandas-ta
        if self.use_talib:
            df['sma_9'] = talib.SMA(df['close'], timeperiod=9)
            df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
            df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
            df['sma_200'] = talib.SMA(df['close'], timeperiod=200)
            
            # Exponential Moving Averages
            df['ema_9'] = talib.EMA(df['close'], timeperiod=9)
            df['ema_21'] = talib.EMA(df['close'], timeperiod=21)
            df['ema_55'] = talib.EMA(df['close'], timeperiod=55)
        else:
            # Simple Moving Averages
            df['sma_9'] = ta.sma(df['close'], length=9)
            df['sma_20'] = ta.sma(df['close'], length=20)
            df['sma_50'] = ta.sma(df['close'], length=50)
            df['sma_200'] = ta.sma(df['close'], length=200)
            
            # Exponential Moving Averages
            df['ema_9'] = ta.ema(df['close'], length=9)
            df['ema_21'] = ta.ema(df['close'], length=21)
            df['ema_55'] = ta.ema(df['close'], length=55)
        
        # Hull Moving Average - only in pandas-ta
        df['hma_21'] = ta.hma(df['close'], length=21)
    
    def _add_macd(self, df: pd.DataFrame) -> None:
        """Add MACD indicator to the dataframe."""
        if self.use_talib:
            macd, signal, hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            df['macd'] = macd
            df['macd_signal'] = signal
            df['macd_hist'] = hist
        else:
            # MACD (12, 26, 9)
            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
            df['macd_hist'] = macd['MACDh_12_26_9']
    
    @jit(nopython=True, parallel=True)
    def _numba_rsi(self, close_prices, length=14):
        """Accelerated RSI calculation using Numba."""
        deltas = np.diff(close_prices)
        seed = deltas[:length+1]
        up = seed[seed >= 0].sum()/length
        down = -seed[seed < 0].sum()/length
        rs = up/down if down != 0 else np.inf
        rsi = np.zeros_like(close_prices)
        rsi[:length] = 100. - 100./(1. + rs)
        
        for i in range(length, len(close_prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0
            else:
                upval = 0
                downval = -delta

            up = (up * (length - 1) + upval) / length
            down = (down * (length - 1) + downval) / length
            
            rs = up/down if down != 0 else np.inf
            rsi[i] = 100. - 100./(1. + rs)
            
        return rsi
    
    def _add_rsi(self, df: pd.DataFrame) -> None:
        """Add RSI indicator to the dataframe."""
        if self.use_gpu and len(df) > 1000:  # Only use GPU for larger datasets
            try:
                # Use Numba accelerated function
                close_array = df['close'].values
                df['rsi_14'] = self._numba_rsi(close_array, length=14)
                df['rsi_7'] = self._numba_rsi(close_array, length=7)
                df['rsi_21'] = self._numba_rsi(close_array, length=21)
                logger.debug("Used GPU-accelerated RSI calculation")
            except Exception as e:
                logger.warning(f"GPU acceleration failed, falling back to standard method: {e}")
                self._add_rsi_standard(df)
        else:
            self._add_rsi_standard(df)
    
    def _add_rsi_standard(self, df: pd.DataFrame) -> None:
        """Standard RSI calculation method."""
        if self.use_talib:
            df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
            df['rsi_7'] = talib.RSI(df['close'], timeperiod=7)
            df['rsi_21'] = talib.RSI(df['close'], timeperiod=21)
        else:
            df['rsi_14'] = ta.rsi(df['close'], length=14)
            df['rsi_7'] = ta.rsi(df['close'], length=7)
            df['rsi_21'] = ta.rsi(df['close'], length=21)
    
    def _add_stochastic(self, df: pd.DataFrame) -> None:
        """Add Stochastic indicator to the dataframe."""
        if self.use_talib:
            k, d = talib.STOCH(df['high'], df['low'], df['close'], 
                               fastk_period=14, slowk_period=3, slowk_matype=0, 
                               slowd_period=3, slowd_matype=0)
            df['stoch_k'] = k
            df['stoch_d'] = d
        else:
            # Stochastic (14, 3, 3)
            stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
            df['stoch_k'] = stoch['STOCHk_14_3_3']
            df['stoch_d'] = stoch['STOCHd_14_3_3']
    
    def _add_bollinger_bands(self, df: pd.DataFrame) -> None:
        """Add Bollinger Bands to the dataframe."""
        if self.use_talib:
            upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20, 
                                                nbdevup=2, nbdevdn=2, matype=0)
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
        else:
            # Bollinger Bands (20, 2)
            bbands = ta.bbands(df['close'], length=20, std=2)
            df['bb_upper'] = bbands['BBU_20_2.0']
            df['bb_middle'] = bbands['BBM_20_2.0']
            df['bb_lower'] = bbands['BBL_20_2.0']
        
        # Calculate percent B - position within the bands (0 to 1)
        df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    def _add_adx(self, df: pd.DataFrame) -> None:
        """Add ADX (Average Directional Index) to the dataframe."""
        if self.use_talib:
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            df['di_plus'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
            df['di_minus'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        else:
            # ADX (14)
            adx = ta.adx(df['high'], df['low'], df['close'], length=14)
            df['adx'] = adx['ADX_14']
            df['di_plus'] = adx['DMP_14']
            df['di_minus'] = adx['DMN_14']
    
    def _add_obv(self, df: pd.DataFrame) -> None:
        """Add On-Balance Volume to the dataframe."""
        if self.use_talib:
            df['obv'] = talib.OBV(df['close'], df['volume'])
        else:
            # On-Balance Volume
            df['obv'] = ta.obv(df['close'], df['volume'])
        
        # Calculate OBV moving average for signal
        df['obv_ema'] = ta.ema(df['obv'], length=21)
    
    def _add_volume_profile(self, df: pd.DataFrame) -> None:
        """Add simple volume analysis indicators."""
        # Volume moving average
        df['volume_sma'] = ta.sma(df['volume'], length=20)
        
        # Relative volume (current volume / average volume)
        df['relative_volume'] = df['volume'] / df['volume_sma']
        
        # Money Flow Index (volume-weighted RSI)
        if self.use_talib:
            df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        else:
            df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
    
    def _add_atr(self, df: pd.DataFrame) -> None:
        """Add Average True Range to the dataframe."""
        if self.use_talib:
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        else:
            # ATR (14)
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # Normalized ATR (ATR / Close price)
        df['natr'] = (df['atr'] / df['close']) * 100
    
    def _add_mayer_multiple(self, df: pd.DataFrame) -> None:
        """
        Add Mayer Multiple (price / 200-day MA) - popular in crypto.
        Values > 2.4 have historically indicated bubble territory.
        """
        # Requires 200 days of data for meaningful calculation
        if len(df) >= 200:
            df['mayer_multiple'] = df['close'] / df['sma_200']
    
    def _add_crypto_patterns(self, df: pd.DataFrame) -> None:
        """Add crypto-specific patterns and indicators."""
        # Add Ichimoku Cloud (popular in crypto trading)
        if len(df) >= 52:  # Need enough data for the longer periods
            try:
                # Instead of using ta.ichimoku which is causing the error,
                # implement our own Ichimoku calculation
                self._calculate_ichimoku(df, tenkan=9, kijun=26, senkou=52)
            except Exception as e:
                logger.warning(f"Could not calculate Ichimoku: {e}")
        
        # VWAP (Volume Weighted Average Price) - commonly used in crypto
        if 'volume' in df.columns:
            try:
                df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
            except Exception as e:
                logger.warning(f"Could not calculate VWAP: {e}")
        
        # Bitcoin dominance effect (if available)
        if 'btc_dominance' in df.columns:
            df['btc_dom_impact'] = df['btc_dominance'] / 100
        
        # Add accumulation/distribution line
        try:
            df['acc_dist'] = ta.ad(df['high'], df['low'], df['close'], df['volume'])
        except Exception as e:
            logger.warning(f"Could not calculate A/D Line: {e}")
    
    def _calculate_ichimoku(self, df, tenkan=9, kijun=26, senkou=52):
        """
        Calculate Ichimoku Cloud components directly without using pandas-ta.
        
        Args:
            df: DataFrame with OHLCV data
            tenkan: Tenkan-sen (Conversion Line) period
            kijun: Kijun-sen (Base Line) period
            senkou: Senkou Span B period
        """
        # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for the past 9 periods
        high_tenkan = df['high'].rolling(window=tenkan).max()
        low_tenkan = df['low'].rolling(window=tenkan).min()
        df['tenkan_sen'] = (high_tenkan + low_tenkan) / 2
        
        # Calculate Kijun-sen (Base Line): (highest high + lowest low)/2 for the past 26 periods
        high_kijun = df['high'].rolling(window=kijun).max()
        low_kijun = df['low'].rolling(window=kijun).min()
        df['kijun_sen'] = (high_kijun + low_kijun) / 2
        
        # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(kijun)
        
        # Calculate Senkou Span B (Leading Span B): (highest high + lowest low)/2 for the past 52 periods, shifted 26 periods ahead
        high_senkou = df['high'].rolling(window=senkou).max()
        low_senkou = df['low'].rolling(window=senkou).min()
        df['senkou_span_b'] = ((high_senkou + low_senkou) / 2).shift(kijun)
        
        # Calculate Chikou Span (Lagging Span): Current closing price, shifted 26 periods back
        df['chikou_span'] = df['close'].shift(-kijun)
        
        return df
    
    async def calculate_multi_timeframe_indicators(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Calculate indicators for multiple timeframes.
        
        Args:
            data_dict: Dictionary mapping timeframes to OHLCV DataFrames
        
        Returns:
            Dictionary of timeframes with indicator-enriched DataFrames
        """
        result = {}
        
        # Use ThreadPoolExecutor to process multiple timeframes in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for timeframe, df in data_dict.items():
                logger.info(f"Calculating indicators for {timeframe} timeframe")
                futures[timeframe] = executor.submit(self._calculate_indicators_sync, df)
            
            # Collect results
            for timeframe, future in futures.items():
                try:
                    result[timeframe] = future.result()
                except Exception as e:
                    logger.error(f"Error calculating indicators for {timeframe}: {str(e)}")
                    result[timeframe] = data_dict[timeframe]
        
        # Add cross-timeframe indicators
        result = self._add_cross_timeframe_indicators(result)
        
        return result
    
    def _calculate_indicators_sync(self, df: pd.DataFrame) -> pd.DataFrame:
        """Synchronous version of calculate_indicators for use with ThreadPoolExecutor."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.calculate_indicators(df))
        finally:
            loop.close()
    
    def _add_cross_timeframe_indicators(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Add indicators that use data from multiple timeframes.
        
        Args:
            data_dict: Dictionary of DataFrames with indicators for each timeframe
            
        Returns:
            Updated dictionary with cross-timeframe indicators
        """
        # Sort timeframes from smallest to largest
        sorted_timeframes = sorted(data_dict.keys(), 
                                  key=lambda x: self._timeframe_to_minutes(x))
        
        if len(sorted_timeframes) <= 1:
            return data_dict
        
        # Add cross-timeframe indicators to each timeframe
        for i, current_tf in enumerate(sorted_timeframes):
            current_df = data_dict[current_tf]
            
            # Skip if the current DataFrame is empty
            if current_df.empty:
                continue
                
            # Create cross-timeframe RSI divergence
            for j in range(i+1, len(sorted_timeframes)):
                higher_tf = sorted_timeframes[j]
                higher_df = data_dict[higher_tf]
                
                if higher_df.empty:
                    continue
                
                # Add higher timeframe RSI to current timeframe
                higher_rsi_resampled = self._resample_indicator(
                    higher_df, current_df.index, 'rsi_14', higher_tf
                )
                
                if higher_rsi_resampled is not None:
                    current_df[f'rsi_14_{higher_tf}'] = higher_rsi_resampled
                    
                    # Calculate RSI divergence between timeframes
                    if 'rsi_14' in current_df.columns:
                        current_df[f'rsi_divergence_{higher_tf}'] = current_df['rsi_14'] - higher_rsi_resampled
            
            # Update the data dictionary
            data_dict[current_tf] = current_df
        
        return data_dict
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes for comparison."""
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 1440
        else:
            return 0
    
    def _resample_indicator(self, source_df: pd.DataFrame, target_index, 
                          indicator_name: str, source_timeframe: str) -> Optional[pd.Series]:
        """
        Resample an indicator from a higher timeframe to a lower timeframe.
        
        Args:
            source_df: DataFrame containing the higher timeframe data
            target_index: Index of the target (lower) timeframe DataFrame
            indicator_name: Name of the indicator to resample
            source_timeframe: String identifying the source timeframe
            
        Returns:
            Resampled indicator Series or None if not possible
        """
        if indicator_name not in source_df.columns:
            return None
            
        try:
            # Create a Series with the indicator values
            source_series = source_df[indicator_name].copy()
            
            # Forward fill to the target index
            # First, ensure both have datetime indices
            if not isinstance(source_series.index, pd.DatetimeIndex):
                logger.warning(f"Source DataFrame does not have DatetimeIndex, cannot resample")
                return None
                
            if not isinstance(target_index, pd.DatetimeIndex):
                logger.warning(f"Target index is not DatetimeIndex, cannot resample")
                return None
            
            # Reindex with forward fill to lower timeframe
            resampled = source_series.reindex(
                target_index, method='ffill'
            )
            
            return resampled
            
        except Exception as e:
            logger.error(f"Error resampling {indicator_name} from {source_timeframe}: {str(e)}")
            return None
    
    def generate_signal_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Generate normalized features from indicators for ML models or trading signals.
        
        Args:
            df: DataFrame with technical indicators
        
        Returns:
            Dictionary of normalized features
        """
        # Get the most recent complete row of data
        if df.empty or df.isna().all().any():
            logger.warning("Cannot generate features from empty or incomplete data")
            return {}
        
        latest = df.iloc[-1].copy()
        features = {}
        
        try:
            # Trend features
            if 'rsi_14' in latest:
                features['rsi_norm'] = latest['rsi_14'] / 100.0  # Normalize to 0-1
            
            if all(x in latest for x in ['macd', 'macd_signal']):
                # MACD histogram normalized by price for cross-pair comparison
                features['macd_hist_norm'] = (latest['macd'] - latest['macd_signal']) / latest['close']
            
            # Momentum features
            if all(x in latest for x in ['close', 'sma_50']):
                features['price_to_sma50'] = latest['close'] / latest['sma_50'] - 1.0
            
            if all(x in latest for x in ['close', 'sma_200']):
                features['price_to_sma200'] = latest['close'] / latest['sma_200'] - 1.0
            
            # Volatility features
            if all(x in latest for x in ['bb_upper', 'bb_lower', 'close']):
                features['bb_width'] = (latest['bb_upper'] - latest['bb_lower']) / latest['close']
                features['bb_position'] = (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
            
            if 'natr' in latest:
                features['volatility'] = latest['natr'] / 100.0  # Normalize percentage to 0-1
            
            # Volume features
            if 'relative_volume' in latest:
                features['rel_volume'] = min(latest['relative_volume'] / 3.0, 1.0)  # Cap at 1.0 for extreme values
            
            if 'mfi' in latest:
                features['mfi_norm'] = latest['mfi'] / 100.0
            
            # Trend strength features
            if 'adx' in latest:
                features['trend_strength'] = latest['adx'] / 100.0
            
            if all(x in latest for x in ['di_plus', 'di_minus']):
                features['trend_direction'] = (latest['di_plus'] - latest['di_minus']) / (latest['di_plus'] + latest['di_minus'])
            
            # Add cross-timeframe features if available
            for col in latest.index:
                if col.startswith('rsi_14_') and col != 'rsi_14':
                    tf = col.replace('rsi_14_', '')
                    features[f'rsi_divergence_{tf}_norm'] = (latest['rsi_14'] - latest[col]) / 100.0
            
            # Crypto-specific features
            if 'vwap' in latest and 'close' in latest:
                features['vwap_distance'] = (latest['close'] / latest['vwap'] - 1.0)
                
            if 'mayer_multiple' in latest:
                features['mayer_multiple_norm'] = min(latest['mayer_multiple'] / 3.0, 1.0)
            
            return features
        
        except Exception as e:
            logger.exception(f"Error generating signal features: {str(e)}")
            return {}
            
    def generate_multi_timeframe_features(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate features from multiple timeframes for ML models.
        
        Args:
            data_dict: Dictionary mapping timeframes to DataFrames with indicators
            
        Returns:
            Dictionary of features from all timeframes
        """
        all_features = {}
        
        for timeframe, df in data_dict.items():
            if df.empty:
                continue
                
            # Get features for this timeframe
            tf_features = self.generate_signal_features(df)
            
            # Add timeframe prefix to avoid key collisions
            prefixed_features = {f"{timeframe}_{k}": v for k, v in tf_features.items()}
            
            # Add to all features
            all_features.update(prefixed_features)
        
        # Add cross-timeframe relationship features
        if len(data_dict) > 1:
            try:
                all_features.update(self._generate_cross_timeframe_relationships(data_dict))
            except Exception as e:
                logger.error(f"Error generating cross-timeframe relationships: {str(e)}")
        
        return all_features
    
    def _generate_cross_timeframe_relationships(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Generate features based on relationships between timeframes."""
        features = {}
        
        sorted_timeframes = sorted(data_dict.keys(), 
                                  key=lambda x: self._timeframe_to_minutes(x))
        
        if len(sorted_timeframes) <= 1:
            return features
            
        # Compare RSI across timeframes
        rsi_values = {}
        for tf in sorted_timeframes:
            df = data_dict[tf]
            if df.empty or 'rsi_14' not in df.columns:
                continue
            rsi_values[tf] = df['rsi_14'].iloc[-1]
        
        # Calculate RSI alignment across timeframes
        if len(rsi_values) > 1:
            rsi_list = list(rsi_values.values())
            features['rsi_alignment'] = 1.0 - (max(rsi_list) - min(rsi_list)) / 100.0
            
            # RSI trend (higher timeframes > lower timeframes indicates uptrend)
            rsi_trend = 0
            for i in range(len(sorted_timeframes)-1):
                if (sorted_timeframes[i] in rsi_values and 
                    sorted_timeframes[i+1] in rsi_values):
                    lower_tf = sorted_timeframes[i]
                    higher_tf = sorted_timeframes[i+1]
                    if rsi_values[higher_tf] > rsi_values[lower_tf]:
                        rsi_trend += 1
                    elif rsi_values[higher_tf] < rsi_values[lower_tf]:
                        rsi_trend -= 1
            
            features['rsi_trend'] = rsi_trend / max(1, len(sorted_timeframes)-1)
        
        return features
    
    async def monitor_positions(self, positions: Dict[str, Any], market_data_dict: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, Any]]:
        """
        Continuously monitor and recalculate indicators specifically for open positions.
        This provides focused, real-time technical analysis for active trades.
        
        Args:
            positions: Dictionary of open positions by symbol
            market_data_dict: Dictionary of market data by symbol and timeframe
        
        Returns:
            Dictionary of position-specific indicators and exit signals
        """
        position_indicators = {}
        
        for symbol, position in positions.items():
            # Skip if we don't have market data for this symbol
            if symbol not in market_data_dict:
                continue
                
            position_indicators[symbol] = {
                "position": position,
                "indicators": {},
                "exit_signals": {
                    "technical": False,
                    "reason": None,
                    "confidence": 0.0,
                    "suggested_exit_price": None
                }
            }
            
            # Extract data for the symbol across all timeframes
            symbol_data = market_data_dict[symbol]
            
            # Get position entry timeframe, defaulting to the lowest if not specified
            entry_timeframe = position.get("timeframe", min(symbol_data.keys()))
            
            # Recalculate indicators for this position's timeframe
            if entry_timeframe in symbol_data:
                df = symbol_data[entry_timeframe].copy()
                
                # Ensure we have the minimum required data
                if len(df) < 2:
                    continue
                
                # Apply indicators specific to exit decisions
                df = await self.calculate_indicators(df)
                
                # Calculate additional exit-specific indicators
                position_side = position.get("type", "buy").lower()
                entry_price = position.get("entry_price", 0)
                current_price = df["close"].iloc[-1]
                
                # Add position-specific calculations
                df["price_change_pct"] = (df["close"] / entry_price - 1) * 100
                if position_side == "sell":
                    df["price_change_pct"] = -df["price_change_pct"]
                
                # Calculate trailing stop levels
                atr = df["atr"].iloc[-1] if "atr" in df.columns else 0
                chandelier_exit = self._calculate_chandelier_exit(
                    df, position_side, multiplier=3
                )
                
                # Store key indicators for this position
                latest = df.iloc[-1]
                position_indicators[symbol]["indicators"] = {
                    "timeframe": entry_timeframe,
                    "current_price": current_price,
                    "price_change_pct": latest["price_change_pct"],
                    "atr": atr,
                    "rsi": latest.get("rsi_14", None),
                    "macd_hist": latest.get("macd_hist", None),
                    "chandelier_exit": chandelier_exit,
                    "bb_position": latest.get("bb_percent_b", None)
                }
                
                # Generate technical exit signals
                self._identify_exit_signals(
                    position_indicators[symbol], 
                    position, 
                    df
                )
                
                # Also check higher timeframes for confirmation
                higher_tf_signals = self._check_higher_timeframe_confirmations(
                    symbol_data, entry_timeframe, position_side
                )
                
                position_indicators[symbol]["indicators"]["higher_timeframe_signals"] = higher_tf_signals
                
                # Adjust exit confidence based on higher timeframe confirmations
                if higher_tf_signals.get("confirms_exit", False):
                    position_indicators[symbol]["exit_signals"]["confidence"] += 0.2
                    position_indicators[symbol]["exit_signals"]["higher_tf_confirms"] = True
            
        return position_indicators
    
    def _calculate_chandelier_exit(self, df: pd.DataFrame, side: str, multiplier: float = 3) -> float:
        """
        Calculate chandelier exit level based on ATR.
        For long positions: Highest high - ATR * multiplier
        For short positions: Lowest low + ATR * multiplier
        """
        if "atr" not in df.columns or df.empty:
            return None
            
        periods = min(len(df), 22)  # Look back about a month in trading days
        
        if side.lower() == "buy":
            # For long positions, highest high minus ATR multiple
            highest_high = df["high"].iloc[-periods:].max()
            return highest_high - (df["atr"].iloc[-1] * multiplier)
        else:
            # For short positions, lowest low plus ATR multiple
            lowest_low = df["low"].iloc[-periods:].min()
            return lowest_low + (df["atr"].iloc[-1] * multiplier)
    
    def _identify_exit_signals(self, position_data: Dict[str, Any], 
                              position: Dict[str, Any], 
                              df: pd.DataFrame) -> None:
        """Identify technical exit signals for a position"""
        exit_signals = position_data["exit_signals"]
        indicators = position_data["indicators"]
        position_side = position.get("type", "buy").lower()
        
        # Initial confidence - will be increased by each confirmed signal
        confidence = 0.0
        reasons = []
        
        # 1. RSI reversal signal
        if "rsi_14" in df.columns:
            rsi = df["rsi_14"].iloc[-1]
            prev_rsi = df["rsi_14"].iloc[-2] if len(df) > 2 else rsi
            
            # For buy positions, check if RSI is falling from overbought
            if position_side == "buy" and prev_rsi >= 70 and rsi < prev_rsi:
                confidence += 0.3
                reasons.append("RSI falling from overbought")
            
            # For sell positions, check if RSI is rising from oversold
            elif position_side == "sell" and prev_rsi <= 30 and rsi > prev_rsi:
                confidence += 0.3
                reasons.append("RSI rising from oversold")
        
        # 2. MACD crossover
        if all(x in df.columns for x in ["macd", "macd_signal"]):
            macd = df["macd"].iloc[-1]
            macd_signal = df["macd_signal"].iloc[-1]
            prev_macd = df["macd"].iloc[-2] if len(df) > 2 else macd
            prev_macd_signal = df["macd_signal"].iloc[-2] if len(df) > 2 else macd_signal
            
            # For buy positions, check if MACD crosses below signal line
            if position_side == "buy" and prev_macd > prev_macd_signal and macd < macd_signal:
                confidence += 0.25
                reasons.append("MACD bearish crossover")
            
            # For sell positions, check if MACD crosses above signal line
            elif position_side == "sell" and prev_macd < prev_macd_signal and macd > macd_signal:
                confidence += 0.25
                reasons.append("MACD bullish crossover")
        
        # 3. Bollinger Band exit signal
        if "bb_percent_b" in df.columns:
            bb_position = df["bb_percent_b"].iloc[-1]
            
            # For buy positions, price hitting upper band is a potential exit
            if position_side == "buy" and bb_position > 0.95:
                confidence += 0.2
                reasons.append("Price at upper Bollinger Band")
            
            # For sell positions, price hitting lower band is a potential exit
            elif position_side == "sell" and bb_position < 0.05:
                confidence += 0.2
                reasons.append("Price at lower Bollinger Band")
        
        # 4. Chandelier exit triggered
        if indicators["chandelier_exit"] is not None:
            current_price = indicators["current_price"]
            
            if position_side == "buy" and current_price < indicators["chandelier_exit"]:
                confidence += 0.4
                reasons.append("Chandelier exit triggered")
            elif position_side == "sell" and current_price > indicators["chandelier_exit"]:
                confidence += 0.4
                reasons.append("Chandelier exit triggered")
        
        # Update exit signals
        exit_signals["technical"] = confidence >= 0.5
        exit_signals["confidence"] = min(confidence, 1.0)  # Cap at 1.0
        exit_signals["reason"] = " & ".join(reasons) if reasons else None
        
        # Suggest exit price - use next bar's open as a conservative estimate
        # or current price if chandelier exit is triggered
        if exit_signals["technical"]:
            if "chandelier_exit" in reasons:
                exit_signals["suggested_exit_price"] = indicators["chandelier_exit"]
            else:
                exit_signals["suggested_exit_price"] = indicators["current_price"]
    
    def _check_higher_timeframe_confirmations(
        self, 
        symbol_data: Dict[str, pd.DataFrame], 
        entry_timeframe: str, 
        position_side: str
    ) -> Dict[str, Any]:
        """Check higher timeframes for confirmation of exit signals"""
        timeframes = sorted(symbol_data.keys(), key=self._timeframe_to_minutes)
        
        try:
            # Find the index of entry timeframe
            tf_index = timeframes.index(entry_timeframe)
        except ValueError:
            return {"confirms_exit": False}
        
        # Get higher timeframes
        higher_timeframes = timeframes[tf_index+1:] if tf_index < len(timeframes) - 1 else []
        
        if not higher_timeframes:
            return {"confirms_exit": False}
        
        # Check confirmation signals in higher timeframes
        confirmations = {"confirms_exit": False, "details": {}}
        
        for tf in higher_timeframes:
            df = symbol_data[tf]
            if df.empty or len(df) < 2:
                continue
                
            tf_signals = {}
            
            # Check MACD
            if all(x in df.columns for x in ["macd", "macd_signal", "macd_hist"]):
                macd_hist = df["macd_hist"].iloc[-1]
                prev_macd_hist = df["macd_hist"].iloc[-2]
                
                # For buy positions, declining histogram suggests exit
                if position_side == "buy" and macd_hist < prev_macd_hist:
                    tf_signals["macd_confirms_exit"] = True
                
                # For sell positions, rising histogram suggests exit
                elif position_side == "sell" and macd_hist > prev_macd_hist:
                    tf_signals["macd_confirms_exit"] = True
            
            # Check RSI trend
            if "rsi_14" in df.columns:
                rsi = df["rsi_14"].iloc[-1]
                prev_rsi = df["rsi_14"].iloc[-2]
                
                if position_side == "buy" and rsi < prev_rsi:
                    tf_signals["rsi_confirms_exit"] = True
                elif position_side == "sell" and rsi > prev_rsi:
                    tf_signals["rsi_confirms_exit"] = True
            
            # Store signals for this timeframe
            confirmations["details"][tf] = tf_signals
            
            # If any higher timeframe has confirming signals, set overall confirmation
            if any(tf_signals.values()):
                confirmations["confirms_exit"] = True
        
        return confirmations 