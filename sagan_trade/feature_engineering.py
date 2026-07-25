"""
Feature Engineering Pipeline for Quantitative Finance.

Provides automated feature discovery, generation, and selection for
financial time series. Includes:
- Technical indicators (100+)
- Microstructure features
- Cross-sectional features
- Alternative data integration
- Feature importance ranking
- Automated feature selection
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Callable, Union, Literal
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import argrelextrema
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    mutual_info_regression, f_regression, SelectKBest, RFE
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    warnings.warn("TA-Lib not available. Using custom implementations.")


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    # Technical indicators
    include_trend: bool = True
    include_momentum: bool = True
    include_volatility: bool = True
    include_volume: bool = True
    include_cycle: bool = False
    include_pattern: bool = False
    
    # Lookback windows
    short_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    medium_windows: List[int] = field(default_factory=lambda: [50, 100])
    long_windows: List[int] = field(default_factory=lambda: [200])
    
    # Microstructure
    include_microstructure: bool = True
    include_ofi: bool = True  # Order Flow Imbalance
    include_vpin: bool = True  # Volume-synchronized PIN
    
    # Cross-sectional
    include_cross_sectional: bool = False
    market_neutralize: bool = True
    sector_neutralize: bool = False
    
    # Alternative data
    include_sentiment: bool = False
    include_fundamentals: bool = False
    
    # Feature selection
    selection_method: Literal["mutual_info", "f_regression", "lasso", "rf_importance", "none"] = "mutual_info"
    max_features: int = 100
    min_importance: float = 0.001
    
    # Preprocessing
    scaler: Literal["standard", "robust", "none"] = "robust"
    clip_outliers: float = 5.0  # Z-score threshold
    fill_method: Literal["ffill", "bfill", "interpolate", "zero"] = "ffill"


class TechnicalIndicators:
    """Comprehensive technical indicator library."""
    
    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window).mean()
    
    @staticmethod
    def ema(series: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def wma(series: pd.Series, window: int) -> pd.Series:
        """Weighted Moving Average."""
        weights = np.arange(1, window + 1)
        return series.rolling(window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    @staticmethod
    def rsi(series: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = -delta.clip(upper=0).rolling(window).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence)."""
        ema_fast = TechnicalIndicators.ema(series, fast)
        ema_slow = TechnicalIndicators.ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(series: pd.Series, window: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands."""
        sma = TechnicalIndicators.sma(series, window)
        std = series.rolling(window).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        return upper, sma, lower
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range."""
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window).mean()
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average Directional Index."""
        plus_dm = high.diff()
        minus_dm = low.diff().abs()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = TechnicalIndicators.atr(high, low, close, 1)
        
        plus_di = 100 * (plus_dm.rolling(window).mean() / tr.rolling(window).mean())
        minus_di = 100 * (minus_dm.rolling(window).mean() / tr.rolling(window).mean())
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        return dx.rolling(window).mean()
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator."""
        lowest_low = low.rolling(k).min()
        highest_high = high.rolling(k).max()
        k_line = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        d_line = k_line.rolling(d).mean()
        return k_line, d_line
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R."""
        highest_high = high.rolling(window).max()
        lowest_low = low.rolling(window).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """Commodity Channel Index."""
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window).mean()
        mean_dev = tp.rolling(window).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (tp - sma_tp) / (0.015 * mean_dev + 1e-10)
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume."""
        direction = np.sign(close.diff()).fillna(0)
        return (direction * volume).cumsum()
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price."""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
        """Money Flow Index."""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window).sum()
        mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-10)))
        return mfi
    
    @staticmethod
    def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20, atr_mult: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Keltner Channels."""
        ema = TechnicalIndicators.ema(close, window)
        atr_val = TechnicalIndicators.atr(high, low, close, window)
        upper = ema + atr_mult * atr_val
        lower = ema - atr_mult * atr_val
        return upper, ema, lower
    
    @staticmethod
    def donchian_channels(high: pd.Series, low: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Donchian Channels."""
        upper = high.rolling(window).max()
        lower = low.rolling(window).min()
        middle = (upper + lower) / 2
        return upper, middle, lower
    
    @staticmethod
    def supertrend(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 10, multiplier: float = 3) -> Tuple[pd.Series, pd.Series]:
        """Supertrend Indicator."""
        atr_val = TechnicalIndicators.atr(high, low, close, window)
        basic_upper = (high + low) / 2 + multiplier * atr_val
        basic_lower = (high + low) / 2 - multiplier * atr_val
        
        final_upper = basic_upper.copy()
        final_lower = basic_lower.copy()
        supertrend = pd.Series(index=close.index, dtype=float)
        trend = pd.Series(index=close.index, dtype=int)
        
        for i in range(1, len(close)):
            if close.iloc[i] > final_upper.iloc[i-1]:
                final_upper.iloc[i] = basic_upper.iloc[i]
            else:
                final_upper.iloc[i] = max(basic_upper.iloc[i], final_upper.iloc[i-1])
                
            if close.iloc[i] < final_lower.iloc[i-1]:
                final_lower.iloc[i] = basic_lower.iloc[i]
            else:
                final_lower.iloc[i] = min(basic_lower.iloc[i], final_lower.iloc[i-1])
                
            if close.iloc[i] <= final_upper.iloc[i]:
                supertrend.iloc[i] = final_upper.iloc[i]
                trend.iloc[i] = -1
            else:
                supertrend.iloc[i] = final_lower.iloc[i]
                trend.iloc[i] = 1
                
        return supertrend, trend


class MicrostructureFeatures:
    """Market microstructure feature engineering."""
    
    @staticmethod
    def order_flow_imbalance(
        bid_price: pd.Series, bid_volume: pd.Series,
        ask_price: pd.Series, ask_volume: pd.Series,
        window: int = 100
    ) -> pd.Series:
        """
        Order Flow Imbalance (OFI).
        
        OFI = sum(bid_volume_change at same price) - sum(ask_volume_change at same price)
        """
        # Price changes
        bid_price_chg = bid_price.diff()
        ask_price_chg = ask_price.diff()
        
        # Volume changes
        bid_vol_chg = bid_volume.diff()
        ask_vol_chg = ask_volume.diff()
        
        # OFI components
        bid_ofi = np.where(bid_price_chg >= 0, bid_vol_chg, 0)
        ask_ofi = np.where(ask_price_chg <= 0, ask_vol_chg, 0)
        
        ofi = bid_ofi - ask_ofi
        return pd.Series(ofi, index=bid_price.index).rolling(window).sum()
    
    @staticmethod
    def vpin(
        buy_volume: pd.Series, sell_volume: pd.Series,
        total_volume: pd.Series, bucket_size: float = 1000
    ) -> pd.Series:
        """
        Volume-Synchronized Probability of Informed Trading (VPIN).
        
        VPIN = |buy_vol - sell_vol| / (buy_vol + sell_vol) per bucket
        """
        # Create volume buckets
        cum_vol = total_volume.cumsum()
        n_buckets = int(cum_vol.iloc[-1] / bucket_size)
        
        vpin_vals = []
        for i in range(n_buckets):
            start_vol = i * bucket_size
            end_vol = (i + 1) * bucket_size
            
            mask = (cum_vol >= start_vol) & (cum_vol < end_vol)
            if mask.any():
                bv = buy_volume[mask].sum()
                sv = sell_volume[mask].sum()
                vpin = abs(bv - sv) / (bv + sv + 1e-10)
                vpin_vals.append(vpin)
                
        # Resample to original index
        vpin_series = pd.Series(vpin_vals, index=cum_vol.iloc[::len(cum_vol)//n_buckets].index[:len(vpin_vals)])
        return vpin_series.reindex(total_volume.index).ffill()
    
    @staticmethod
    def realized_volatility(returns: pd.Series, window: int = 100, scaling: float = np.sqrt(252 * 390)) -> pd.Series:
        """Realized Volatility (high-frequency)."""
        return returns.rolling(window).std() * scaling
    
    @staticmethod
    def bipower_variation(returns: pd.Series, window: int = 100) -> pd.Series:
        """Bipower Variation (robust to jumps)."""
        abs_ret = returns.abs()
        bpv = (np.pi / 2) * (abs_ret * abs_ret.shift(1)).rolling(window).sum()
        return bpv
    
    @staticmethod
    def jump_variation(returns: pd.Series, window: int = 100) -> pd.Series:
        """Jump Variation = RV - BPV."""
        rv = MicrostructureFeatures.realized_volatility(returns, window) ** 2
        bpv = MicrostructureFeatures.bipower_variation(returns, window)
        return (rv - bpv).clip(lower=0)
    
    @staticmethod
    def kyle_lambda(returns: pd.Series, volume: pd.Series, window: int = 100) -> pd.Series:
        """Kyle's Lambda (price impact coefficient)."""
        # Regression: returns = lambda * signed_volume + epsilon
        signed_vol = volume * np.sign(returns)
        lambda_vals = []
        
        for i in range(window, len(returns)):
            y = returns.iloc[i-window:i].values
            x = signed_vol.iloc[i-window:i].values
            x = x.reshape(-1, 1)
            
            if np.std(x) > 0:
                lam = np.cov(y, x.ravel())[0, 1] / np.var(x)
                lambda_vals.append(lam)
            else:
                lambda_vals.append(np.nan)
                
        return pd.Series(lambda_vals, index=returns.index[window:])
    
    @staticmethod
    def amihud_illiquidity(returns: pd.Series, volume: pd.Series, price: pd.Series, window: int = 100) -> pd.Series:
        """Amihud Illiquidity Ratio."""
        dollar_vol = volume * price
        illiq = returns.abs() / (dollar_vol + 1e-10)
        return illiq.rolling(window).mean()
    
    @staticmethod
    def roll_spread(returns: pd.Series, window: int = 100) -> pd.Series:
        """Roll's Spread Estimator."""
        cov = returns.rolling(window).cov(returns.shift(1))
        spread = 2 * np.sqrt(-cov.clip(upper=0))
        return spread
    
    @staticmethod
    def corwin_schultz_spread(high: pd.Series, low: pd.Series, window: int = 2) -> pd.Series:
        """Corwin-Schultz Spread Estimator."""
        # High-low ratio over 2 days
        hl_ratio = np.log(high / low)
        hl_ratio_2d = hl_ratio.rolling(window).sum()
        
        # Overnight return
        beta = np.sum(hl_ratio_2d ** 2) / (2 - np.sqrt(2))
        gamma = np.sqrt(beta / (2 - np.sqrt(2)))
        spread = 2 * (np.exp(gamma) - 1) / (1 + np.exp(gamma))
        
        return pd.Series(spread, index=high.index)


class CrossSectionalFeatures:
    """Cross-sectional feature engineering."""
    
    @staticmethod
    def rank_ic(returns: pd.DataFrame, factor: pd.DataFrame, window: int = 20) -> pd.Series:
        """Rank Information Coefficient (cross-sectional correlation)."""
        ic = returns.rolling(window).corrwith(factor, axis=1)
        return ic.mean(axis=1)
    
    @staticmethod
    def market_neutralize(returns: pd.DataFrame, market_returns: pd.Series) -> pd.DataFrame:
        """Market-neutralize returns."""
        beta = returns.rolling(60).cov(market_returns) / market_returns.rolling(60).var()
        neutral = returns.sub(beta.mul(market_returns, axis=0), axis=0)
        return neutral
    
    @staticmethod
    def sector_neutralize(returns: pd.DataFrame, sector_map: Dict[str, str]) -> pd.DataFrame:
        """Sector-neutralize returns."""
        sectors = pd.Series(sector_map)
        sector_returns = returns.groupby(sectors, axis=1).mean()
        
        neutral = pd.DataFrame(index=returns.index, columns=returns.columns)
        for sector in sector_returns.columns:
            assets = sectors[sectors == sector].index
            if len(assets) > 1:
                sector_ret = sector_returns[sector]
                neutral[assets] = returns[assets].sub(sector_ret, axis=0)
                
        return neutral
    
    @staticmethod
    def z_score_cross_sectional(data: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """Cross-sectional Z-score normalization."""
        mean = data.rolling(window).mean(axis=1)
        std = data.rolling(window).std(axis=1)
        return data.sub(mean, axis=0).div(std + 1e-10, axis=0)


class FeatureEngine:
    """
    Main feature engineering pipeline.
    
    Automatically generates, selects, and manages features for ML models.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.feature_names: List[str] = []
        self.importance_scores: Dict[str, float] = {}
        self.selected_features: List[str] = []
        self.scaler = None
        self._fitted = False
        
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> "FeatureEngine":
        """Fit the feature engineering pipeline."""
        # Generate features
        features = self.generate_features(data)
        
        # Select features
        if target is not None:
            self.select_features(features, target)
        else:
            self.selected_features = list(features.columns)
            
        # Fit scaler
        if self.config.scaler != "none":
            self._fit_scaler(features[self.selected_features])
            
        self._fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted pipeline."""
        if not self._fitted:
            raise ValueError("FeatureEngine must be fitted before transform")
            
        features = self.generate_features(data)
        features = features[self.selected_features]
        
        if self.config.scaler != "none":
            features = self._apply_scaler(features)
            
        return features
    
    def fit_transform(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(data, target).transform(data)
    
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive feature set from OHLCV data.
        
        Expected columns: open, high, low, close, volume
        """
        df = data.copy()
        features = pd.DataFrame(index=df.index)
        
        # Ensure required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found")
                
        open_ = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # Returns
        features['returns_1'] = close.pct_change()
        features['log_returns_1'] = np.log(close / close.shift(1))
        
        for w in [5, 10, 20, 50]:
            features[f'returns_{w}'] = close.pct_change(w)
            features[f'log_returns_{w}'] = np.log(close / close.shift(w))
        
        # Trend indicators
        if self.config.include_trend:
            for w in self.config.short_windows + self.config.medium_windows + self.config.long_windows:
                features[f'sma_{w}'] = TechnicalIndicators.sma(close, w) / close - 1
                features[f'ema_{w}'] = TechnicalIndicators.ema(close, w) / close - 1
                features[f'wma_{w}'] = TechnicalIndicators.wma(close, w) / close - 1
                
                # Price relative to MA
                features[f'price_to_sma_{w}'] = close / TechnicalIndicators.sma(close, w) - 1
                features[f'price_to_ema_{w}'] = close / TechnicalIndicators.ema(close, w) - 1
            
            # MACD
            macd, signal, hist = TechnicalIndicators.macd(close)
            features['macd'] = macd
            features['macd_signal'] = signal
            features['macd_hist'] = hist
            
            # ADX
            for w in [14, 20]:
                features[f'adx_{w}'] = TechnicalIndicators.adx(high, low, close, w)
            
            # Supertrend
            st, trend = TechnicalIndicators.supertrend(high, low, close)
            features['supertrend'] = (close - st) / close
            features['supertrend_trend'] = trend
            
            # Parabolic SAR (approximation)
            features['sar'] = self._parabolic_sar(high, low, close)
        
        # Momentum indicators
        if self.config.include_momentum:
            for w in [14, 20]:
                features[f'rsi_{w}'] = TechnicalIndicators.rsi(close, w) / 100 - 0.5
                k, d = TechnicalIndicators.stochastic(high, low, close)
                features[f'stoch_k_{w}'] = k / 100 - 0.5
                features[f'stoch_d_{w}'] = d / 100 - 0.5
                features[f'williams_r_{w}'] = TechnicalIndicators.williams_r(high, low, close, w) / 100 + 1
                features[f'cci_{w}'] = TechnicalIndicators.cci(high, low, close, w) / 100
            
            # Rate of Change
            for w in [5, 10, 20]:
                features[f'roc_{w}'] = close.pct_change(w)
            
            # Momentum
            for w in [10, 20]:
                features[f'momentum_{w}'] = close / close.shift(w) - 1
        
        # Volatility indicators
        if self.config.include_volatility:
            for w in [10, 20, 50]:
                # ATR
                features[f'atr_{w}'] = TechnicalIndicators.atr(high, low, close, w) / close
                
                # Bollinger Bands
                upper, middle, lower = TechnicalIndicators.bollinger_bands(close, w)
                features[f'bb_upper_{w}'] = (upper - close) / close
                features[f'bb_lower_{w}'] = (close - lower) / close
                features[f'bb_width_{w}'] = (upper - lower) / middle
                features[f'bb_position_{w}'] = (close - lower) / (upper - lower + 1e-10)
                
                # Keltner Channels
                kc_upper, kc_middle, kc_lower = TechnicalIndicators.keltner_channels(high, low, close, w)
                features[f'kc_upper_{w}'] = (kc_upper - close) / close
                features[f'kc_lower_{w}'] = (close - kc_lower) / close
                
                # Donchian Channels
                dc_upper, dc_middle, dc_lower = TechnicalIndicators.donchian_channels(high, low, w)
                features[f'dc_upper_{w}'] = (dc_upper - close) / close
                features[f'dc_lower_{w}'] = (close - dc_lower) / close
                
                # Historical Volatility
                features[f'hist_vol_{w}'] = features['log_returns_1'].rolling(w).std() * np.sqrt(252)
        
        # Volume indicators
        if self.config.include_volume:
            # OBV
            features['obv'] = TechnicalIndicators.obv(close, volume)
            features['obv_change'] = features['obv'].pct_change()
            
            # VWAP
            features['vwap'] = TechnicalIndicators.vwap(high, low, close, volume) / close - 1
            
            # MFI
            for w in [14, 20]:
                features[f'mfi_{w}'] = TechnicalIndicators.mfi(high, low, close, volume, w) / 100 - 0.5
            
            # Volume features
            for w in [5, 10, 20]:
                features[f'volume_sma_{w}'] = volume.rolling(w).mean() / volume - 1
                features[f'volume_std_{w}'] = volume.rolling(w).std() / volume.rolling(w).mean()
                features[f'volume_ratio_{w}'] = volume / volume.rolling(w).mean() - 1
            
            # Price-Volume Trend
            features['pvt'] = (volume * close.pct_change()).cumsum()
        
        # Microstructure features
        if self.config.include_microstructure:
            # Proxy using OHLCV
            features['hl_spread'] = (high - low) / close
            features['oc_spread'] = (close - open_).abs() / close
            
            # Garman-Klass volatility
            features['gk_vol'] = np.sqrt(
                0.5 * np.log(high/low)**2 - (2*np.log(2)-1) * np.log(close/open_)**2
            )
            
            # Parkinson volatility
            features['parkinson_vol'] = np.sqrt(1/(4*np.log(2)) * np.log(high/low)**2)
            
            # Yang-Zhang volatility
            features['yz_vol'] = self._yang_zhang_vol(open_, high, low, close)
            
            # Roll spread
            features['roll_spread'] = MicrostructureFeatures.roll_spread(features['log_returns_1'])
            
            # Amihud illiquidity
            features['amihud_illiq'] = MicrostructureFeatures.amihud_illiquidity(
                features['log_returns_1'], volume, close
            )
        
        # Time features
        features['hour'] = df.index.hour if hasattr(df.index, 'hour') else 0
        features['day_of_week'] = df.index.dayofweek if hasattr(df.index, 'dayofweek') else 0
        features['month'] = df.index.month if hasattr(df.index, 'month') else 0
        features['quarter'] = df.index.quarter if hasattr(df.index, 'quarter') else 0
        
        # Cyclical encoding
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 5)
        features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 5)
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            features[f'returns_lag_{lag}'] = features['returns_1'].shift(lag)
            features[f'volume_lag_{lag}'] = features['volume_ratio_5'].shift(lag)
        
        # Rolling statistics
        for w in [5, 10, 20]:
            features[f'returns_skew_{w}'] = features['returns_1'].rolling(w).skew()
            features[f'returns_kurt_{w}'] = features['returns_1'].rolling(w).kurt()
            features[f'returns_max_{w}'] = features['returns_1'].rolling(w).max()
            features[f'returns_min_{w}'] = features['returns_1'].rolling(w).min()
        
        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Handle missing values
        if self.config.fill_method == "ffill":
            features = features.ffill()
        elif self.config.fill_method == "bfill":
            features = features.bfill()
        elif self.config.fill_method == "interpolate":
            features = features.interpolate()
        elif self.config.fill_method == "zero":
            features = features.fillna(0)
            
        # Clip outliers
        if self.config.clip_outliers > 0:
            for col in features.columns:
                zscore = (features[col] - features[col].mean()) / (features[col].std() + 1e-10)
                features[col] = features[col].clip(
                    features[col].mean() - self.config.clip_outliers * features[col].std(),
                    features[col].mean() + self.config.clip_outliers * features[col].std()
                )
        
        self.feature_names = list(features.columns)
        return features
    
    def _parabolic_sar(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                       af: float = 0.02, max_af: float = 0.2) -> pd.Series:
        """Parabolic SAR approximation."""
        sar = close.copy()
        ep = high.copy()  # Extreme point
        trend = pd.Series(1, index=close.index)  # 1 = up, -1 = down
        af_series = pd.Series(af, index=close.index)
        
        for i in range(2, len(close)):
            if trend.iloc[i-1] == 1:  # Uptrend
                sar.iloc[i] = sar.iloc[i-1] + af_series.iloc[i-1] * (ep.iloc[i-1] - sar.iloc[i-1])
                if low.iloc[i] < sar.iloc[i]:
                    trend.iloc[i] = -1
                    sar.iloc[i] = ep.iloc[i-1]
                    ep.iloc[i] = low.iloc[i]
                    af_series.iloc[i] = af
                else:
                    trend.iloc[i] = 1
                    if high.iloc[i] > ep.iloc[i-1]:
                        ep.iloc[i] = high.iloc[i]
                        af_series.iloc[i] = min(af_series.iloc[i-1] + af, max_af)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af_series.iloc[i] = af_series.iloc[i-1]
            else:  # Downtrend
                sar.iloc[i] = sar.iloc[i-1] + af_series.iloc[i-1] * (ep.iloc[i-1] - sar.iloc[i-1])
                if high.iloc[i] > sar.iloc[i]:
                    trend.iloc[i] = 1
                    sar.iloc[i] = ep.iloc[i-1]
                    ep.iloc[i] = high.iloc[i]
                    af_series.iloc[i] = af
                else:
                    trend.iloc[i] = -1
                    if low.iloc[i] < ep.iloc[i-1]:
                        ep.iloc[i] = low.iloc[i]
                        af_series.iloc[i] = min(af_series.iloc[i-1] + af, max_af)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af_series.iloc[i] = af_series.iloc[i-1]
                        
        return sar
    
    def _yang_zhang_vol(self, open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """Yang-Zhang Volatility Estimator."""
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        
        log_ho = np.log(high / open_)
        log_lo = np.log(low / open_)
        log_co = np.log(close / open_)
        
        log_oc = np.log(open_ / close.shift(1))
        log_oc_sq = log_oc ** 2
        
        log_cc = np.log(close / close.shift(1))
        log_cc_sq = log_cc ** 2
        
        rs = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(window).sum() / (window - 1)
        open_vol = log_oc_sq.rolling(window).sum() / (window - 1)
        close_vol = log_cc_sq.rolling(window).sum() / (window - 1)
        
        return np.sqrt(open_vol + k * close_vol + (1 - k) * rs) * np.sqrt(252)
    
    def select_features(
        self, 
        features: pd.DataFrame, 
        target: pd.Series,
        method: Optional[str] = None
    ) -> List[str]:
        """Select most important features."""
        method = method or self.config.selection_method
        
        # Align
        common_idx = features.index.intersection(target.index)
        X = features.loc[common_idx].fillna(0)
        y = target.loc[common_idx].fillna(0)
        
        if method == "mutual_info":
            scores = mutual_info_regression(X, y, random_state=42)
            importance = pd.Series(scores, index=X.columns).sort_values(ascending=False)
            
        elif method == "f_regression":
            scores, _ = f_regression(X.fillna(0), y)
            importance = pd.Series(scores, index=X.columns).sort_values(ascending=False)
            
        elif method == "lasso":
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.fillna(0))
            lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
            lasso.fit(X_scaled, y)
            importance = pd.Series(np.abs(lasso.coef_), index=X.columns).sort_values(ascending=False)
            
        elif method == "rf_importance":
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X.fillna(0), y)
            importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
            
        else:
            importance = pd.Series(1.0, index=X.columns)
            
        # Filter by importance threshold
        importance = importance[importance >= self.config.min_importance]
        
        # Limit number of features
        self.importance_scores = importance.to_dict()
        self.selected_features = importance.head(self.config.max_features).index.tolist()
        
        return self.selected_features
    
    def _fit_scaler(self, features: pd.DataFrame):
        """Fit scaler on features."""
        if self.config.scaler == "standard":
            self.scaler = StandardScaler()
        elif self.config.scaler == "robust":
            self.scaler = RobustScaler()
        else:
            return
            
        self.scaler.fit(features.fillna(0))
    
    def _apply_scaler(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scaler."""
        if self.scaler is None:
            return features
            
        scaled = self.scaler.transform(features.fillna(0))
        return pd.DataFrame(scaled, index=features.index, columns=features.columns)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        return pd.Series(self.importance_scores).sort_values(ascending=False)
    
    def plot_feature_importance(self, top_n: int = 20):
        """Plot feature importance."""
        try:
            import matplotlib.pyplot as plt
            importance = self.get_feature_importance().head(top_n)
            importance.plot(kind='barh', figsize=(10, 8))
            plt.title(f'Top {top_n} Feature Importance')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.show()
        except ImportError:
            warnings.warn("Matplotlib not available for plotting")


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    # Technical indicators
    include_trend: bool = True
    include_momentum: bool = True
    include_volatility: bool = True
    include_volume: bool = True
    include_cycle: bool = False
    include_pattern: bool = False
    
    # Lookback windows
    short_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    medium_windows: List[int] = field(default_factory=lambda: [50, 100])
    long_windows: List[int] = field(default_factory=lambda: [200])
    
    # Microstructure
    include_microstructure: bool = True
    include_ofi: bool = True
    include_vpin: bool = True
    
    # Cross-sectional
    include_cross_sectional: bool = False
    market_neutralize: bool = True
    sector_neutralize: bool = False
    
    # Alternative data
    include_sentiment: bool = False
    include_fundamentals: bool = False
    
    # Feature selection
    selection_method: Literal["mutual_info", "f_regression", "lasso", "rf_importance", "none"] = "mutual_info"
    max_features: int = 100
    min_importance: float = 0.001
    
    # Preprocessing
    scaler: Literal["standard", "robust", "none"] = "robust"
    clip_outliers: float = 5.0
    fill_method: Literal["ffill", "bfill", "interpolate", "zero"] = "ffill"


# Convenience functions
def create_feature_engine(
    config: Optional[FeatureConfig] = None,
    **kwargs
) -> FeatureEngine:
    """Create feature engine with default or custom config."""
    if config is None:
        config = FeatureConfig()
    for k, v in kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)
    return FeatureEngine(config)


def generate_features(
    data: pd.DataFrame,
    config: Optional[FeatureConfig] = None
) -> pd.DataFrame:
    """Quick feature generation."""
    engine = create_feature_engine(config)
    return engine.generate_features(data)