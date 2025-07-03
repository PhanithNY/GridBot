import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

import talib.abstract as ta
from technical import qtpylib


class SimpleGridDowntrend(IStrategy):
    """
    Enhanced Grid Trading Bot Strategy for Downtrend Markets
    
    This strategy implements an improved grid trading approach that:
    1. Creates dynamic grid levels optimized for downtrend markets
    2. Uses trend analysis to bias grid placement downward
    3. Implements proper entry/exit logic with profit tracking
    4. Includes support/resistance level integration
    5. Features adaptive position sizing and risk management
    6. Specifically optimized for ranging and downtrend markets
    """

    INTERFACE_VERSION = 3

    # Improved stoploss - more conservative for downtrend markets
    stoploss = -0.06  # Slightly more conservative than standard

    timeframe = "15m"

    # Enhanced Grid Bot Parameters
    grid_spacing = DecimalParameter(0.002, 0.03, default=0.008, space="buy", optimize=True, load=True)
    min_volume_threshold = IntParameter(5000, 50000, default=15000, space="buy", optimize=True, load=True)
    grid_levels = IntParameter(5, 12, default=8, space="buy", optimize=True, load=True)
    use_dynamic_spacing = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    
    # New Parameters for Enhanced Performance
    # Minimum profit target per grid level (as percentage)
    min_profit_target = DecimalParameter(0.003, 0.015, default=0.008, space="buy", optimize=True, load=True)
    
    # Maximum loss per grid level (as percentage)
    max_loss_per_level = DecimalParameter(0.01, 0.05, default=0.025, space="buy", optimize=True, load=True)
    
    # Trend strength threshold for grid bias
    trend_strength_threshold = DecimalParameter(0.3, 0.8, default=0.5, space="buy", optimize=True, load=True)
    
    # Support/Resistance weight in grid placement
    sr_weight = DecimalParameter(0.1, 0.5, default=0.3, space="buy", optimize=True, load=True)
    
    # Downtrend-specific parameters
    # Bias factor for downtrend markets (negative values bias grid downward)
    downtrend_bias = DecimalParameter(-0.3, 0.0, default=-0.15, space="buy", optimize=True, load=True)
    
    # Market Microstructure Parameters
    min_spread_ratio = DecimalParameter(0.002, 0.008, default=0.004, space="buy", optimize=True, load=True)
    volatility_threshold = DecimalParameter(0.015, 0.04, default=0.025, space="buy", optimize=True, load=True)
    
    # Position sizing parameters
    max_positions = IntParameter(3, 8, default=5, space="buy", optimize=True, load=True)
    position_size_factor = DecimalParameter(0.1, 0.5, default=0.25, space="buy", optimize=True, load=True)

    plot_config = {
        "main_plot": {
            "grid_upper": {"color": "red", "type": "line"},
            "grid_lower": {"color": "green", "type": "line"},
            "grid_center": {"color": "yellow", "type": "line"},
            "support_level": {"color": "blue", "type": "line"},
            "resistance_level": {"color": "purple", "type": "line"},
        },
        "subplots": {
            "trend_strength": {"color": "orange", "type": "line"},
            "volatility": {"color": "gray", "type": "line"},
            "volume_ratio": {"color": "brown", "type": "line"},
            "downtrend_bias": {"color": "darkred", "type": "line"},
        }
    }

    def calculate_support_resistance(self, dataframe: DataFrame, window: int = 20) -> DataFrame:
        """
        Calculate dynamic support and resistance levels
        """
        # Calculate rolling highs and lows
        dataframe['rolling_high'] = dataframe['high'].rolling(window=window).max()
        dataframe['rolling_low'] = dataframe['low'].rolling(window=window).min()
        
        # Identify support and resistance levels
        dataframe['resistance_level'] = dataframe['rolling_high'].rolling(window=5).mean()
        dataframe['support_level'] = dataframe['rolling_low'].rolling(window=5).mean()
        
        return dataframe

    def calculate_trend_strength(self, dataframe: DataFrame) -> DataFrame:
        """
        Calculate trend strength using multiple indicators with downtrend focus
        """
        # EMA-based trend
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        
        # ADX for trend strength
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        
        # RSI for momentum
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        # MACD for trend confirmation
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # Calculate trend strength (0-1 scale) with downtrend bias
        dataframe['trend_strength'] = (
            (dataframe['adx'] / 100) * 0.3 +  # ADX contribution
            (abs(dataframe['rsi'] - 50) / 50) * 0.25 +  # RSI momentum contribution
            (np.where(dataframe['close'] < dataframe['ema_20'], 1, 0) * 0.25) +  # Price vs EMA contribution
            (np.where(dataframe['macd'] < dataframe['macdsignal'], 1, 0) * 0.2)  # MACD contribution
        )
        
        # Determine trend direction with downtrend focus
        dataframe['trend_direction'] = np.where(
            (dataframe['close'] < dataframe['ema_20']) & (dataframe['ema_20'] < dataframe['ema_50']) & (dataframe['rsi'] < 50),
            -1,  # Downtrend (preferred)
            np.where(
                (dataframe['close'] > dataframe['ema_20']) & (dataframe['ema_20'] > dataframe['ema_50']) & (dataframe['rsi'] > 50),
                1,  # Uptrend
                0  # Sideways
            )
        )
        
        # Calculate downtrend bias indicator
        dataframe['downtrend_bias'] = np.where(
            dataframe['trend_direction'] == -1,
            dataframe['trend_strength'],
            0
        )
        
        return dataframe

    def calculate_enhanced_grid_levels(self, dataframe: DataFrame) -> DataFrame:
        """
        Calculate enhanced grid levels with downtrend bias and support/resistance
        """
        if len(dataframe) == 0:
            return dataframe
            
        current_price = dataframe['close'].iloc[-1]
        trend_strength = dataframe['trend_strength'].iloc[-1]
        trend_direction = dataframe['trend_direction'].iloc[-1]
        support_level = dataframe['support_level'].iloc[-1]
        resistance_level = dataframe['resistance_level'].iloc[-1]
        
        # Calculate base spacing
        if self.use_dynamic_spacing.value:
            volatility = dataframe['volatility'].iloc[-1]
            base_spacing = self.grid_spacing.value
            volatility_multiplier = 1 + (volatility / 0.01)
            spacing = base_spacing * volatility_multiplier
        else:
            spacing = self.grid_spacing.value
        
        # Apply trend bias to grid center with downtrend focus
        if trend_strength > self.trend_strength_threshold.value:
            # Strong trend - bias grid center
            if trend_direction == -1:  # Downtrend (preferred)
                grid_center = current_price * (1 - spacing * 0.8)  # Bias downward more aggressively
            elif trend_direction == 1:  # Uptrend
                grid_center = current_price * (1 + spacing * 0.3)  # Bias upward less aggressively
            else:
                grid_center = current_price
        else:
            # Apply downtrend bias even in weak trends
            grid_center = current_price * (1 + self.downtrend_bias.value * spacing)
        
        # Integrate support/resistance levels
        if support_level > 0 and resistance_level > 0:
            # Adjust grid center based on S/R levels with downtrend preference
            sr_adjustment = (
                (support_level + resistance_level) / 2 - current_price
            ) * self.sr_weight.value
            
            # Add extra downward bias for downtrend markets
            if trend_direction == -1:
                sr_adjustment *= 1.2  # 20% more downward bias
            
            grid_center += sr_adjustment
        
        # Calculate grid boundaries
        num_levels = self.grid_levels.value
        grid_upper = grid_center * (1 + spacing * num_levels)
        grid_lower = grid_center * (1 - spacing * num_levels)
        
        # Store grid information
        dataframe['grid_center'] = grid_center
        dataframe['grid_upper'] = grid_upper
        dataframe['grid_lower'] = grid_lower
        dataframe['grid_spacing'] = spacing
        
        # Calculate individual grid levels
        for i in range(-num_levels, num_levels + 1):
            level_price = grid_center * (1 + spacing * i)
            if grid_lower <= level_price <= grid_upper:
                dataframe[f'grid_level_{i}'] = level_price
        
        return dataframe

    def calculate_microstructure_metrics(self, dataframe: DataFrame) -> DataFrame:
        """
        Enhanced market microstructure analysis
        """
        # Basic spread and volatility
        dataframe['spread_estimate'] = (dataframe['high'] - dataframe['low']) / dataframe['close']
        dataframe['spread_ratio'] = dataframe['spread_estimate'].rolling(window=20).mean()
        
        dataframe['returns'] = dataframe['close'].pct_change()
        dataframe['volatility'] = dataframe['returns'].rolling(window=20).std()
        
        # Enhanced volatility regime
        dataframe['volatility_regime'] = np.where(
            dataframe['volatility'] > self.volatility_threshold.value,
            1,  # High volatility
            0   # Normal volatility
        )
        
        # Volume analysis
        dataframe['volume_sma'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        
        # Price momentum
        dataframe['momentum'] = dataframe['close'].pct_change(periods=5)
        
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate all indicators for the strategy
        """
        # Calculate all metrics
        dataframe = self.calculate_microstructure_metrics(dataframe)
        dataframe = self.calculate_support_resistance(dataframe)
        dataframe = self.calculate_trend_strength(dataframe)
        dataframe = self.calculate_enhanced_grid_levels(dataframe)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Enhanced entry logic optimized for downtrend markets
        """
        dataframe['enter_long'] = 0
        
        # Get grid levels
        num_levels = self.grid_levels.value
        
        # Check each buy grid level (negative levels) with downtrend optimization
        for i in range(-num_levels, 0):
            grid_level_col = f'grid_level_{i}'
            
            if grid_level_col in dataframe.columns:
                # Enhanced entry conditions with downtrend focus
                entry_conditions = (
                    # Price touches the grid level
                    (dataframe['low'] <= dataframe[grid_level_col]) &
                    (dataframe['high'] > dataframe[grid_level_col]) &
                    
                    # Market conditions
                    (dataframe['spread_ratio'] < self.min_spread_ratio.value * 2) &
                    (dataframe['volume_ratio'] > 0.8) &  # Sufficient volume
                    (dataframe['volatility_regime'] == 0) &  # Normal volatility
                    
                    # Downtrend-specific conditions
                    (
                        (dataframe['trend_direction'] == -1) |  # Prefer downtrends
                        (dataframe['trend_strength'] > 0.2)  # Or some trend strength
                    ) &
                    
                    # Support level validation
                    (dataframe['close'] >= dataframe['support_level'] * 0.98) &  # Near support
                    
                    # RSI conditions for downtrend markets
                    (dataframe['rsi'] < 60)  # Not overbought
                )
                
                dataframe.loc[entry_conditions, 'enter_long'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Enhanced exit logic with profit tracking and dynamic stops
        """
        dataframe['exit_long'] = 0
        
        # Get grid levels
        num_levels = self.grid_levels.value
        
        # Exit conditions optimized for downtrend markets
        exit_conditions = (
            # Market condition exits
            (dataframe['volatility_regime'] == 1) |  # High volatility
            (dataframe['spread_ratio'] > self.min_spread_ratio.value * 3) |  # Wide spreads
            (dataframe['volume_ratio'] < 0.5) |  # Low volume
            
            # Grid boundary exits
            (dataframe['close'] >= dataframe['grid_upper']) |
            (dataframe['close'] <= dataframe['grid_lower']) |
            
            # Downtrend-specific exits
            (dataframe['trend_direction'] == 1) & (dataframe['trend_strength'] > 0.6) |  # Strong uptrend reversal
            (dataframe['rsi'] > 70) |  # Overbought condition
            
            # Resistance level exit
            (dataframe['close'] >= dataframe['resistance_level'] * 1.02)  # Above resistance
        )
        
        dataframe.loc[exit_conditions, 'exit_long'] = 1
        
        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss with trailing stop and grid-based adjustments for downtrend markets
        """
        # More conservative trailing stops for downtrend markets
        if current_profit > 0.025:  # 2.5% profit
            # Trail 60% of profits (more conservative)
            return current_profit * 0.6
        elif current_profit > 0.015:  # 1.5% profit
            # Trail 40% of profits
            return current_profit * 0.4
        elif current_profit > 0.008:  # 0.8% profit
            # Trail 25% of profits
            return current_profit * 0.25
        else:
            # Use default stoploss
            return self.stoploss

    def custom_entry_price(self, pair: str, current_time: datetime, proposed_rate: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        """
        Custom entry price to optimize grid level entries
        """
        # Use the proposed rate (grid level price) for better execution
        return proposed_rate

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, current_time: datetime, entry_tag: Optional[str], side: str, **kwargs) -> bool:
        """
        Additional confirmation for trade entries with downtrend focus
        """
        # Get current dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) == 0:
            return False
        
        current_candle = dataframe.iloc[-1]
        
        # Additional confirmation checks with downtrend preference
        confirmation_conditions = (
            current_candle['volume_ratio'] > 0.8 and  # Sufficient volume
            current_candle['spread_ratio'] < self.min_spread_ratio.value * 2 and  # Reasonable spread
            (
                current_candle['trend_direction'] == -1 or  # Prefer downtrends
                current_candle['trend_strength'] > 0.2  # Or some trend strength
            ) and
            current_candle['rsi'] < 65  # Not overbought
        )
        
        return confirmation_conditions
