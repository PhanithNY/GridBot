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


class GridBotSample(IStrategy):
    """
    Grid Trading Bot Strategy
    
    This strategy implements a grid trading approach that:
    1. Creates a grid of price levels around the current price
    2. Places buy orders at lower levels and sell orders at higher levels
    3. Uses market microstructure analysis to optimize grid placement
    4. Adapts to market volatility for dynamic grid spacing
    """

    INTERFACE_VERSION = 3

    stoploss = -0.10

    timeframe = "15m"

    # Grid Bot Parameters
    # This parameter determines the percentage distance between each grid level
    # For example, if grid_spacing = 0.005 (0.5%):
    # - If current price is $100
    # - Grid levels might be: $99.50, $100, $100.50, $101, $101.50
    # - Each level is 0.5% apart from the next
    # - 0.001 is 0.1%: Will create more frequent trades
    # - 0.05 is 5%: Will create less frequent trades
    # - 0.005 is 0.5%: Default value, balance between frequent and infrequent trades
    # We use this these values (0.1% & 5%) for testing purposes. These values are not optimized for trading fee.
    grid_spacing = DecimalParameter(0.001, 0.05, default=0.005, space="buy", optimize=True, load=True)

    # Minimum volume threshold for considering a market active
    # This parameter determines the minimum volume required to consider a market active
    # - 1000: Basic threshold for most pairs
    # - 10000: Higher threshold for more liquid pairs
    # - 100000: Very high threshold for major pairs
    # The actual value should be adjusted based on the specific pair's typical volume
    min_volume_threshold = IntParameter(1000, 100000, default=10000, space="buy", optimize=True, load=True)

    # This parameter determines the number of grid levels to create
    # - 3 is 3 levels: $99.50, $100, $100.50
    # - 10 is 10 levels: $99.50, $100, $100.50, $101, $101.50, $102, $102.50, $103, $103.50, $104, $104.50
    # - 5 is 5 levels: $99.50, $100, $100.50, $101, $101.50
    # We use this these values (3 & 10) for testing purposes. These values are not optimized for trading fee.
    grid_levels = IntParameter(3, 15, default=7, space="buy", optimize=True, load=True)

    # This parameter determines if the grid spacing should be dynamic
    # - True: Will adjust the grid spacing based on market volatility
    # - False: Will use a fixed grid spacing
    # Good for trading in low/high volatility markets and prevent overtrading.
    # For stable markets, we don't need to use this.
    # We use this these values (True & False) for testing purposes. These values are not optimized for trading fee.
    use_dynamic_spacing = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    
    # Market Microstructure Parameters
    # This is profit margin for the strategy. See https://trello.com/c/xUiUIDCd for more details.
    # # Example 1: Stock at $100
    # High: $100.50
    # Low: $99.50
    # Close: $100

    # Spread Ratio = ($100.50 - $99.50) / $100
    #             = $1 / $100
    #             = 0.01 (1%)

    # Example 2: Stock at $100
    # High: $100.20
    # Low: $99.80
    # Close: $100

    # Spread Ratio = ($100.20 - $99.80) / $100
    #             = $0.40 / $100
    #             = 0.004 (0.4%)

    min_spread_ratio = DecimalParameter(0.003, 0.01, default=0.005, space="buy", optimize=True, load=True)

    # This parameter determines the volatility threshold for the strategy.
    # Use bigger than 5% for high volatility markets. This will help in backtesting.
    volatility_threshold = DecimalParameter(0.01, 0.05, default=0.02, space="buy", optimize=True, load=True)

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "main_plot": {
            "grid_upper": {"color": "red", "type": "line"},
            "grid_lower": {"color": "green", "type": "line"},
            "support": {"color": "blue", "type": "line"},
            "resistance": {"color": "purple", "type": "line"},
            "grid_level_-7": {"color": "lightgreen", "type": "line"},
            "grid_level_-6": {"color": "lightgreen", "type": "line"},
            "grid_level_-5": {"color": "lightgreen", "type": "line"},
            "grid_level_-4": {"color": "lightgreen", "type": "line"},
            "grid_level_-3": {"color": "lightgreen", "type": "line"},
            "grid_level_-2": {"color": "lightgreen", "type": "line"},
            "grid_level_-1": {"color": "lightgreen", "type": "line"},
            "grid_level_0": {"color": "yellow", "type": "line"},
            "grid_level_1": {"color": "lightcoral", "type": "line"},
            "grid_level_2": {"color": "lightcoral", "type": "line"},
            "grid_level_3": {"color": "lightcoral", "type": "line"},
            "grid_level_4": {"color": "lightcoral", "type": "line"},
            "grid_level_5": {"color": "lightcoral", "type": "line"},
            "grid_level_6": {"color": "lightcoral", "type": "line"},
            "grid_level_7": {"color": "lightcoral", "type": "line"},
        }
    }

    def informative_pairs(self):
        """
        Define additional informative pairs for the strategy
        Currently not using any additional pairs
        """
        return []

    def calculate_grid_levels(self, current_price: float, volatility: float) -> tuple:
        """
        Calculate grid levels based on current price and volatility
        
        Args:
            current_price: Current market price
            volatility: Current market volatility
            
        Returns:
            tuple: (grid_upper, grid_lower, spacing)
        """
        if self.use_dynamic_spacing.value:
            # Adjust spacing based on volatility
            base_spacing = self.grid_spacing.value
            volatility_multiplier = 1 + (volatility / 0.01)  # Scale volatility impact
            spacing = base_spacing * volatility_multiplier
        else:
            spacing = self.grid_spacing.value

        # Calculate grid boundaries
        num_levels = self.grid_levels.value
        grid_upper = current_price * (1 + spacing * num_levels)
        grid_lower = current_price * (1 - spacing * num_levels)

        return grid_upper, grid_lower, spacing

    def calculate_microstructure_metrics(self, dataframe: DataFrame) -> DataFrame:
        """
        Calculate market microstructure metrics for better grid placement
        
        Args:
            dataframe: OHLCV data
            
        Returns:
            DataFrame with added microstructure metrics
        """
        # Estimate bid-ask spread from OHLC data
        dataframe['spread_estimate'] = (dataframe['high'] - dataframe['low']) / dataframe['close']
        dataframe['spread_ratio'] = dataframe['spread_estimate'].rolling(window=20).mean()
        
        # Calculate price volatility
        dataframe['returns'] = dataframe['close'].pct_change()
        dataframe['volatility'] = dataframe['returns'].rolling(window=20).std()
        
        # Identify volatility regime using volatility_threshold
        dataframe['volatility_regime'] = np.where(
            dataframe['volatility'] > self.volatility_threshold.value,  # Use configured threshold
            1,  # High volatility regime
            0   # Normal volatility regime
        )
        
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate and populate all indicators used in the strategy
        
        Args:
            dataframe: OHLCV data
            metadata: Additional pair metadata
            
        Returns:
            DataFrame with added indicators
        """
        # Calculate market microstructure metrics
        dataframe = self.calculate_microstructure_metrics(dataframe)
        
        # Calculate grid levels if we have data
        if len(dataframe) > 0:
            current_price = dataframe['close'].iloc[-1]
            volatility = dataframe['volatility'].iloc[-1]
            grid_upper, grid_lower, spacing = self.calculate_grid_levels(current_price, volatility)
            
            # Add grid levels to dataframe
            dataframe['grid_upper'] = grid_upper
            dataframe['grid_lower'] = grid_lower
            dataframe['grid_spacing'] = spacing

            # Calculate and store all grid levels
            num_levels = self.grid_levels.value
            for i in range(-num_levels, num_levels + 1):
                if i == 0:
                    level_price = current_price
                else:
                    level_price = current_price * (1 + spacing * i)
                
                # Only store if within grid boundaries
                if grid_lower <= level_price <= grid_upper:
                    dataframe[f'grid_level_{i}'] = level_price

            # Print grid levels for the latest candle
            print(f"\nGrid Levels for {metadata['pair']}:")
            print(f"Current Price: {current_price:.8f}")
            print(f"Grid Spacing: {spacing:.4f}")
            print(f"Grid Upper: {grid_upper:.8f}")
            print(f"Grid Lower: {grid_lower:.8f}")
            
            # Print all levels
            print("\nAll Grid Levels:")
            for i in range(-num_levels, num_levels + 1):
                if i == 0:
                    level_price = current_price
                    level_type = "CURRENT"
                else:
                    level_price = current_price * (1 + spacing * i)
                    level_type = "BUY" if i < 0 else "SELL"
                
                # Only print if within grid boundaries
                if grid_lower <= level_price <= grid_upper:
                    print(f"Level {i}: {level_price:.8f} ({level_type})")

        # Calculate support and resistance levels
        window = 20
        dataframe['support'] = dataframe['low'].rolling(window=window).min()
        dataframe['resistance'] = dataframe['high'].rolling(window=window).max()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define entry conditions for the strategy
        
        Args:
            dataframe: OHLCV data with indicators
            metadata: Additional pair metadata
            
        Returns:
            DataFrame with entry signals
        """
        dataframe.loc[
            (
                # Market microstructure conditions
                (dataframe["spread_ratio"] < self.min_spread_ratio.value * 2)  # Check for reasonable spreads
                & (dataframe["volatility_regime"] == 0)  # Only trade in normal volatility
                & (dataframe["close"] > dataframe["grid_lower"])  # Price within grid bounds
                & (dataframe["close"] < dataframe["grid_upper"])  # Price within grid bounds
                & (dataframe["volume"] >= self.min_volume_threshold.value)  # Ensure sufficient market activity
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define exit conditions for the strategy
        
        Args:
            dataframe: OHLCV data with indicators
            metadata: Additional pair metadata
            
        Returns:
            DataFrame with exit signals
        """
        dataframe.loc[
            (
                # Market microstructure conditions
                (dataframe["volatility_regime"] == 1)  # Exit in high volatility
                | (dataframe["spread_ratio"] > self.min_spread_ratio.value * 3)  # Exit on wide spreads
                | (dataframe["close"] >= dataframe["grid_upper"])  # Exit above grid
                | (dataframe["close"] <= dataframe["grid_lower"])  # Exit below grid
                | (dataframe["volume"] < self.min_volume_threshold.value)  # Exit if insufficient market activity
            ),
            "exit_long",
        ] = 1

        return dataframe
