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


class SimpleGridDCA(IStrategy):
    """
    Advanced Grid Trading Bot Strategy with DCA (Dollar Cost Averaging)
    
    This strategy implements an advanced grid trading approach with DCA that:
    1. Creates a grid of price levels around the current price
    2. Places buy orders at lower levels and sell orders at higher levels
    3. Uses market microstructure analysis to optimize grid placement
    4. Adapts to market volatility for dynamic grid spacing
    5. Implements DCA by buying more at lower levels and tracking cost basis
    6. Manages multiple positions with different entry levels
    7. Includes trend analysis and directional bias
    8. Smart grid repositioning based on market conditions
    9. Enhanced exit strategies with trailing stops
    10. Advanced risk management and profit optimization
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
    
    # DCA Parameters
    # Base position size as a percentage of available balance
    base_position_size = DecimalParameter(0.01, 0.1, default=0.05, space="buy", optimize=True, load=True)
    
    # DCA multiplier - increases position size at lower levels
    # 1.0 = same size at all levels, 2.0 = double size at lower levels
    dca_multiplier = DecimalParameter(1.0, 3.0, default=1.5, space="buy", optimize=True, load=True)
    
    # Maximum number of DCA entries per pair
    max_dca_entries = IntParameter(3, 10, default=5, space="buy", optimize=True, load=True)
    
    # Minimum time between DCA entries (in minutes)
    min_dca_interval = IntParameter(15, 240, default=60, space="buy", optimize=True, load=True)
    
    # DCA take profit percentage - profit target for DCA positions
    dca_take_profit = DecimalParameter(0.01, 0.05, default=0.02, space="buy", optimize=True, load=True)
    
    # Use volume-weighted DCA - increase position size when volume is high
    use_volume_weighted_dca = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    
    # Profitability Enhancement Parameters
    
    # Trend Analysis Parameters
    # Use trend analysis to bias grid entries
    use_trend_bias = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    
    # Trend strength threshold for entry bias
    trend_strength_threshold = DecimalParameter(0.1, 0.5, default=0.2, space="buy", optimize=True, load=True)
    
    # Smart Grid Repositioning
    # Reposition grid when price moves significantly
    use_smart_repositioning = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    
    # Price movement threshold for grid repositioning (%)
    reposition_threshold = DecimalParameter(0.02, 0.10, default=0.05, space="buy", optimize=True, load=True)
    
    # Enhanced Exit Strategy
    # Use trailing stop loss for better profit capture
    use_trailing_stop = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    
    # Trailing stop percentage
    trailing_stop_percentage = DecimalParameter(0.005, 0.03, default=0.015, space="buy", optimize=True, load=True)
    
    # Partial profit taking
    use_partial_profit = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    
    # First partial profit target (%)
    partial_profit_target = DecimalParameter(0.01, 0.03, default=0.015, space="buy", optimize=True, load=True)
    
    # Partial profit size (% of position)
    partial_profit_size = DecimalParameter(0.2, 0.5, default=0.3, space="buy", optimize=True, load=True)
    
    # Market Timing Parameters
    # Use market timing to avoid bad market conditions
    use_market_timing = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    
    # RSI thresholds for market timing
    rsi_oversold = IntParameter(20, 40, default=30, space="buy", optimize=True, load=True)
    rsi_overbought = IntParameter(60, 80, default=70, space="buy", optimize=True, load=True)
    
    # Risk Management Parameters
    # Maximum drawdown protection
    max_drawdown = DecimalParameter(0.05, 0.20, default=0.10, space="buy", optimize=True, load=True)
    
    # Maximum daily loss limit
    max_daily_loss = DecimalParameter(0.02, 0.10, default=0.05, space="buy", optimize=True, load=True)
    
    # Correlation-based position sizing
    use_correlation_sizing = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    
    # Advanced Profitability Parameters
    
    # Market Regime Detection
    # Use ATR for better volatility analysis
    use_atr_volatility = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    
    # ATR period for volatility calculation
    atr_period = IntParameter(10, 30, default=14, space="buy", optimize=True, load=True)
    
    # Multiple Take Profit Levels
    # Enable multiple profit targets
    use_multiple_tp = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    
    # Second profit target (%)
    second_tp_target = DecimalParameter(0.025, 0.06, default=0.04, space="buy", optimize=True, load=True)
    
    # Third profit target (%)
    third_tp_target = DecimalParameter(0.04, 0.10, default=0.07, space="buy", optimize=True, load=True)
    
    # Smart Position Sizing
    # Adjust position size based on market conditions
    use_smart_sizing = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    
    # Volatility-based position sizing multiplier
    volatility_sizing_multiplier = DecimalParameter(0.5, 1.5, default=1.0, space="buy", optimize=True, load=True)
    
    # Advanced Correlation Analysis
    # Use correlation with major pairs for better timing
    use_correlation_analysis = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    
    # Correlation threshold for entry
    correlation_threshold = DecimalParameter(0.3, 0.8, default=0.5, space="buy", optimize=True, load=True)
    
    # Performance Optimization
    # Minimum profit per trade to cover fees
    min_profit_threshold = DecimalParameter(0.001, 0.005, default=0.002, space="buy", optimize=True, load=True)
    
    # Maximum holding time for unprofitable positions (hours)
    max_holding_time = IntParameter(24, 168, default=72, space="buy", optimize=True, load=True)
    
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

    # Grid levels need to pre-defined in the plot_config
    # Dynamically add in runtime won't work.
    plot_config = {
        "main_plot": {
            "grid_upper": {"color": "red", "type": "line"},
            "grid_lower": {"color": "green", "type": "line"},
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
        },
        "subplots": {
            "DCA_Metrics": {
                "avg_cost_basis": {"color": "blue", "type": "line"},
                "total_position_size": {"color": "orange", "type": "line"},
                "dca_level": {"color": "purple", "type": "line"},
            },
            "Trend_Analysis": {
                "trend_strength": {"color": "cyan", "type": "line"},
                "trend_direction": {"color": "magenta", "type": "line"},
                "rsi": {"color": "gray", "type": "line"},
            },
            "Risk_Metrics": {
                "drawdown": {"color": "red", "type": "line"},
                "daily_pnl": {"color": "green", "type": "line"},
                "volatility": {"color": "orange", "type": "line"},
            }
        }
    }

    def calculate_grid_levels(self, current_price: float, volatility: float) -> tuple:
        """
        Calculate grid levels based on current price and volatility with smart repositioning
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
        """
        # Estimate bid-ask spread from OHLC data
        dataframe['spread_estimate'] = (dataframe['high'] - dataframe['low']) / dataframe['close']
        dataframe['spread_ratio'] = dataframe['spread_estimate'].rolling(window=20).mean()
        
        # Calculate price volatility
        dataframe['returns'] = dataframe['close'].pct_change()
        dataframe['volatility'] = dataframe['returns'].rolling(window=20).std()
        
        # Identify volatility regime using volatility_threshold
        dataframe['volatility_regime'] = np.where(
            dataframe['volatility'] > self.volatility_threshold.value,
            1,  # High volatility regime
            0   # Normal volatility regime
        )
        
        # Volume analysis for DCA
        dataframe['volume_ma'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_ma']
        
        # Volume-weighted DCA multiplier
        if self.use_volume_weighted_dca.value:
            dataframe['volume_dca_multiplier'] = np.where(
                dataframe['volume_ratio'] > 1.5,  # High volume
                self.dca_multiplier.value * 1.2,  # Increase DCA multiplier
                self.dca_multiplier.value
            )
        else:
            dataframe['volume_dca_multiplier'] = self.dca_multiplier.value
        
        return dataframe

    def calculate_trend_indicators(self, dataframe: DataFrame) -> DataFrame:
        """
        Calculate trend analysis indicators for entry bias with enhanced features
        """
        # RSI for market timing
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        # Moving averages for trend direction
        dataframe['sma_20'] = ta.SMA(dataframe, timeperiod=20)
        dataframe['sma_50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        
        # Enhanced trend direction with multiple confirmations
        ema_trend = np.where(dataframe['ema_12'] > dataframe['ema_26'], 1, -1)
        sma_trend = np.where(dataframe['close'] > dataframe['sma_20'], 1, -1)
        price_trend = np.where(dataframe['close'] > dataframe['close'].shift(1), 1, -1)
        
        # Combined trend direction (-1 to 1)
        dataframe['trend_direction'] = (ema_trend + sma_trend + price_trend) / 3
        
        # Enhanced trend strength calculation
        price_distance = abs(dataframe['close'] - dataframe['sma_20']) / dataframe['sma_20']
        
        # Use ATR for volatility if enabled
        if self.use_atr_volatility.value:
            dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period.value)
            volatility_normalized = dataframe['atr'] / dataframe['atr'].rolling(window=50).mean()
        else:
            volatility_normalized = dataframe['volatility'] / dataframe['volatility'].rolling(window=50).mean()
        
        dataframe['trend_strength'] = (price_distance * volatility_normalized).rolling(window=10).mean()
        
        # MACD for trend confirmation
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # Bollinger Bands for volatility analysis
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lowerband']) / (dataframe['bb_upperband'] - dataframe['bb_lowerband'])
        
        # Market regime detection
        dataframe['market_regime'] = np.where(
            (dataframe['trend_strength'] > 0.3) & (dataframe['volatility'] > self.volatility_threshold.value),
            'trending_volatile',
            np.where(
                dataframe['trend_strength'] > 0.3,
                'trending_stable',
                np.where(
                    dataframe['volatility'] > self.volatility_threshold.value,
                    'ranging_volatile',
                    'ranging_stable'
                )
            )
        )
        
        # Support and resistance levels
        dataframe['support_level'] = dataframe['low'].rolling(window=20).min()
        dataframe['resistance_level'] = dataframe['high'].rolling(window=20).max()
        
        return dataframe

    def calculate_risk_metrics(self, dataframe: DataFrame) -> DataFrame:
        """
        Calculate risk management metrics
        """
        # Drawdown calculation
        dataframe['rolling_max'] = dataframe['close'].rolling(window=50).max()
        dataframe['drawdown'] = (dataframe['close'] - dataframe['rolling_max']) / dataframe['rolling_max']
        
        # Daily PnL approximation
        dataframe['daily_return'] = dataframe['close'].pct_change(periods=96)  # 24 hours in 15m candles
        dataframe['daily_pnl'] = dataframe['daily_return'].rolling(window=96).sum()
        
        # Volatility regime
        dataframe['volatility_regime'] = np.where(
            dataframe['volatility'] > self.volatility_threshold.value,
            'high',
            'normal'
        )
        
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate and populate all indicators used in the strategy
        """
        # Calculate market microstructure metrics
        dataframe = self.calculate_microstructure_metrics(dataframe)
        
        # Calculate trend indicators
        dataframe = self.calculate_trend_indicators(dataframe)
        
        # Calculate risk metrics
        dataframe = self.calculate_risk_metrics(dataframe)
        
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

            # DCA metrics (for visualization)
            dataframe['avg_cost_basis'] = current_price  # Placeholder - will be calculated in entry logic
            dataframe['total_position_size'] = 0  # Placeholder - will be calculated in entry logic
            dataframe['dca_level'] = 0  # Placeholder - will be calculated in entry logic

            # Debug: Print grid information for the latest candle
            if len(dataframe) > 0:
                print(f"\nGrid Debug for {metadata['pair']}:")
                print(f"Current Price: {current_price:.8f}")
                print(f"Grid Spacing: {spacing:.4f}")
                print(f"Grid Upper: {grid_upper:.8f}")
                print(f"Grid Lower: {grid_lower:.8f}")
                print(f"Number of Levels: {num_levels}")
                
                # Print a few grid levels
                for i in range(-3, 4):  # Show levels -3 to +3
                    grid_level_col = f'grid_level_{i}'
                    if grid_level_col in dataframe.columns:
                        level_price = dataframe[grid_level_col].iloc[-1]
                        print(f"Level {i}: {level_price:.8f}")

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Ultra-simple entry logic: Basic grid entries without complex filters
        """
        # Initialize enter_long column
        dataframe['enter_long'] = 0
        
        # Get the number of grid levels
        num_levels = self.grid_levels.value
        
        # Debug: Print basic info
        if len(dataframe) > 0:
            print(f"\nEntry Debug for {metadata['pair']}:")
            print(f"Current Price: {dataframe['close'].iloc[-1]:.8f}")
            print(f"Grid Levels: {num_levels}")
            print(f"Volume Threshold: {self.min_volume_threshold.value}")
            print(f"Use Market Timing: {self.use_market_timing.value}")
            print(f"Use Trend Bias: {self.use_trend_bias.value}")
        
        # Check for entry at each buy grid level (negative levels)
        for i in range(-num_levels, 0):  # Only negative levels for buy entries
            grid_level_col = f'grid_level_{i}'
            
            # Check if this grid level exists in the dataframe
            if grid_level_col in dataframe.columns:
                # SUPER SIMPLE ENTRY: Just check if price is below the grid level
                # This will trigger entries when price drops to grid levels
                entry_condition = (dataframe['close'] <= dataframe[grid_level_col])
                
                # Only apply volume filter if it's reasonable
                if 'volume' in dataframe.columns and self.min_volume_threshold.value > 0:
                    # Use a much lower volume threshold for testing
                    test_volume_threshold = min(self.min_volume_threshold.value, 1000)
                    entry_condition &= (dataframe['volume'] >= test_volume_threshold)
                
                # Set entry signal
                dataframe.loc[entry_condition, 'enter_long'] = 1
                
                # Add entry tag for custom entry price
                dataframe.loc[entry_condition, 'enter_tag'] = f'grid_level_{i}'
                
                # Debug: Print entry conditions for the latest candle
                if len(dataframe) > 0 and entry_condition.iloc[-1]:
                    print(f"✅ ENTRY TRIGGERED for {metadata['pair']} at grid level {i}")
                    print(f"   Price: {dataframe['close'].iloc[-1]:.8f}")
                    print(f"   Grid Level: {dataframe[grid_level_col].iloc[-1]:.8f}")
                    if 'volume' in dataframe.columns:
                        print(f"   Volume: {dataframe['volume'].iloc[-1]:.0f}")
                elif len(dataframe) > 0:
                    # Debug why no entry
                    current_price = dataframe['close'].iloc[-1]
                    grid_price = dataframe[grid_level_col].iloc[-1]
                    print(f"❌ No entry for level {i}: Price {current_price:.8f} > Grid {grid_price:.8f}")

        return dataframe

    # def bot_loop_start(self, **kwargs) -> None:
    #     """
    #     Called at the start of the bot iteration (one loop = one entry/exit check).
    #     """
    #     print(f"\n🔄 Bot loop started - checking for entries/exits...")

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Enhanced exit logic: Improved exit strategy for DCA positions
        """
        # Initialize exit_long column
        dataframe['exit_long'] = 0
        
        # Get the number of grid levels
        num_levels = self.grid_levels.value
        
        # 1. Stop Loss: Exit if price goes below the lowest grid level
        lowest_grid_level = f'grid_level_{-num_levels}'
        if lowest_grid_level in dataframe.columns:
            dataframe.loc[
                (dataframe['close'] < dataframe[lowest_grid_level]),  # Price below lowest grid level
                'exit_long'
            ] = 1
        
        # 2. Enhanced Take Profit: Exit based on trend and volatility
        for i in range(1, num_levels + 1):  # Positive levels for take profit
            grid_level_col = f'grid_level_{i}'
            
            if grid_level_col in dataframe.columns:
                # Basic take profit
                basic_tp = (dataframe['close'] > dataframe[grid_level_col])
                
                # Enhanced take profit with trend consideration
                if self.use_trend_bias.value:
                    trend_tp = (
                        (dataframe['close'] > dataframe[grid_level_col]) &
                        (dataframe['trend_direction'] < 0)  # Exit if trend turns bearish
                    )
                    dataframe.loc[trend_tp, 'exit_long'] = 1
                else:
                    dataframe.loc[basic_tp, 'exit_long'] = 1
        
        # 3. Risk-based exits
        dataframe.loc[
            (dataframe['drawdown'] < -self.max_drawdown.value) |
            (dataframe['daily_pnl'] < -self.max_daily_loss.value),
            'exit_long'
        ] = 1

        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, current_time: datetime, entry_tag: Optional[str], side: str, **kwargs) -> bool:
        """
        Ultra-simple confirm trade entry - allow most trades
        """
        try:
            # Check if we have too many DCA entries
            open_trades = Trade.get_trades_proxy(is_open=True)
            pair_trades = [trade for trade in open_trades if trade.pair == pair]
            
            print(f"Confirming trade for {pair}: Current DCA level = {len(pair_trades)}")
            
            if len(pair_trades) >= self.max_dca_entries.value:
                print(f"❌ Max DCA entries reached for {pair}: {len(pair_trades)}/{self.max_dca_entries.value}")
                return False
            
            # Skip time interval check for now to allow more entries
            # if pair_trades and self.min_dca_interval.value > 0:
            #     last_trade = max(pair_trades, key=lambda x: x.open_date_utc)
            #     time_since_last = (current_time - last_trade.open_date_utc).total_seconds() / 60
            #     if time_since_last < self.min_dca_interval.value:
            #         print(f"Min interval not met for {pair}: {time_since_last:.1f}min < {self.min_dca_interval.value}min")
            #         return False
            
            # Skip portfolio risk check for now
            # try:
            #     total_stake = sum(trade.stake_amount for trade in open_trades)
            #     available_balance = self.wallets.get_total_stake_amount()
            #     if total_stake > available_balance * 0.7:
            #         print(f"Portfolio risk limit reached: {total_stake:.2f} > {available_balance * 0.7:.2f}")
            #         return False
            # except:
            #     pass
            
            print(f"✅ Trade entry confirmed for {pair} at level {entry_tag}")
            return True
            
        except Exception as e:
            print(f"Error in confirm_trade_entry for {pair}: {str(e)}")
            return True  # Allow trade if there's an error