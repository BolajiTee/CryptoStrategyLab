import sys
import ta
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

from . import tools as ut

class Strategy:
    def __init__(self, params: Dict[str, Any], ohlcv: pd.DataFrame) -> None:
        self.params = params
        self.data = ohlcv.copy()
        
        # Initialize risk management parameters
        self.max_open_positions = params.get('max_open_positions', 3)
        self.max_daily_loss_pct = params.get('max_daily_loss_pct', 2.0)
        self.max_position_size_pct = params.get('max_position_size_pct', 5.0)
        self.trailing_stop_pct = params.get('trailing_stop_pct', 1.0)
        self.break_even_pct = params.get('break_even_pct', 0.5)
        
        # Initialize trading state
        self.open_positions = 0
        self.daily_pnl = 0
        self.daily_trades = 0
        self.last_trade_time = None
        
        self.populate_indicators()
        self.set_trade_mode()
        self.good_to_trade = True
        self.position_was_closed = False
        self.last_position_side = None

    def set_trade_mode(self):
        self.params.setdefault("mode", "both")
        valid_modes = ("long", "short", "both")
        if self.params["mode"] not in valid_modes:
            raise ValueError(f"Wrong strategy mode. Can either be {', '.join(valid_modes)}.")
        
        self.ignore_shorts = self.params["mode"] == "long"
        self.ignore_longs = self.params["mode"] == "short"

    def populate_indicators(self):
        # Trend Indicators
        self.data['ema_20'] = ta.trend.ema_indicator(self.data['close'], window=20)
        self.data['ema_50'] = ta.trend.ema_indicator(self.data['close'], window=50)
        self.data['ema_200'] = ta.trend.ema_indicator(self.data['close'], window=200)
        
        # Momentum Indicators
        self.data['rsi'] = ta.momentum.rsi(self.data['close'], window=14)
        self.data['macd'] = ta.trend.macd_diff(self.data['close'])
        
        # Volatility Indicators
        self.data['atr'] = ta.volatility.average_true_range(
            self.data['high'], 
            self.data['low'], 
            self.data['close'], 
            window=14
        )
        
        # Volume Indicators
        self.data['obv'] = ta.volume.on_balance_volume(self.data['close'], self.data['volume'])
        self.data['volume_sma'] = ta.trend.sma_indicator(self.data['volume'], window=20)
        
        # Custom Indicators
        self.calculate_support_resistance()
        self.calculate_market_regime()

    def calculate_support_resistance(self):
        # Calculate recent highs and lows for support/resistance
        window = 20
        self.data['recent_high'] = self.data['high'].rolling(window=window).max()
        self.data['recent_low'] = self.data['low'].rolling(window=window).min()
        
        # Calculate dynamic support/resistance levels
        self.data['resistance'] = self.data['recent_high'] * 0.995  # 0.5% below recent high
        self.data['support'] = self.data['recent_low'] * 1.005    # 0.5% above recent low

    def calculate_market_regime(self):
        # Determine market regime using multiple indicators
        self.data['trend_strength'] = abs(self.data['ema_20'] - self.data['ema_50']) / self.data['atr']
        self.data['volatility'] = self.data['atr'] / self.data['close']
        
        # Market regime classification
        conditions = [
            (self.data['ema_20'] > self.data['ema_50']) & (self.data['ema_50'] > self.data['ema_200']),
            (self.data['ema_20'] < self.data['ema_50']) & (self.data['ema_50'] < self.data['ema_200']),
            (self.data['trend_strength'] < 0.5) & (self.data['volatility'] < 0.02)
        ]
        choices = ['bullish', 'bearish', 'ranging']
        self.data['market_regime'] = np.select(conditions, choices, default='neutral')

    def check_risk_management(self, row: pd.Series) -> bool:
        # Check if we've hit daily loss limit
        if self.daily_pnl < -self.initial_balance * (self.max_daily_loss_pct / 100):
            return False
            
        # Check if we've hit max positions
        if self.open_positions >= self.max_open_positions:
            return False
            
        # Check if we're in a high volatility period
        if row['volatility'] > 0.05:  # 5% volatility threshold
            return False
            
        return True

    def calculate_position_size(self, balance: float, price: float, stop_loss: float) -> float:
        # Calculate position size based on risk parameters
        risk_amount = balance * (self.max_position_size_pct / 100)
        price_risk = abs(price - stop_loss)
        position_size = risk_amount / price_risk
        
        # Adjust position size based on market regime
        if self.data['market_regime'].iloc[-1] == 'ranging':
            position_size *= 0.5  # Reduce position size in ranging markets
            
        return position_size

    def populate_long_signals(self):
        # Entry conditions for long positions
        self.data['long_entry'] = (
            (self.data['close'] > self.data['ema_20']) &  # Price above short-term EMA
            (self.data['ema_20'] > self.data['ema_50']) &  # Uptrend
            (self.data['rsi'] < 70) &  # Not overbought
            (self.data['macd'] > 0) &  # Positive MACD
            (self.data['volume'] > self.data['volume_sma']) &  # Above average volume
            (self.data['close'] > self.data['support'])  # Above support
        )
        
        # Exit conditions for long positions
        self.data['long_exit'] = (
            (self.data['close'] < self.data['ema_20']) |  # Price below short-term EMA
            (self.data['rsi'] > 80) |  # Overbought
            (self.data['macd'] < 0)  # Negative MACD
        )

    def populate_short_signals(self):
        # Entry conditions for short positions
        self.data['short_entry'] = (
            (self.data['close'] < self.data['ema_20']) &  # Price below short-term EMA
            (self.data['ema_20'] < self.data['ema_50']) &  # Downtrend
            (self.data['rsi'] > 30) &  # Not oversold
            (self.data['macd'] < 0) &  # Negative MACD
            (self.data['volume'] > self.data['volume_sma']) &  # Above average volume
            (self.data['close'] < self.data['resistance'])  # Below resistance
        )
        
        # Exit conditions for short positions
        self.data['short_exit'] = (
            (self.data['close'] > self.data['ema_20']) |  # Price above short-term EMA
            (self.data['rsi'] < 20) |  # Oversold
            (self.data['macd'] > 0)  # Positive MACD
        )

    def calculate_long_sl_price(self, entry_price: float, row: pd.Series) -> float:
        # Dynamic stop loss based on ATR
        atr_multiplier = 2.0
        return entry_price - (row['atr'] * atr_multiplier)

    def calculate_short_sl_price(self, entry_price: float, row: pd.Series) -> float:
        # Dynamic stop loss based on ATR
        atr_multiplier = 2.0
        return entry_price + (row['atr'] * atr_multiplier)

    def evaluate_orders(self, time: datetime, row: pd.Series):
        # Update daily tracking
        if self.last_trade_time is None or (time - self.last_trade_time).days >= 1:
            self.daily_pnl = 0
            self.daily_trades = 0
            self.last_trade_time = time

        # Check risk management conditions
        if not self.check_risk_management(row):
            return

        # Handle existing positions
        if self.position.side == "long":
            # Check for trailing stop
            if row['high'] > self.position.open_price * (1 + self.break_even_pct/100):
                new_sl = row['high'] * (1 - self.trailing_stop_pct/100)
                if new_sl > self.position.sl_price:
                    self.position.sl_price = new_sl

            if self.position.check_for_sl(row):
                self.close_trade(time, self.position.sl_price, "SL long")
                self.daily_pnl += self.position.net_pnl
                self.daily_trades += 1
                self.open_positions -= 1
                
            elif row["long_exit"]:
                self.close_trade(time, row['close'], "Exit long")
                self.daily_pnl += self.position.net_pnl
                self.daily_trades += 1
                self.open_positions -= 1

        elif self.position.side == "short":
            # Check for trailing stop
            if row['low'] < self.position.open_price * (1 - self.break_even_pct/100):
                new_sl = row['low'] * (1 + self.trailing_stop_pct/100)
                if new_sl < self.position.sl_price:
                    self.position.sl_price = new_sl

            if self.position.check_for_sl(row):
                self.close_trade(time, self.position.sl_price, "SL short")
                self.daily_pnl += self.position.net_pnl
                self.daily_trades += 1
                self.open_positions -= 1
                
            elif row["short_exit"]:
                self.close_trade(time, row['close'], "Exit short")
                self.daily_pnl += self.position.net_pnl
                self.daily_trades += 1
                self.open_positions -= 1

        # Evaluate new positions
        if self.position.side is None:
            if not self.ignore_longs and row["long_entry"]:
                sl_price = self.calculate_long_sl_price(row['close'], row)
                position_size = self.calculate_position_size(self.balance, row['close'], sl_price)
                self.balance -= position_size
                self.position.open(time, "long", position_size, row['close'], "Open long", sl_price)
                self.open_positions += 1
                
            elif not self.ignore_shorts and row["short_entry"]:
                sl_price = self.calculate_short_sl_price(row['close'], row)
                position_size = self.calculate_position_size(self.balance, row['close'], sl_price)
                self.balance -= position_size
                self.position.open(time, "short", position_size, row['close'], "Open short", sl_price)
                self.open_positions += 1

    def close_trade(self, time: datetime, price: float, reason: str):
        self.position.close(time, price, reason)
        open_balance = self.balance
        self.balance += self.position.initial_margin + self.position.net_pnl
        trade_info = self.position.info()
        trade_info["open_balance"] = open_balance
        trade_info["close_balance"] = self.balance
        trade_info["market_regime"] = self.data['market_regime'].iloc[-1]
        self.trades_info.append(trade_info)

    def run_backtest(self, initial_balance: float = 1000, leverage: float = 1, 
                    open_fee_rate: float = 0.001, close_fee_rate: float = 0.001):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = ut.Position(leverage=leverage, 
                                  open_fee_rate=open_fee_rate, 
                                  close_fee_rate=close_fee_rate)
        
        self.equity_update_interval = pd.Timedelta(hours=1)
        self.previous_equity_update_time = datetime(1900, 1, 1)
        self.trades_info = []
        self.equity_record = []

        for time, row in self.data.iterrows():
            self.evaluate_orders(time, row)
            self.previous_equity_update_time = ut.update_equity_record(
                time,
                self.position,
                self.balance,
                row["close"],
                self.previous_equity_update_time,
                self.equity_update_interval,
                self.equity_record
            )

        self.trades_info = pd.DataFrame(self.trades_info)
        self.equity_record = pd.DataFrame(self.equity_record).set_index("time")
        self.final_equity = round(self.equity_record.iloc[-1]["equity"], 2)

    def save_equity_record(self, path: str):
        self.equity_record.to_csv(path+'_equity_record.csv', header=True, index=True)

    def save_trades_info(self, path: str):
        self.trades_info.to_csv(path+'_trades_info.csv', header=True, index=True) 