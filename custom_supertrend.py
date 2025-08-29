#!/usr/bin/env python3
"""
Custom Supertrend implementation based on Pine Script
Directly translated from TradingView Pine Script v4
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional

class CustomSupertrend:
    """
    Custom Supertrend implementation matching TradingView Pine Script exactly
    """
    
    def __init__(self, atr_period: int = 10, multiplier: float = 3.0, change_atr: bool = True):
        self.atr_period = atr_period
        self.multiplier = multiplier
        self.change_atr = change_atr
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Calculate ATR (Average True Range)
        """
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        if self.change_atr:
            # Use standard ATR (EMA-based)
            atr = tr.ewm(span=self.atr_period, adjust=False).mean()
        else:
            # Use SMA-based ATR (atr2 in Pine Script)
            atr = tr.rolling(window=self.atr_period).mean()
        
        return atr
    
    def calculate_supertrend(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Calculate Supertrend indicator
        Returns: (supertrend_up, supertrend_down, trend, supertrend_final)
        """
        # Source = hl2 (high + low) / 2
        src = (high + low) / 2
        
        # Calculate ATR
        atr = self.calculate_atr(high, low, close)
        
        # Initialize series
        up = pd.Series(index=close.index, dtype=float)
        dn = pd.Series(index=close.index, dtype=float)
        trend = pd.Series(index=close.index, dtype=int)
        supertrend = pd.Series(index=close.index, dtype=float)
        
        # Calculate initial up and down bands
        up_basic = src - (self.multiplier * atr)
        dn_basic = src + (self.multiplier * atr)
        
        # Initialize first values
        up.iloc[0] = up_basic.iloc[0]
        dn.iloc[0] = dn_basic.iloc[0]
        trend.iloc[0] = 1
        
        # Calculate Supertrend
        for i in range(1, len(close)):
            # Up calculation
            up1 = up.iloc[i-1] if not pd.isna(up.iloc[i-1]) else up_basic.iloc[i]
            up.iloc[i] = max(up_basic.iloc[i], up1) if close.iloc[i-1] > up1 else up_basic.iloc[i]
            
            # Down calculation  
            dn1 = dn.iloc[i-1] if not pd.isna(dn.iloc[i-1]) else dn_basic.iloc[i]
            dn.iloc[i] = min(dn_basic.iloc[i], dn1) if close.iloc[i-1] < dn1 else dn_basic.iloc[i]
            
            # Trend calculation
            prev_trend = trend.iloc[i-1] if not pd.isna(trend.iloc[i-1]) else 1
            
            if prev_trend == -1 and close.iloc[i] > dn.iloc[i-1]:
                trend.iloc[i] = 1
            elif prev_trend == 1 and close.iloc[i] < up.iloc[i-1]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = prev_trend
        
        # Final Supertrend line
        supertrend = np.where(trend == 1, up, dn)
        
        return up, dn, trend, pd.Series(supertrend, index=close.index)
    
    def get_signals(self, trend: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Get buy/sell signals from trend changes
        """
        buy_signal = (trend == 1) & (trend.shift(1) == -1)
        sell_signal = (trend == -1) & (trend.shift(1) == 1)
        
        return buy_signal, sell_signal
    
    def analyze(self, candles: pd.DataFrame) -> Optional[dict]:
        """
        Analyze candles and return Supertrend information
        """
        if candles.empty or len(candles) < self.atr_period + 1:
            return None
        
        try:
            high = candles['high']
            low = candles['low'] 
            close = candles['close']
            
            up, dn, trend, supertrend = self.calculate_supertrend(high, low, close)
            buy_signals, sell_signals = self.get_signals(trend)
            
            # Get latest values
            latest_trend = trend.iloc[-1]
            latest_supertrend = supertrend.iloc[-1]
            latest_buy = buy_signals.iloc[-1] if len(buy_signals) > 0 else False
            latest_sell = sell_signals.iloc[-1] if len(sell_signals) > 0 else False
            
            return {
                'trend': 'Long' if latest_trend == 1 else 'Short',
                'trend_value': int(latest_trend),
                'supertrend_level': float(latest_supertrend),
                'up_level': float(up.iloc[-1]) if latest_trend == 1 else None,
                'down_level': float(dn.iloc[-1]) if latest_trend == -1 else None,
                'buy_signal': bool(latest_buy),
                'sell_signal': bool(latest_sell),
                'full_trend': trend,
                'full_supertrend': supertrend
            }
            
        except Exception as e:
            print(f"Supertrend calculation error: {e}")
            return None

# Test function
def test_supertrend():
    """Test the custom supertrend implementation"""
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='15T')
    
    # Generate sample OHLC data
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    high_prices = close_prices + np.random.rand(100) * 2
    low_prices = close_prices - np.random.rand(100) * 2
    open_prices = close_prices + np.random.randn(100) * 0.5
    
    test_data = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices
    })
    
    # Test Supertrend
    st = CustomSupertrend(atr_period=10, multiplier=3.0)
    result = st.analyze(test_data)
    
    if result:
        print("✅ Custom Supertrend Test Successful")
        print(f"   Trend: {result['trend']}")
        print(f"   Level: {result['supertrend_level']:.4f}")
        print(f"   Buy Signal: {result['buy_signal']}")
        print(f"   Sell Signal: {result['sell_signal']}")
    else:
        print("❌ Custom Supertrend Test Failed")

if __name__ == "__main__":
    test_supertrend()