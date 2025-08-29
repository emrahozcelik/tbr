# Automaton_TBR/supertrend_analyzer.py
import os
from typing import Dict, Any, Optional
import pandas as pd
from loguru import logger

from data_loader import DataLoader
from schemas import SupertrendAnalysisModel
from custom_supertrend import CustomSupertrend

class SupertrendAnalyzer:
    """
    Supertrend indikatörüne dayalı analizleri gerçekleştirir.
    """
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.timeframe = os.getenv("SUPERTREND_TIMEFRAME", "15")
        self.atr_period = int(os.getenv("SUPERTREND_ATR_PERIOD", "10"))
        self.multiplier = float(os.getenv("SUPERTREND_ATR_MULTIPLIER", "3.0"))
        self.change_atr = os.getenv("SUPERTREND_CHANGE_ATR", "true").lower() == "true"
        
        # Custom Supertrend instance
        self.supertrend = CustomSupertrend(
            atr_period=self.atr_period,
            multiplier=self.multiplier,
            change_atr=self.change_atr
        )
        
        logger.info(f"SupertrendAnalyzer başlatıldı. Zaman aralığı: {self.timeframe}, ATR Periyodu: {self.atr_period}, Çarpan: {self.multiplier}, Change ATR: {self.change_atr}")

    def analyze(self, candles: pd.DataFrame, symbol: str, main_timeframe: str) -> Optional[SupertrendAnalysisModel]:
        """
        Verilen mum verileri üzerinde Supertrend analizini uygular.

        Args:
            candles (pd.DataFrame): Analiz edilecek mum verileri.
            symbol (str): Analiz edilen sembol.
            main_timeframe (str): Ana zaman aralığı (loglama için).

        Returns:
            Optional[SupertrendAnalysisModel]: Analiz sonuçlarını içeren Pydantic modeli veya sinyal yoksa None.
        """
        if candles.empty:
            logger.warning(f"[{symbol}] Supertrend analizi için mum verisi bulunamadı.")
            return None

        try:
            # Custom Supertrend hesaplama (TradingView Pine Script ile uyumlu)
            supertrend_result = self.supertrend.analyze(candles)

            if supertrend_result is None:
                logger.warning(f"[{symbol}] Custom Supertrend hesaplaması başarısız oldu.")
                return None

            # Sonuçları al
            trend_direction = supertrend_result['trend']  # 'Long' or 'Short'
            supertrend_level = supertrend_result['supertrend_level']
            trend_value = supertrend_result['trend_value']  # 1 for Long, -1 for Short
            
            logger.debug(f"[{symbol}] Custom Supertrend: Trend={trend_direction}, Level={supertrend_level:.4f}, Value={trend_value}")

            return SupertrendAnalysisModel(
                symbol=symbol,
                timeframe=self.timeframe,
                trend=trend_direction,
                supertrend_level=supertrend_level
            )

        except Exception as e:
            logger.error(f"[{symbol}] Supertrend analizi sırasında beklenmedik hata: {e}", exc_info=True)
            return None