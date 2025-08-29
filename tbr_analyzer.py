# C:/Users/Emrah/Desktop/Automaton_TBR/tbr_analyzer.py

import pandas as pd
import logging
import numpy as np
from typing import Optional
from pydantic import ValidationError
from schemas import TBRAnalysisModel
import os # os modülünü import ettiğinizden emin olun

logger = logging.getLogger(__name__)

class TBRAnalyzer:
    """
    Three Bar Reversal (TBR) pattern analyzer.
    """

    # === EKSİK OLAN __init__ METODU ===
    def __init__(
        self,
        data_loader, # DataLoader tipini import etmeye gerek yok, kod çalışır.
    ) -> None:
        self.data_loader = data_loader
        self.timeframe = os.getenv("TBR_TIMEFRAME", "15")
        self.confirm_policy = os.getenv("TBR_CONFIRM_POLICY", "strict")
    # ==================================
    
    # Statik metodlarınız varsa burada kalmalı, örn:
    # @staticmethod
    # def _is_bullish_reversal(...): ...

    def analyze(self, candles: pd.DataFrame, symbol: str, timeframe: str) -> Optional[TBRAnalysisModel]:
        """
        Son iki mumu (canlı ve bir önceki kapanmış) analiz ederek "Enhanced" TBR paterni arar.
        Analiz en güncel mumdan başlar. Sinyal bulunursa Pydantic modeli, yoksa None döndürür.
        """
        try:
            if candles is None or candles.empty or len(candles) < 4:
                logger.debug(f"[{symbol}] TBRAnalyzer: Analiz için yetersiz veri (mum sayısı < 4).")
                return None

            o = candles['open'].values
            h = candles['high'].values
            l = candles['low'].values
            c = candles['close'].values

            # Son iki mumu kontrol et: önce canlı mum, sonra bir önceki kapanmış mum.
            # `reversed` sayesinde önce en güncel olan (len(candles) - 1) indeksi kontrol edilir.
            for i in reversed(range(len(candles) - 2, len(candles))):
                
                # Mumun canlı mı yoksa kapanmış mı olduğunu belirle
                candle_type = "ANLIK" if i == len(candles) - 1 else "KAPANMIŞ"

                # === Bearish (Short) Sinyal Kontrolü ===
                is_bearish_pattern = (
                    c[i-2] > o[i-2] and
                    h[i-1] > h[i-2] and
                    l[i-1] > l[i-2] and
                    c[i-1] > o[i-1] and
                    c[i] < o[i] and
                    l[i] < l[i-2]
                )
                
                if is_bearish_pattern:
                    logger.debug(f"[{symbol}] Bearish TBR pattern adayı bulundu ({candle_type} mum). Enhanced koşulu kontrol ediliyor...")
                    if c[i] < l[i-2]:
                        logger.info(f"[{symbol}] {candle_type} Bearish Enhanced TBR sinyali tespit edildi (Mum Zamanı: {candles.index[i]})")
                        signal_data = {
                            'direction': 'short',
                            'price': float(c[i]),
                            'entry_price': float(l[i-2]),
                            'pattern_level': float(l[i-2]),
                            'timestamp': pd.to_datetime(candles.index[i]).to_pydatetime(),
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'pattern_high': float(h[i-2]),
                            'pattern_low': float(l[i-2]),
                        }
                        return TBRAnalysisModel(**signal_data)
                    else:
                        logger.debug(f"[{symbol}] Bearish pattern bulundu ama Enhanced koşulu sağlanmadı (close: {c[i]:.5f}, pattern_low: {l[i-2]:.5f})")

                # === Bullish (Long) Sinyal Kontrolü ===
                is_bullish_pattern = (
                    c[i-2] < o[i-2] and
                    l[i-1] < l[i-2] and
                    h[i-1] < h[i-2] and
                    c[i-1] < o[i-1] and
                    c[i] > o[i] and
                    h[i] > h[i-2]
                )
                
                if is_bullish_pattern:
                    logger.debug(f"[{symbol}] Bullish TBR pattern adayı bulundu ({candle_type} mum). Enhanced koşulu kontrol ediliyor...")
                    if c[i] > h[i-2]:
                        logger.info(f"[{symbol}] {candle_type} Bullish Enhanced TBR sinyali tespit edildi (Mum Zamanı: {candles.index[i]})")
                        signal_data = {
                            'direction': 'long',
                            'price': float(c[i]),
                            'entry_price': float(h[i-2]),
                            'pattern_level': float(h[i-2]),
                            'timestamp': pd.to_datetime(candles.index[i]).to_pydatetime(),
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'pattern_high': float(h[i-2]),
                            'pattern_low': float(l[i-2]),
                        }
                        return TBRAnalysisModel(**signal_data)
                    else:
                        logger.debug(f"[{symbol}] Bullish pattern bulundu ama Enhanced koşulu sağlanmadı (close: {c[i]:.5f}, pattern_high: {h[i-2]:.5f})")

            # Döngü bitti ve sinyal bulunamadı
            logger.debug(f"[{symbol}] Son 2 mumda (canlı ve kapanmış) TBR pattern bulunamadı.")
            return None

        except ValidationError as e:
            logger.error(f"[{symbol}] TBR Pydantic validasyon hatası: {e}")
            return None
        except Exception as e:
            logger.error(f"[{symbol}] TBRAnalyzer içinde beklenmedik bir hata oluştu: {e}", exc_info=True)
            return None
