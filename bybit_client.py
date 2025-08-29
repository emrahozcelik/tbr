"""
Bybit API istemcisi.
V5 API'yi kullanarak market verilerini çeker.
"""

import os
import time
import hmac
import hashlib
import pandas as pd
import requests
from datetime import datetime, timedelta
from loguru import logger
from typing import Dict, Any, Optional, List
from exceptions import APIConnectionError
# from utils import normalize_timeframe, to_bybit_kline_interval, to_bybit_oi_interval # KALDIRILDI

class BybitClient:
    """Bybit API istemcisi."""
    
    def __init__(self) -> None:
        """Bybit API istemcisini başlatır."""
        self.api_key: str = os.getenv("API_KEY", "")
        self.api_secret: str = os.getenv("API_SECRET", "")
        self.base_url: str = "https://api.bybit.com"
        
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """
        Bybit API için HMAC imzası oluşturur.
        
        Args:
            params (dict): İmzalanacak parametreler
            
        Returns:
            str: HMAC imzası
        """
        # Parametreleri sırala ve birleştir
        sorted_params = sorted(params.items())
        signature_payload = '&'.join([f'{k}={v}' for k, v in sorted_params])
        
        # HMAC imzası oluştur
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            signature_payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def fetch_klines(self, symbol: str, interval: str, limit: int = 220, from_timestamp: Optional[int] = None) -> pd.DataFrame:
        """
        Belirtilen sembol ve zaman dilimi için mum verilerini çeker.
        
        Args:
            symbol (str): Kripto sembolü (örn. "BTCUSDT")
            interval (str): Mum zaman dilimi (örn. "60" - 1 saat, "D" - 1 gün)
            limit (int): Çekilecek maksimum mum sayısı
            from_timestamp (int, optional): Bu timestamp'ten önceki mumları çeker
            
        Returns:
            pd.DataFrame: Mum verileri DataFrame'i
        """
        try:
            # Bybit API'nin beklediği interval formatına dönüştür (SSoT canonical -> Bybit)
            bybit_interval = interval # Dönüştürme kaldırıldı, interval doğrudan kullanılıyor.
            
            # API endpoint
            endpoint = "/v5/market/kline"
            
            # Parametreler
            params: Dict[str, Any] = {
                "category": "linear",
                "symbol": symbol,
                "interval": bybit_interval,
                "limit": limit
            }
            
            # From timestamp parametresi varsa ekle
            if from_timestamp:
                params["end"] = from_timestamp
            
            # API isteği
            url = f"{self.base_url}{endpoint}"
            response = requests.get(url, params=params)
            data = response.json()
            
            # Yanıt kontrolü
            if data["retCode"] != 0:
                raise APIConnectionError(f"Bybit API hatası ({data['retCode']}): {data['retMsg']}")
            
            # Veri yoksa boş DataFrame döndür
            if not data["result"]["list"]:
                logger.warning(f"{symbol} {interval} için veri bulunamadı!")
                return pd.DataFrame()
            
            # Debug: API'den gelen veri sayısını logla
            api_data_count = len(data["result"]["list"])
            logger.debug(f"Bybit API'den {symbol} {interval} için {api_data_count} mum verisi alındı (limit: {limit})")
            
            # Verileri DataFrame'e dönüştür
            candles: List[Dict[str, Any]] = []
            for candle in data["result"]["list"]:
                # Bybit API'den gelen veriler ters sırada (en yeni -> en eski)
                timestamp = int(candle[0])  # Unix timestamp (ms)
                open_price = float(candle[1])
                high_price = float(candle[2])
                low_price = float(candle[3])
                close_price = float(candle[4])
                volume = float(candle[5])
                
                candles.append({
                    "timestamp": pd.Timestamp(timestamp, unit='ms'),
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume
                })
            
            # DataFrame oluştur ve sırala (en eski -> en yeni)
            df = pd.DataFrame(candles)
            df = df.sort_values("timestamp")
            
            # Debug: Final veri sayısını logla
            final_count = len(df)
            if final_count < limit:
                logger.warning(f"VERI EKSİKLİĞİ: {symbol} {interval} - İstenen: {limit}, Alınan: {final_count}")
            else:
                logger.debug(f"{symbol} {interval} için {final_count} mum verisi başarıyla çekildi.")
            
            return df
            
        except requests.exceptions.RequestException as e:
            raise APIConnectionError(f"Bybit API'ye bağlanırken ağ hatası: {e}")
        except Exception as e:
            raise APIConnectionError(f"Veri çekme sırasında beklenmedik hata ({symbol} {interval}): {e}")
    
    def fetch_funding_rate(self, symbol: str) -> Optional[float]:
        """
        Belirtilen sembol için güncel funding oranını çeker.
        
        Args:
            symbol (str): Kripto sembolü (örn. "BTCUSDT")
            
        Returns:
            float: Funding oranı (yüzde olarak)
        """
        try:
            # API endpoint
            endpoint = "/v5/market/funding/history"
            
            # Parametreler
            params = {
                "category": "linear",
                "symbol": symbol,
                "limit": 1
            }
            
            # API isteği
            url = f"{self.base_url}{endpoint}"
            response = requests.get(url, params=params)
            data = response.json()
            
            # Yanıt kontrolü
            if data["retCode"] != 0:
                raise APIConnectionError(f"Bybit API funding oranı hatası ({data['retCode']}): {data['retMsg']}")
            
            # Veri yoksa None döndür
            if not data["result"]["list"]:
                logger.warning(f"{symbol} için funding oranı bulunamadı!")
                return None
            
            # Funding oranını al
            funding_rate = float(data["result"]["list"][0]["fundingRate"])
            
            # Yüzde olarak döndür (örn. 0.0001 -> %0.01)
            return funding_rate * 100
            
        except requests.exceptions.RequestException as e:
            raise APIConnectionError(f"Bybit API'ye bağlanırken ağ hatası (funding): {e}")
        except Exception as e:
            raise APIConnectionError(f"Funding oranı çekme sırasında beklenmedik hata ({symbol}): {e}")
    
    def fetch_ticker(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Belirtilen sembol için 24 saatlik fiyat ve hacim değişimi verilerini çeker.
        
        Args:
            symbol (str): Kripto sembolü (örn. "BTCUSDT")
            
        Returns:
            dict: 24 saatlik fiyat ve hacim değişimi verileri
        """
        try:
            # API endpoint
            endpoint = "/v5/market/tickers"
            
            # Parametreler
            params = {
                "category": "linear",
                "symbol": symbol
            }
            
            # API isteği
            url = f"{self.base_url}{endpoint}"
            response = requests.get(url, params=params)
            data = response.json()
            
            # Yanıt kontrolü
            if data["retCode"] != 0:
                logger.error(f"Bybit API ticker hatası: {data['retMsg']}")
                return None
            
            # Veri yoksa None döndür
            if not data["result"]["list"]:
                logger.warning(f"{symbol} için ticker verisi bulunamadı!")
                return None
            
            # Ticker verilerini al
            ticker_data = data["result"]["list"][0]
            
            # Debug için tüm ticker verilerini logla
            logger.debug(f"Ticker verileri: {ticker_data}")
            
            # 24 saatlik fiyat değişimi hesapla
            price_change_24h = float(ticker_data.get("price24hPcnt", "0")) * 100
            
            # 24 saatlik hacim
            volume_24h = float(ticker_data.get("volume24h", "0"))
            
            # Hacim değişimi hesapla - turnover24h kullan
            turnover_24h = float(ticker_data.get("turnover24h", "0"))
            
            # Hacim değişimi yüzdesi hesapla - API'den doğrudan alamıyoruz
            # Önceki günün hacmini hesaplamak için başka bir API çağrısı gerekebilir
            # Şimdilik sadece mevcut hacmi gösterelim
            
            # Sonuçları döndür
            return {
                "price_change_24h": price_change_24h,
                "volume_24h": volume_24h,
                "turnover_24h": turnover_24h,
                "last_price": float(ticker_data.get("lastPrice", "0"))
            }
            
        except Exception as e:
            logger.error(f"Ticker verisi çekme hatası ({symbol}): {str(e)}")
            return None 

    def get_open_interest(self, symbol: str, interval: str, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Belirtilen sembol için Open Interest verisini çeker.

        Args:
            symbol (str): Kripto sembolü (örn. "BTCUSDT").
            interval (str): Zaman aralığı ('5min', '15min', '30min', '1h', '4h', '1d').
            limit (int): Çekilecek veri sayısı.

        Returns:
            list: Open Interest verilerini içeren liste.
        """
        endpoint = "/v5/market/open-interest"
        # ✅ DÜZELTME: Bybit Open Interest API'si farklı interval formatı kullanıyor
        # Doğru interval mapping
        # SSoT: Canonical -> Bybit OI interval
        bybit_interval = interval # Dönüştürme kaldırıldı

        url = self.base_url + endpoint
        params = {
            "category": "linear",
            "symbol": symbol,
            "intervalTime": bybit_interval,
            "limit": limit
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get("retCode") == 0 and data.get("retMsg") == "OK":
                logger.debug(f"[{symbol} {interval}] Open Interest verisi başarıyla çekildi.")
                # API en yeni veriyi ilk sırada döner, biz en eskiye göre sıralayalım.
                return sorted(data["result"]["list"], key=lambda x: int(x['timestamp']))
            else:
                logger.error(f"Open Interest çekilirken API hatası: {data.get('retMsg')}")
                return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Open Interest çekilirken bağlantı hatası: {e}")
            return []

    def get_public_trades(self, symbol: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        CVD hesaplaması için son herkese açık işlemleri çeker.

        Args:
            symbol (str): Kripto sembolü (örn. "BTCUSDT").
            limit (int): Çekilecek maksimum işlem sayısı (max 1000).

        Returns:
            list: Ham işlem verilerini içeren liste.
        """
        endpoint = "/v5/market/recent-trade"  # DÜZELTME: Doğru endpoint
        url = self.base_url + endpoint
        params = {
            "category": "linear",
            "symbol": symbol,
            "limit": min(limit, 1000)  # API limiti max 1000
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get("retCode") == 0 and data.get("retMsg") == "OK":
                result_list = data.get("result", {}).get("list", [])
                logger.debug(f"[{symbol}] CVD hesaplaması için {len(result_list)} adet ham işlem verisi çekildi.")
                return result_list
            else:
                logger.error(f"Ham işlem verisi çekilirken API hatası: {data.get('retMsg', 'Bilinmeyen hata')}")
                return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Ham işlem verisi çekilirken bağlantı hatası: {e}")
            return []
        except Exception as e:
            logger.error(f"Ham işlem verisi işlenirken beklenmeyen hata: {e}")
            return []