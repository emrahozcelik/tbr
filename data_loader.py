# data_loader.py

import pandas as pd
from loguru import logger
import time
from typing import Optional, Dict, Any

# Varsayılan olarak bybit istemcisini import ediyoruz,
# gelecekte diğer borsalar için de soyutlama yapılabilir.
from bybit_client import BybitClient
from exceptions import APIConnectionError, InvalidDataError

class DataLoader:
    """
    Farklı veri kaynaklarından (borsalar, dosyalar vb.) mum verilerini
    çekmek ve standart bir formatta sunmak için sorumlu olan sınıf.
    """

    def __init__(self, bybit_client: BybitClient):
        """
        DataLoader'ı bir Bybit istemcisi ile başlatır.

        Args:
            bybit_client (BybitClient): Bybit borsası ile iletişim kuracak istemci.
        """
        if not bybit_client:
            raise ValueError("BybitClient örneği sağlanmalıdır.")
        self.bybit_client = bybit_client
        logger.info("DataLoader başlatıldı.")

    def get_candles(self, symbol: str, timeframe: str, limit: int = 30, max_retries: int = 3,
                   backoff_factor: float = 0.5, enforce_min_bars: bool = True,
                   minimum_required_bars: int = 3) -> Optional[pd.DataFrame]:
        """
        Belirtilen sembol ve zaman dilimi için mum verilerini Bybit'ten alır.
        Hata durumunda yeniden deneme mekanizması içerir.

        Args:
            symbol (str): Alınacak sembol (örn. 'BTCUSDT').
            timeframe (str): Zaman dilimi (örn. '15', '240').
            limit (int): Alınacak mum sayısı.
            max_retries (int): Hata durumunda maksimum yeniden deneme sayısı.
            backoff_factor (float): Yeniden denemeler arasındaki bekleme süresi çarpanı.
            enforce_min_bars (bool): Minimum bar sayısını zorunlu kıl.
            minimum_required_bars (int): Gerekli minimum bar sayısı.

        Returns:
            pd.DataFrame: Mum verilerini içeren bir DataFrame veya hata durumunda None.
        """
        for attempt in range(max_retries):
            try:
                log_message = (
                    f"Mum verileri alınıyor: Sembol={symbol}, Zaman Dilimi={timeframe}, "
                    f"Limit={limit} (Deneme {attempt + 1}/{max_retries})"
                )
                logger.debug(log_message)

                df = self.bybit_client.fetch_klines(symbol=symbol, interval=timeframe, limit=limit)

                if df is not None and not df.empty:
                    actual_count = len(df)
                    df.attrs['symbol'] = symbol
                    df.attrs['timeframe'] = timeframe

                    min_required = minimum_required_bars if enforce_min_bars else 1
                    if actual_count < min_required:
                        raise InvalidDataError(f"Analiz için kritik veri eksikliği: {symbol} {timeframe} - Alınan: {actual_count}, Gerekli: {min_required}.")
                    
                    if actual_count < limit:
                        logger.warning(f"VERİ EKSİKLİĞİ TESPİT EDİLDİ: {symbol} {timeframe} - İstenen: {limit}, Alınan: {actual_count}")
                    
                    logger.info(f"Başarıyla {actual_count} adet mum verisi alındı: Sembol={symbol}, Zaman Dilimi={timeframe}")
                    return df
                else:
                    # API'den boş ama hatasız bir yanıt gelmesi durumu
                    raise InvalidDataError(f"Veri bulunamadı (API boş yanıt döndü): Sembol={symbol}, Zaman Dilimi={timeframe}")

            except APIConnectionError as e:
                logger.warning(f"Veri bağlantı hatası (Deneme {attempt + 1}/{max_retries}): {e}")
                if attempt >= max_retries - 1:
                    logger.critical(f"Maksimum deneme sayısına ulaşıldı. Veri alınamıyor: {symbol}")
                    raise  # Orijinal APIConnectionError'ı yeniden fırlat
                
                sleep_time = backoff_factor * (2 ** attempt)
                logger.info(f"{sleep_time} saniye sonra yeniden denenecek...")
                time.sleep(sleep_time)

            except Exception as e:
                # Diğer beklenmedik hatalar için
                logger.error(f"Mum verileri alınırken beklenmedik bir hata oluştu: {e}", exc_info=True)
                raise InvalidDataError(f"Veri işlenirken beklenmedik hata: {e}")

        # Döngüden bir şekilde çıkılırsa (normalde olmamalı), bu bir hata durumudur.
        raise RuntimeError(f"Veri alma döngüsü beklenmedik bir şekilde sonlandı: {symbol}")

    def get_historical_data_for_backtest(self, symbols: list[str], timeframe: str, start_timestamp: int, end_timestamp: int, max_iterations: int = 15) -> Dict[str, pd.DataFrame]:
        """
        Backtesting için belirtilen sembol listesindeki her bir sembol için
        tarih aralığındaki tüm mum verilerini çeker.

        Args:
            symbols (list[str]): Verisi çekilecek sembollerin listesi.
            timeframe (str): Mum zaman dilimi.
            start_timestamp (int): Başlangıç timestamp (ms).
            end_timestamp (int): Bitiş timestamp (ms).
            max_iterations (int): Her sembol için maksimum sayfalama iterasyonu.

        Returns:
            Dict[str, pd.DataFrame]: Sembol adlarını anahtar, DataFrame'leri değer olarak içeren bir sözlük.
        """
        all_symbols_data = {}
        for symbol in symbols:
            logger.info(f"Backtest için geçmiş veri çekiliyor: {symbol} [{timeframe}] (Max iterasyon: {max_iterations})")
            all_chunks = []
            current_end_timestamp = end_timestamp

            for i in range(max_iterations):
                try:
                    logger.debug(f"Iterasyon {i + 1}/{max_iterations}. Parça çekiliyor, bitiş: {pd.to_datetime(current_end_timestamp, unit='ms')}")
                    df_chunk = self.bybit_client.fetch_klines(
                        symbol=symbol,
                        interval=timeframe,
                        limit=1000,
                        from_timestamp=current_end_timestamp
                    )
                    if df_chunk is None or df_chunk.empty:
                        logger.info(f"[{symbol}] API daha fazla veri döndürmedi. Veri çekme tamamlandı.")
                        break
                    all_chunks.append(df_chunk)
                    oldest_timestamp_in_chunk = int(df_chunk.iloc[0]['timestamp'].value / 1_000_000)
                    if oldest_timestamp_in_chunk <= start_timestamp:
                        logger.info(f"[{symbol}] İstenen tarih aralığının başlangıcına ulaşıldı.")
                        break
                    current_end_timestamp = oldest_timestamp_in_chunk
                except Exception as e:
                    logger.error(f"[{symbol}] Geçmiş veri çekilirken hata (Iterasyon {i + 1}): {e}", exc_info=True)
                    break
            else:
                logger.warning(f"[{symbol}] Maksimum iterasyon sayısına ({max_iterations}) ulaşıldı. Veri eksik olabilir.")

            if not all_chunks:
                logger.error(f"[{symbol}] Belirtilen aralık için hiçbir veri çekilemedi.")
                all_symbols_data[symbol] = pd.DataFrame() # Boş DF ekle
                continue

            all_chunks.reverse()
            final_df = pd.concat(all_chunks, ignore_index=True)
            final_df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
            final_df.sort_values('timestamp', inplace=True, ignore_index=True)
            final_df = final_df[final_df['timestamp'] >= pd.to_datetime(start_timestamp, unit='ms')]
            final_df['symbol'] = symbol # Sembol adını sütun olarak ekle
            
            logger.info(f"[{symbol}] Toplam {len(final_df)} adet benzersiz geçmiş veri çekildi.")
            all_symbols_data[symbol] = final_df

        return all_symbols_data

    def get_latest_candle(self, symbol: str, timeframe: str = '15') -> Optional[Dict[str, Any]]:
        """
        Belirtilen sembol için sadece en son mumu çeker.
        """
        # Tek mum amaçlı çağrılarda minimum bar zorunluluğunu devre dışı bırak
        df = self.get_candles(symbol, timeframe, limit=3, enforce_min_bars=False)
        if df is not None and not df.empty:
            return df.iloc[-1].to_dict()
        return None
