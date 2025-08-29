# Automaton_TBR/main.py
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd

from dotenv import load_dotenv
from loguru import logger

# --- Gerekli Modül Importları ---
from bybit_client import BybitClient
from data_loader import DataLoader
from exceptions import CriticalAnalysisError
from schemas import FinalSignalModel, TBRAnalysisModel, SupertrendAnalysisModel
from tbr_analyzer import TBRAnalyzer
from supertrend_analyzer import SupertrendAnalyzer
from alert_manager import AlertManager
from stats_tracker import StatsTracker
from telegram_notifier import TelegramNotifier
from utils import format_price_standard


def setup_logging(bot_instance, log_level="INFO"):
    """Loguru için log kurulumunu yapar."""
    logger.remove()
    log_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    logger.add(sys.stderr, level=log_level.upper(), format=log_format)
    logger.add(
        "logs/automaton_tbr_{time:YYYY-MM-DD_HH-mm-ss}.log",
        rotation=bot_instance.should_rotate_log,
        retention=3,
        level=log_level.upper(),
        format=log_format,
        encoding='utf-8'
    )
    logger.info("Loglama kurulumu tamamlandı.")


class SimpleTradingBot:
    """
    TBR ve Supertrend stratejilerini kullanan basitleştirilmiş ticaret botu.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.services: Dict[str, Any] = {}
        self.analyzers: Dict[str, Any] = {}
        self.last_analysis_times: Dict[str, float] = {}
        self._cycle_count = 0
        self.log_rotation_threshold = 120

    def should_rotate_log(self, message, file):
        """Her 120 döngüde bir log rotasyonu yapar."""
        if self._cycle_count > 0 and self._cycle_count % self.log_rotation_threshold == 0:
            # Sadece bir kez döndürmek için, döngünün başında bunu kontrol et
            # Bu fonksiyon her log mesajında çağrıldığı için,
            # rotasyonun sadece döngü sayısının katlarında bir kez tetiklenmesini sağlamalıyız.
            # Bunu basitleştirmek için, rotasyonun ana döngüde manuel olarak yönetilmesi daha iyi olabilir.
            # Şimdilik, bu basit mantıkla devam edelim, ancak ana döngüdeki kontrole geçebiliriz.
            # Geçici bir çözüm olarak, rotasyonun sadece bir kez tetiklendiğinden emin olmak için
            # bir bayrak kullanabiliriz.
            if not getattr(self, '_log_rotated_for_cycle', False):
                self._log_rotated_for_cycle = True
                return True
        elif self._cycle_count % self.log_rotation_threshold != 0:
            self._log_rotated_for_cycle = False
        return False

    def initialize_services(self):
        """Gerekli servisleri ve analizörleri başlatır."""
        logger.info("Servisler ve analizörler başlatılıyor...")
        try:
            # Servisler
            bybit_client = BybitClient()
            self.services['data_loader'] = DataLoader(bybit_client=bybit_client)
            # AlertManager artık config'i doğrudan .env'den okuyacak şekilde güncellendi varsayılıyor
            # veya burada gerekli parametreler tek tek geçirilir. Şimdilik config'i geçelim.
            self.services['alert_manager'] = AlertManager(config=self.config)
            self.services['stats_tracker'] = StatsTracker(
                stats_dir=self.config.get('stats_dir', 'stats'),
                main_timeframe=self.config.get('main_timeframe', '240'),
                alert_manager=self.services['alert_manager']
            )

            # Analizörler artık kendi configlerini .env'den okuyor
            self.analyzers['tbr'] = TBRAnalyzer(data_loader=self.services['data_loader'])
            self.analyzers['supertrend'] = SupertrendAnalyzer(data_loader=self.services['data_loader'])

            logger.info("Basit Trading Bot servisleri başarıyla başlatıldı.")
            
            # Sistem başlangıç mesajı gönder
            self.send_startup_message()
            
        except Exception as e:
            logger.critical(f"Başlatma sırasında kritik hata: {e}", exc_info=True)
            raise

    def send_startup_message(self):
        """Sistem başlangıcında telegram mesajı gönderir."""
        try:
            alert_manager = self.services.get('alert_manager')
            if alert_manager and hasattr(alert_manager, 'telegram_notifier') and alert_manager.telegram_notifier:
                symbols = self.config.get('symbols', [])
                timeframe = self.config.get('main_timeframe', '240')
                analysis_interval = self.config.get('analysis_interval', 60)
                
                alert_manager.telegram_notifier.send_system_startup_message(
                    symbols=symbols,
                    timeframe=timeframe,
                    analysis_interval=analysis_interval
                )
                logger.info("Sistem başlangıç mesajı Telegram'a gönderildi.")
            else:
                logger.warning("Telegram bildirici mevcut değil, başlangıç mesajı gönderilemedi.")
        except Exception as e:
            logger.error(f"Başlangıç mesajı gönderilirken hata: {e}")

    def send_cycle_summary(self, cycle_count: int, symbols: list, duration: float):
        """Döngü özeti mesajı gönderir."""
        try:
            alert_manager = self.services.get('alert_manager')
            stats_tracker = self.services.get('stats_tracker')
            
            if alert_manager and hasattr(alert_manager, 'telegram_notifier') and alert_manager.telegram_notifier:
                active_signals_count = len(stats_tracker.get_active_signals_summary()) if stats_tracker else 0
                
                alert_manager.telegram_notifier.send_analysis_cycle_summary(
                    cycle_count=cycle_count,
                    symbols_analyzed=symbols,
                    active_signals_count=active_signals_count,
                    duration=duration
                )
                logger.debug(f"Döngü #{cycle_count} özeti Telegram'a gönderildi.")
        except Exception as e:
            logger.error(f"Döngü özeti gönderilirken hata: {e}")

    def send_shutdown_message(self, reason: str):
        """Sistem kapatılma mesajı gönderir."""
        try:
            alert_manager = self.services.get('alert_manager')
            if alert_manager and hasattr(alert_manager, 'telegram_notifier') and alert_manager.telegram_notifier:
                alert_manager.telegram_notifier.send_system_shutdown_message(reason)
                logger.info("Sistem kapatılma mesajı Telegram'a gönderildi.")
        except Exception as e:
            logger.error(f"Kapatılma mesajı gönderilirken hata: {e}")

    def send_symbol_status(self, symbol: str, status: str, details: str = ""):
        """Sembol analiz durumu mesajı gönderir."""
        try:
            alert_manager = self.services.get('alert_manager')
            if alert_manager and hasattr(alert_manager, 'telegram_notifier') and alert_manager.telegram_notifier:
                alert_manager.telegram_notifier.send_symbol_analysis_status(symbol, status, details)
        except Exception as e:
            logger.error(f"Sembol durum mesajı gönderilirken hata: {e}")

    def send_active_signals_summary(self, active_signals: list):
        """Aktif sinyaller özeti gönderir."""
        try:
            alert_manager = self.services.get('alert_manager')
            if alert_manager and hasattr(alert_manager, 'telegram_notifier') and alert_manager.telegram_notifier:
                alert_manager.telegram_notifier.send_active_signals_summary(active_signals)
                logger.debug("Aktif sinyaller özeti Telegram'a gönderildi.")
        except Exception as e:
            logger.error(f"Aktif sinyaller özeti gönderilirken hata: {e}")

    def run_analysis_cycle(self):
        """Ana analiz döngüsünü çalıştırır."""
        cycle_count = 0
        self._cycle_count = 0
        while True:
            try:
                cycle_count += 1
                self._cycle_count = cycle_count
                
                # Log rotasyon bayrağını her döngü başında sıfırla
                if self._cycle_count % self.log_rotation_threshold != 1:
                    self._log_rotated_for_cycle = False

                loop_start_time = time.time()
                
                logger.info(f"--- Analiz Döngüsü #{cycle_count} Başladı ---")

                # Aktif sinyalleri kontrol et
                self.check_active_signals()

                # Sembolleri analiz et
                symbols = self.config.get('symbols', [])
                for symbol in symbols:
                    self.analyze_symbol(symbol)
                
                # Döngü özeti gönder (her 100 döngüde bir - yaklaşık 1.5 saatte bir)
                if cycle_count % 100 == 0:
                    self.send_cycle_summary(cycle_count, symbols, loop_duration)
                
                loop_duration = time.time() - loop_start_time
                sleep_duration = max(0, self.config.get('analysis_interval', 60) - loop_duration)
                
                logger.info(f"--- Döngü #{cycle_count} Tamamlandı. Süre: {loop_duration:.2f}s, Bekleme: {sleep_duration:.2f}s ---")
                time.sleep(sleep_duration)

            except KeyboardInterrupt:
                logger.warning("Kullanıcı tarafından durduruldu. Sistem kapatılıyor...")
                self.send_shutdown_message("Kullanıcı tarafından manuel durdurma")
                break
            except Exception as e:
                logger.error(f"Ana döngüde beklenmedik hata: {e}", exc_info=True)
                self.services['alert_manager'].send_alert(f"Trading Bot KRİTİK HATA: {e}", "error")
                time.sleep(60)

    def check_active_signals(self):
        """Aktif sinyallerin durumunu kontrol eder."""
        stats_tracker = self.services.get('stats_tracker')
        if not stats_tracker or not stats_tracker.active_signals:
            logger.debug("Takip edilen aktif sinyal bulunmuyor.")
            return

        logger.info(f"🔍 Aktif sinyaller kontrol ediliyor... ({len(stats_tracker.active_signals)} sinyal)")
        data_loader = self.services.get('data_loader')
        active_signals = stats_tracker.get_active_signals_summary()
        active_symbols = list(set(s.get('symbol') for s in active_signals))
        
        current_prices = {}
        for symbol in active_symbols:
            try:
                latest_candle = data_loader.get_latest_candle(symbol, timeframe='15')
                if latest_candle:
                    current_prices[symbol] = float(latest_candle.get('close', 0))
                    logger.debug(f"📊 {symbol}: Güncel fiyat = {current_prices[symbol]}")
                else:
                    logger.warning(f"⚠️ {symbol}: Güncel mum verisi alınamadı")
            except Exception as e:
                logger.error(f"❌ {symbol}: Fiyat alınırken hata - {e}")

        # Aktif sinyallerin takibi için gerekli olan güncel Supertrend verisini de gönderiyoruz.
        # Her sembol için Supertrend verisini toplayalım.
        all_timeframe_data = {}
        for symbol in active_symbols:
            try:
                # Supertrend verisini ana zaman diliminden alalım
                main_timeframe = self.config.get('main_timeframe', '240')
                candles = data_loader.get_candles(symbol, timeframe=main_timeframe, limit=50) # Supertrend için yeterli veri
                if candles is not None and not candles.empty:
                    supertrend_data = self.analyzers['supertrend'].analyze(candles, symbol, main_timeframe)
                    if symbol not in all_timeframe_data:
                        all_timeframe_data[symbol] = {}
                    all_timeframe_data[symbol]['supertrend'] = supertrend_data
                    logger.debug(f"📈 {symbol}: Supertrend verisi alındı")
                else:
                    logger.warning(f"⚠️ {symbol}: Supertrend için mum verisi alınamadı")
            except Exception as e:
                logger.error(f"❌ {symbol}: Supertrend verisi alınırken hata - {e}")

        if current_prices:
            stats_tracker.check_active_signals(current_prices, all_timeframe_data)
        else:
            logger.warning("⚠️ Hiçbir sembol için güncel fiyat alınamadı")
        
        # Aktif sinyallerin özet durumunu gönder (her 50 döngüde bir ve sadece aktif sinyal varsa)
        # Aktif sinyallerin özet durumunu gönder (her 50 döngüde bir ve sadece aktif sinyal varsa)
        active_signals_summary = stats_tracker.get_active_signals_summary(current_prices=current_prices)
        if active_signals_summary:
            if hasattr(self, '_last_active_summary_cycle'):
                if not hasattr(self, '_cycle_count'):
                    self._cycle_count = 0
                if self._cycle_count - self._last_active_summary_cycle >= 50:
                    self.send_active_signals_summary(active_signals_summary)
                    self._last_active_summary_cycle = self._cycle_count
            else:
                self._last_active_summary_cycle = 0

    def analyze_symbol(self, symbol: str):
        """Belirtilen bir sembol için analizleri yürütür."""
        logger.info(f"--- {symbol} için analiz başlıyor ---")
        
        stats_tracker = self.services.get('stats_tracker')
        if stats_tracker.has_active_signal_for_symbol(symbol) or stats_tracker.is_on_cooldown(symbol):
            logger.info(f"[{symbol}] Aktif sinyal veya soğuma periyodu nedeniyle analiz atlanıyor.")
            # Atlanan durumları da bildirme - çok fazla mesaj oluşturuyor
            return

        # Veri Toplama
        data_loader = self.services['data_loader']
        main_timeframe = self.config.get('main_timeframe', '240')
        candles = data_loader.get_candles(symbol, timeframe=main_timeframe, limit=200)

        if candles is None or candles.empty:
            logger.error(f"[{symbol}] Mum verileri alınamadı. Analiz atlanıyor.")
            self.send_symbol_status(symbol, "error", "Mum verileri alınamadı")
            return

        try:
            # TBR Analizi
            tbr_signal: Optional[TBRAnalysisModel] = self.analyzers['tbr'].analyze(candles, symbol, main_timeframe)
            if not tbr_signal:
                logger.debug(f"[{symbol}] TBR analizi sonucu sinyal bulunamadı.")
                # "Sinyal bulunamadı" mesajlarını gönderme - gereksiz spam
                return
            
            logger.info(f"[{symbol}] Potansiyel TBR sinyali bulundu: {tbr_signal.direction} @ {tbr_signal.price}")
            self.send_symbol_status(symbol, "signal_found", f"TBR {tbr_signal.direction} @ {format_price_standard(tbr_signal.price)}")
            
            # Potansiyel sinyal için kısa süreli cooldown başlat (tekrar eden mesajları engellemek için)
            stats_tracker.start_potential_signal_cooldown(symbol, minutes=10)

            # Supertrend Teyidi
            supertrend_signal: Optional[SupertrendAnalysisModel] = self.analyzers['supertrend'].analyze(candles, symbol, main_timeframe)
            if not supertrend_signal or not supertrend_signal.trend:
                logger.warning(f"[{symbol}] Supertrend teyidi için trend bilgisi alınamadı.")
                return

            supertrend_trend = supertrend_signal.trend.lower()
            tbr_direction = tbr_signal.direction.lower()

            is_confirmed = (tbr_direction == 'long' and supertrend_trend == 'long') or \
                           (tbr_direction == 'short' and supertrend_trend == 'short')

            if not is_confirmed:
                logger.info(f"[{symbol}] TBR sinyali Supertrend tarafından teyit edilmedi. (TBR: {tbr_direction}, Supertrend: {supertrend_trend})")
                # Supertrend teyidi olmayan durumları da bildirme - gereksiz spam
                return

            logger.info(f"[{symbol}] Sinyal Supertrend tarafından teyit edildi!")
            self.send_symbol_status(symbol, "confirmed", f"TBR + Supertrend teyidi ({tbr_direction.upper()})")
            
            # Final sinyali oluştur ve gönder
            final_signal = self.create_final_signal(tbr_signal, supertrend_signal)
            if final_signal:
                self.finalize_signal(symbol, final_signal)
            else:
                logger.warning(f"[{symbol}] Geçerli bir SL/TP hesaplanamadığı için nihai sinyal oluşturulamadı.")
                self.send_symbol_status(symbol, "error", "SL/TP hesaplanamadı")

        except Exception as e:
            logger.error(f"[{symbol}] Analiz sırasında hata: {e}", exc_info=True)
            self.send_symbol_status(symbol, "error", f"Analiz hatası: {str(e)[:50]}")
            
    def create_final_signal(self, tbr_signal: TBRAnalysisModel, supertrend_signal: SupertrendAnalysisModel) -> Optional[FinalSignalModel]:
        """
        Backtest'te geliştirilen, Supertrend tabanlı SL ve min/max risk kurallarına göre FinalSignalModel oluşturur.
        Geçerli bir SL bulunamazsa None döndürür.
        """
        entry_price = tbr_signal.entry_price
        direction = tbr_signal.direction.lower()
        
        min_risk_pct = 0.01
        max_risk_pct = 0.022
        buffer_pct = 0.0005
        risk_reward_ratio = 1.5
        
        supertrend_level = supertrend_signal.supertrend_level
        
        # 1. Supertrend Varlığını ve Geçerliliğini Kontrol Et
        if supertrend_level is None or pd.isna(supertrend_level):
            logger.warning(f"[{tbr_signal.symbol}] Supertrend seviyesi mevcut değil (None/NaN). Sinyal iptal.")
            return None

        candidate_sl = (supertrend_level - (entry_price * buffer_pct)) if direction == 'long' else (supertrend_level + (entry_price * buffer_pct))
        
        is_st_valid = (direction == 'long' and candidate_sl < entry_price) or \
                      (direction == 'short' and candidate_sl > entry_price)

        if not is_st_valid:
            logger.warning(f"[{tbr_signal.symbol}] Supertrend seviyesi ({candidate_sl:.5f}) giriş fiyatına ({entry_price:.5f}) göre geçersiz. Sinyal iptal.")
            return None
            
        # 2. Geçerli Supertrend Seviyesini Min/Max Risk Kurallarına Göre Ayarla
        candidate_risk_pct = abs(entry_price - candidate_sl) / entry_price
        final_sl = None
        
        if candidate_risk_pct < min_risk_pct:
            final_sl = entry_price * (1 - min_risk_pct) if direction == 'long' else entry_price * (1 + min_risk_pct)
            strategy_detail = "Min Risk Applied"
        elif candidate_risk_pct > max_risk_pct:
            final_sl = entry_price * (1 - max_risk_pct) if direction == 'long' else entry_price * (1 + max_risk_pct)
            strategy_detail = "Max Risk Applied"
        else:
            final_sl = candidate_sl
            strategy_detail = "Supertrend SL"

        # 3. Nihai Take-Profit ve Risk Hesapla
        final_risk = abs(entry_price - final_sl)
        take_profit = entry_price + (final_risk * risk_reward_ratio) if direction == 'long' else entry_price - (final_risk * risk_reward_ratio)
        
        logger.info(f"[{tbr_signal.symbol}] Geçerli SL/TP hesaplandı. SL: {final_sl:.5f}, TP: {take_profit:.5f}, Risk: {final_risk:.5f}")

        return FinalSignalModel(
            symbol=tbr_signal.symbol,
            direction=tbr_signal.direction,
            primary_entry=entry_price,
            stop_loss=final_sl,
            initial_risk=final_risk,
            tp1=take_profit,
            htf_trend=None,
            strategy_used=f"TBR + Supertrend ({strategy_detail})",
            positive_factors=[
                f"TBR Pattern ({tbr_signal.direction})",
                f"Supertrend ({supertrend_signal.trend})"
            ]
        )

    def finalize_signal(self, symbol: str, final_signal: FinalSignalModel):
        """Geçerli bir sinyali işler, bildirir ve takibe alır."""
        logger.info(f"[{symbol}] GEÇERLİ KURULUM BULUNDU!")
        
        alert_manager = self.services['alert_manager']
        stats_tracker = self.services['stats_tracker']

        alert_message = alert_manager.format_trade_signal_alert(final_signal)
        alert_manager.send_alert(alert_message, "new_signal")

        stats_tracker.record_signal(final_signal)
        logger.info(f"[{symbol}] Sinyal kaydedildi ve takibe alındı.")


if __name__ == "__main__":
    load_dotenv(dotenv_path='.env')
    
    # .env dosyasından ayarları doğrudan yükle
    config = {
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "symbols": os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(","),
        "analysis_interval": int(os.getenv("ANALYSIS_INTERVAL", "60")),
        "main_timeframe": os.getenv("MAIN_TIMEFRAME", "15"),
        "stats_dir": os.getenv("STATS_DIR", "stats"),
        # AlertManager için gerekli ayarlar
        "telegram_enabled": os.getenv("TELEGRAM_ENABLED", "false").lower() == "true",
        "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
    }

    # Önce bot örneğini oluştur
    bot = SimpleTradingBot(config)
    
    # Sonra loglamayı bu örnekle kur
    setup_logging(bot, config.get("log_level", "INFO"))

    try:
        logger.info("--- Basit Trading Bot Başlatılıyor ---")
        
        bot.initialize_services()

        logger.info("--- Ana Analiz Döngüsü Başlatılıyor ---")
        bot.run_analysis_cycle()
        
    except Exception as e:
        logger.critical(f"Uygulama başlatılamadı: {e}", exc_info=True)
        sys.exit(1)
