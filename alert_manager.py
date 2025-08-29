# alert_manager.py
from loguru import logger
from typing import Dict, Any, Optional

from schemas import FinalSignalModel
from utils import format_price_standard

# TelegramNotifier'ı import etmeye çalış, yoksa None olarak ayarla
try:
    from telegram_notifier import TelegramNotifier
except ImportError:
    TelegramNotifier = None


class AlertManager:
    """
    Farklı olay türleri için uyarıları formatlar ve Telegram'a gönderir.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        AlertManager'ı başlatır.

        Args:
            config (Dict[str, Any]): Uyarı ayarlarını içeren konfigürasyon.
        """
        self.config = config
        self.telegram_enabled = config.get('telegram_enabled', False) and TelegramNotifier is not None

        if self.telegram_enabled:
            self.telegram_notifier = TelegramNotifier(
                bot_token=config.get('telegram_bot_token', ''),
                chat_id=config.get('telegram_chat_id', '')
            )
            logger.info("AlertManager başlatıldı - Telegram bildirimleri AKTİF.")
        else:
            self.telegram_notifier = None
            logger.warning("AlertManager başlatıldı - Telegram bildirimleri DEVRE DIŞI.")
            if TelegramNotifier is None:
                logger.warning("-> 'telegram_notifier' modülü bulunamadı.")
            if not config.get('telegram_enabled', False):
                 logger.warning("-> Konfigürasyonda 'telegram_enabled' false olarak ayarlanmış.")

    def send_alert(self, message: str, alert_type: str = "info"):
        """
        Formatlanmış bir mesajı yapılandırılmış kanallara gönderir.

        Args:
            message (str): Gönderilecek mesaj.
            alert_type (str): Uyarının türü (loglama için kullanılır).
        """
        logger.debug(f"Uyarı gönderiliyor (tip: {alert_type}): {message}")

        if self.telegram_enabled and self.telegram_notifier:
            try:
                self.telegram_notifier.send_message(message, parse_mode='Markdown')
                logger.info(f"Telegram'a '{alert_type}' tipinde metin uyarısı gönderildi.")
            except Exception as e:
                logger.error(f"Telegram'a uyarı gönderilirken hata oluştu: {e}")

    def format_trade_signal_alert(self, signal: 'FinalSignalModel') -> str:
        """
        Zenginleştirilmiş FinalSignalModel'den detaylı bir ticaret sinyali bildirimi oluşturur.

        Args:
            signal (FinalSignalModel): Tüm detayları içeren Pydantic sinyal modeli.

        Returns:
            str: Telegram için formatlanmış Markdown bildirim metni.
        """
        try:
            # --- Sinyal Verilerini Doğrudan Modelden Al ---
            symbol = signal.symbol
            direction = signal.direction.upper()
            entry_price = signal.primary_entry
            stop_loss = signal.stop_loss
            tp1 = signal.tp1
            strategy = signal.strategy_used
            
            # --- Mesajı Oluşturma ---
            direction_emoji = "🟢⬆️" if direction == 'LONG' else "🔴⬇️"
            header = f"📊 *{symbol} | {direction_emoji} {direction} * 📊\n"
            separator = "━━━━━━━━━━━━━━━━━━\n"

            # Ana Bilgiler
            trade_info = f"**Strateji:** `{strategy}`\n"
            trade_info += f"**Giriş Fiyatı:** `{format_price_standard(entry_price)}`\n"
            trade_info += f"**Stop Loss:** `{format_price_standard(stop_loss)}`\n"
            trade_info += f"**Take Profit:** `{format_price_standard(tp1)}`\n"

            # Risk/Reward Oranı
            if signal.initial_risk > 0 and tp1:
                reward = abs(tp1 - entry_price)
                risk_reward = reward / signal.initial_risk
                trade_info += f"**Risk/Reward:** `1:{risk_reward:.2f}`\n"
            
            risk_pct = (signal.initial_risk / entry_price) * 100
            trade_info += f"**Risk Yüzdesi:** `{risk_pct:.2f}%`\n"

            # Teyitler
            confirmations = "**Teyitler:**\n"
            if signal.positive_factors:
                for factor in signal.positive_factors:
                    confirmations += f"  ✅ `{factor}`\n"
            else:
                confirmations += "  - Belirtilmemiş\n"

            # Alt Bilgi
            footer = "\n_Lütfen kendi analizinizi yapın. Bu bir yatırım tavsiyesi değildir._"

            # Tüm Parçaları Birleştir
            full_message = (
                header
                + separator
                + trade_info
                + separator
                + confirmations
                + footer
            )
            return full_message

        except Exception as e:
            logger.error(f"Sinyal mesajı formatlanırken hata oluştu: {e}", exc_info=True)
            symbol = getattr(signal, 'symbol', 'N/A')
            direction = getattr(signal, 'direction', 'N/A')
            return f"🚨 HATA: {symbol} için {direction} sinyali formatlanamadı. Lütfen logları kontrol edin."