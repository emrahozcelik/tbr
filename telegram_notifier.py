import os
import requests
from typing import Optional, List, Dict, Any
from loguru import logger
from datetime import datetime
from utils import format_price_standard # Fiyat formatlama için
import time

class TelegramNotifier:
    """Telegram üzerinden bildirimleri yöneten sınıf"""
    
    def __init__(self, bot_token: str, chat_id: str):
        """
        Telegram bildirim sınıfını başlatır
        
        Args:
            bot_token: Telegram Bot API token'ı
            chat_id: Hedef sohbet ID'si
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}/"
        logger.info("Telegram bildirici başlatıldı")
        
    def send_message(self, message: str, parse_mode: Optional[str] = "Markdown") -> bool:
        """
        Telegram'a metin mesajı gönderir
        
        Args:
            message: Gönderilecek mesaj
            parse_mode: Telegram'a gönderilecek parse modu (Markdown, HTML veya None)
            
        Returns:
            bool: Mesaj başarıyla gönderildiyse True
        """
        try:
            url = self.base_url + "sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
            }
            # Sadece None değilse parse_mode'u ekle
            if parse_mode is not None:
                payload["parse_mode"] = parse_mode
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.debug(f"Telegram mesajı gönderildi: {message[:50]}...")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Telegram API isteği hatası: {e}")
            # Yanıt içeriğini logla (hata ayıklama için)
            if e.response is not None:
                 logger.error(f"Telegram API Yanıtı: {e.response.text}")
            return False
        except Exception as e:
            logger.error(f"Telegram mesajı gönderme sırasında beklenmeyen hata: {str(e)}")
            return False

    def send_photo(self, photo_path: str, caption: str, parse_mode: Optional[str] = None) -> bool:
        """
        Telegram'a başlık ile birlikte bir fotoğraf gönderir.

        Args:
            photo_path (str): Gönderilecek fotoğrafın dosya yolu.
            caption (str): Fotoğrafın başlığı (mesaj metni).
            parse_mode (Optional[str]): Başlık için parse modu.

        Returns:
            bool: Fotoğraf başarıyla gönderildiyse True.
        """
        if not os.path.exists(photo_path):
            logger.error(f"Gönderilecek fotoğraf bulunamadı: {photo_path}")
            return False

        # Telegram caption limiti (1024 karakter)
        if len(caption) > 1024:
            caption = caption[:1021] + "..."
            logger.warning("Telegram caption 1024 karakter limitini aştı ve kısaltıldı.")

        try:
            url = self.base_url + "sendPhoto"
            payload = {
                "chat_id": self.chat_id,
                "caption": caption,
            }
            if parse_mode:
                payload["parse_mode"] = parse_mode

            with open(photo_path, 'rb') as photo_file:
                files = {'photo': photo_file}
                response = requests.post(url, params=payload, files=files, timeout=20)
                response.raise_for_status()
            logger.debug(f"Telegram'a grafikli sinyal gönderildi: {caption[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Telegram'a fotoğraf gönderilirken hata: {e}")
            return False

    def send_system_startup_message(self, symbols: List[str], timeframe: str, analysis_interval: int) -> bool:
        """
        Sistem başlangıcında detaylı bilgi mesajı gönderir
        
        Args:
            symbols: Taranacak sembol listesi
            timeframe: Ana zaman dilimi
            analysis_interval: Analiz aralığı (saniye)
            
        Returns:
            bool: Mesaj başarıyla gönderildiyse True
        """
        try:
            current_time = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
            
            message = f"""🚀 *TBR Trading Bot Aktif*

📅 *Başlangıç Zamanı:* {current_time}

⚙️ *Sistem Ayarları:*
• Zaman Dilimi: {timeframe}
• Analiz Aralığı: {analysis_interval} saniye
• Strateji: TBR + Supertrend

📊 *Taranan Semboller ({len(symbols)} adet):*
{chr(10).join([f"• {symbol}" for symbol in symbols])}

✅ *Sistem durumu:* Aktif ve tarama yapıyor
🔍 *Beklenen sinyaller:* TBR pattern + Supertrend teyidi

_Bot şimdi belirlenen sembolleri sürekli tarayacak ve geçerli kurulumları bildirecektir._"""

            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Sistem başlangıç mesajı gönderilirken hata: {e}")
            return False

    def send_analysis_cycle_summary(self, cycle_count: int, symbols_analyzed: List[str], 
                                  active_signals_count: int, duration: float) -> bool:
        """
        Analiz döngüsü özeti mesajı gönderir
        
        Args:
            cycle_count: Döngü numarası
            symbols_analyzed: Analiz edilen semboller
            active_signals_count: Aktif sinyal sayısı
            duration: Döngü süresi
            
        Returns:
            bool: Mesaj başarıyla gönderildiyse True
        """
        try:
            current_time = datetime.now().strftime("%H:%M:%S")
            
            message = f"""📊 *Analiz Döngüsü #{cycle_count}*

⏰ *Zaman:* {current_time}
⚡ *Süre:* {duration:.1f} saniye
🎯 *Aktif Sinyaller:* {active_signals_count}

🔍 *Taranan Semboller:*
{chr(10).join([f"• {symbol}" for symbol in symbols_analyzed])}

_Sistem normal çalışıyor..._"""

            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Döngü özeti mesajı gönderilirken hata: {e}")
            return False

    def send_symbol_analysis_status(self, symbol: str, status: str, details: str = "") -> bool:
        """
        Sembol analiz durumu mesajı gönderir
        
        Args:
            symbol: Sembol adı
            status: Durum (analyzing, signal_found, no_signal, error, skipped)
            details: Ek detaylar
            
        Returns:
            bool: Mesaj başarıyla gönderildiyse True
        """
        try:
            current_time = datetime.now().strftime("%H:%M:%S")
            
            status_icons = {
                "analyzing": "🔍",
                "signal_found": "🎯",
                "no_signal": "❌",
                "error": "⚠️",
                "skipped": "⏭️",
                "confirmed": "✅"
            }
            
            icon = status_icons.get(status, "📊")
            
            if status == "signal_found":
                message = f"""{icon} *{symbol} - SİNYAL BULUNDU!*

⏰ *Zaman:* {current_time}
📈 *Durum:* Potansiyel kurulum tespit edildi
{details}"""
            
            elif status == "confirmed":
                message = f"""{icon} *{symbol} - SİNYAL TEYİT EDİLDİ!*

⏰ *Zaman:* {current_time}
✅ *Durum:* Geçerli kurulum onaylandı
{details}

_Sinyal gönderiliyor ve takibe alınıyor..._"""
            
            elif status == "skipped":
                message = f"""{icon} *{symbol} - Analiz Atlandı*

⏰ *Zaman:* {current_time}
📊 *Sebep:* {details}

_Bu sembol için analiz yapılmadı._"""
            
            elif status == "no_signal":
                message = f"""{icon} *{symbol} - Sinyal Yok*

⏰ *Zaman:* {current_time}
📊 *Durum:* Geçerli kurulum bulunamadı
{details if details else '_TBR pattern veya Supertrend teyidi mevcut değil._'}"""
            
            elif status == "error":
                message = f"""{icon} *{symbol} - Analiz Hatası*

⏰ *Zaman:* {current_time}
⚠️ *Hata:* {details}

_Bu sembol için analiz tamamlanamadı._"""
            
            else:
                message = f"""{icon} *{symbol} - {status.title()}*

⏰ *Zaman:* {current_time}
{details}"""

            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Sembol analiz durumu mesajı gönderilirken hata: {e}")
            return False

    def send_active_signals_summary(self, active_signals: List[Dict[str, Any]]) -> bool:
        """
        Aktif sinyallerin özet durumunu gönderir
        
        Args:
            active_signals: Aktif sinyal listesi
            
        Returns:
            bool: Mesaj başarıyla gönderildiyse True
        """
        try:
            if not active_signals:
                message = """📊 *Aktif Sinyal Durumu*

🎯 *Toplam Aktif Sinyal:* 0

_Şu anda takip edilen aktif sinyal bulunmuyor._"""
            else:
                current_time = datetime.now().strftime("%H:%M:%S")
                
                signal_list = []
                for signal in active_signals:
                    symbol = signal.get('symbol', 'N/A')
                    direction = signal.get('direction', 'N/A')
                # --- ÇÖZÜM: Artık doğru PNL verisi gelecek ---
                pnl_pct = signal.get('pnl_percentage', 0.0) # pnl_percentage'ı kullan
                
                pnl_icon = "🟢" if pnl_pct > 0 else "🔴" if pnl_pct < 0 else "⚪"
                
                signal_list.append(
                    f"• {symbol} ({direction.upper()}) - {pnl_icon} {pnl_pct:+.2f}%"
                )
                
                message = f"""📊 *Aktif Sinyal Durumu*

⏰ *Güncelleme:* {current_time}
🎯 *Toplam Aktif Sinyal:* {len(active_signals)}

{chr(10).join(signal_list)}

_Sinyaller sürekli takip ediliyor..._"""

            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Aktif sinyal özeti mesajı gönderilirken hata: {e}")
            return False

    def send_system_shutdown_message(self, reason: str = "Manuel durdurma") -> bool:
        """
        Sistem kapatılması mesajı gönderir
        
        Args:
            reason: Kapatılma sebebi
            
        Returns:
            bool: Mesaj başarıyla gönderildiyse True
        """
        try:
            current_time = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
            
            message = f"""🛑 *TBR Trading Bot Durduruldu*

📅 *Kapatılma Zamanı:* {current_time}
📋 *Sebep:* {reason}

⚠️ *Sistem durumu:* Pasif
🔍 *Tarama durumu:* Durduruldu

_Bot artık yeni sinyalleri taramayacaktır._"""

            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Sistem kapatılma mesajı gönderilirken hata: {e}")
            return False

