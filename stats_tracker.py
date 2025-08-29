"""
İstatistik Takip Modülü (StatsTracker)
-------------------------------------
Bu modül, Automaton tarafından üretilen sinyallerin takibi,
işlem sonuçlarının kaydedilmesi ve performans istatistiklerinin tutulmasından sorumludur.
"""

import os
import csv
import json
import shutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from loguru import logger
from schemas import TrackedSignalModel, FinalSignalModel
from utils import format_price_standard

SIGNAL_FIELDS = [
    "signal_id", "symbol", "timeframe", "entry_time", "entry_price", "direction",
    "sl_price", "tp_price", "tp1_price", "tp1_5_price", "tp2_price", "tp3_price",
    "status", "confirmations", "result_time", "result_price", "profit_percentage",
    "exit_r_value", # <-- YENİ ALAN
    "initial_risk", "is_breakeven",
    "volatility_level", "fib_level", "tp_strategy_used", "sl_type",
    "sl_percentage", "tp_percentage", "pattern_name", "regime",
    "initial_pivots", "trailing_activation_price", "highest_price_in_trail", "partial_profit_price",
    "is_partial_profit_hit"
]

class StatsTracker:
    def __init__(self, stats_dir: str = "stats", main_timeframe: str = "240", alert_manager: Optional[Any] = None):
        self.stats_dir = stats_dir
        self.main_timeframe = main_timeframe
        os.makedirs(self.stats_dir, exist_ok=True)
        self.backup_dir = os.path.join(self.stats_dir, "backups")
        os.makedirs(self.backup_dir, exist_ok=True)

        self.signals_file = os.path.join(self.stats_dir, "trade_signals.csv")
        self.results_file = os.path.join(self.stats_dir, "trade_results.csv")
        self.metrics_file = os.path.join(self.stats_dir, "performance_metrics.json")
        self.locks_file = os.path.join(self.stats_dir, "structure_locks.json")
        self.pending_setups_file = os.path.join(self.stats_dir, "pending_setups.json")
        
        self.alert_manager = alert_manager
        
        self._initialize_files()
        self.active_signals: Dict[str, TrackedSignalModel] = {}
        self.last_pivot_points: Dict[str, List[Dict[str, Any]]] = {}
        self.signal_cooldown: Dict[str, datetime] = {}

        self.structure_lockout: Dict[str, List[Dict[str, Any]]] = self._load_structure_locks()
        self.pending_setups = self._load_pending_setups()
        
        self.entry_timeout_hours: int = 20
        self.last_signal_count = 0
        self.last_signal_symbols = set()
        self.notification_service = None

        self._load_active_signals()
        self._update_signal_tracking()
        logger.info(f"StatsTracker başlatıldı: {len(self.active_signals)} aktif sinyal, {len(self.structure_lockout)} yapısal kilit, {len(self.pending_setups)} bekleyen kurulum yüklendi.")

    def set_notification_service(self, notification_service):
        self.notification_service = notification_service

    def _load_structure_locks(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.locks_file):
                with open(self.locks_file, 'r', encoding='utf-8') as f:
                    locks_from_file = json.load(f)
                    for key, pivots in locks_from_file.items():
                        for pivot in pivots:
                            if 'timestamp' in pivot and isinstance(pivot['timestamp'], str):
                                pivot['timestamp'] = datetime.fromisoformat(pivot['timestamp'])
                    logger.info(f"✅ {len(locks_from_file)} yapısal kilit dosyadan yüklendi.")
                    return locks_from_file
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"❌ Yapısal kilitler dosyası okunamadı: {e}. Boş başlatılıyor.")
        return {}

    def _save_structure_locks(self):
        try:
            with open(self.locks_file, 'w', encoding='utf-8') as f:
                json.dump(self._json_ready(self.structure_lockout), f, indent=4)
            logger.debug(f"Yapısal kilitler dosyaya kaydedildi: {self.locks_file}")
        except IOError as e:
            logger.error(f"❌ Yapısal kilitler dosyaya kaydedilemedi: {e}")

    def _json_ready(self, obj):
        if isinstance(obj, dict): return {k: self._json_ready(v) for k, v in obj.items()}
        if isinstance(obj, list): return [self._json_ready(item) for item in obj]
        if isinstance(obj, datetime): return obj.isoformat()
        return obj

    def _initialize_files(self):
        for file_path, fields in [(self.signals_file, SIGNAL_FIELDS), (self.results_file, SIGNAL_FIELDS)]:
            try:
                if not os.path.exists(file_path):
                    with open(file_path, 'w', newline='', encoding='utf-8') as f:
                        csv.DictWriter(f, fieldnames=fields).writeheader()
                    logger.info(f"✅ CSV dosyası oluşturuldu: {file_path}")
                else:
                    self._validate_csv_headers(file_path, fields)
            except IOError as e:
                logger.error(f"❌ {file_path} dosyası oluşturulurken hata: {e}")
        
        json_files = [
            (self.metrics_file, {"total_signals": 0, "completed_trades": 0, "successful_trades": 0, "failed_trades": 0, "success_rate": 0.0, "avg_profit_percentage": 0.0, "total_profit_percentage": 0.0, "tp1_hits": 0, "tp1_5_hits": 0, "tp2_hits": 0, "tp3_hits": 0, "pivot_success_hits": 0, "cancelled_trades": 0, "best_trade": None, "worst_trade": None, "last_updated": datetime.now().isoformat()}),
            (self.locks_file, {}),
            (self.pending_setups_file, {})
        ]
        for file_path, initial_data in json_files:
            try:
                if not os.path.exists(file_path):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(initial_data, f, indent=4)
                    logger.info(f"✅ JSON dosyası oluşturuldu: {file_path}")
            except IOError as e:
                logger.error(f"❌ {file_path} dosyası oluşturulurken hata: {e}")

    def _validate_csv_headers(self, file_path: str, expected_fields: List[str]):
        try:
            if os.path.getsize(file_path) < 10:
                self._recreate_csv_file(file_path, expected_fields)
                return
            with open(file_path, 'r', encoding='utf-8') as f:
                actual_headers = [h.strip() for h in f.readline().strip().split(',')]
            if actual_headers != expected_fields:
                self._recreate_csv_file(file_path, expected_fields)
        except Exception as e:
            logger.error(f"❌ CSV başlık kontrolü hatası: {e}")
            self._recreate_csv_file(file_path, expected_fields)

    def _recreate_csv_file(self, file_path: str, fields: List[str]):
        try:
            if os.path.exists(file_path):
                backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(file_path, backup_path)
                logger.info(f"📁 Dosya yedeklendi: {backup_path}")
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()
            logger.info(f"✅ CSV dosyası yeniden oluşturuldu: {file_path}")
        except Exception as e:
            logger.error(f"❌ CSV dosyası yeniden oluşturulurken hata: {e}")

    def _validate_signal_data(self, signal_dict: Dict[str, Any]) -> bool:
        required = ["symbol", "entry_price", "direction", "timeframe", "sl_price"]
        return all(signal_dict.get(f) is not None for f in required)

    def generate_signal_id(self, symbol: str, direction: str) -> str:
        return f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{direction}"

    def record_signal(self, final_signal: FinalSignalModel) -> Optional[str]:
        """FinalSignalModel nesnesini alır, TrackedSignalModel'e dönüştürür ve kaydeder."""
        try:
            symbol = final_signal.symbol
            direction = final_signal.direction.upper()
            
            if not symbol or self.has_active_signal_for_symbol(symbol):
                logger.warning(f"⚠️ {symbol or 'Bilinmeyen'} için zaten aktif sinyal var veya sembol eksik, kayıt atlanıyor.")
                return None

            signal_model = TrackedSignalModel(
                signal_id=self.generate_signal_id(symbol, direction),
                symbol=symbol,
                timeframe=str(self.main_timeframe), # Ana zaman dilimini string olarak kullan
                entry_time=datetime.now(),
                entry_price=final_signal.primary_entry,
                direction=direction,
                status="PENDING_ENTRY",
                sl_price=final_signal.stop_loss,
                initial_risk=final_signal.initial_risk,
                is_breakeven=False,
                # FinalSignalModel'den gelen zengin verileri aktar
                confirmations=", ".join(final_signal.positive_factors),
                pattern_name=final_signal.signal_type,
                # Referans TP'yi de kaydedelim
                tp1_price=final_signal.tp1
            )

            signal_dict = signal_model.dict(exclude_none=True)
            with open(self.signals_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=SIGNAL_FIELDS, extrasaction='ignore')
                # Dosya boşsa başlıkları yaz
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow(signal_dict)

            self.active_signals[signal_model.signal_id] = signal_model
            logger.info(f"✅ Yeni sinyal kaydedildi: {signal_model.signal_id}")
            
            # Metrikleri güncellemek için sözlük halini kullanmaya devam edebiliriz
            self._update_metrics(new_signal=True, signal_data=signal_dict)
            return signal_model.signal_id
        except Exception as e:
            logger.error(f"❌ Sinyal kaydı sırasında kritik hata: {e}", exc_info=True)
            return None

    def _load_active_signals(self):
        try:
            if not os.path.exists(self.signals_file) or os.path.getsize(self.signals_file) < 50:
                self.active_signals = {}
                return

            df = pd.read_csv(self.signals_file)
            active_statuses = ['PENDING_ENTRY', 'ACTIVE', 'TP1_HIT', 'TP1.5_HIT', 'TP2_HIT', 'TP3_HIT', 'TRAILING_PROFIT']
            active_df = df[df['status'].isin(active_statuses)]

            self.active_signals = {}
            for _, row in active_df.iterrows():
                try:
                    # Veri tiplerini düzelt
                    row_dict = row.replace({np.nan: None}).to_dict()
                    
                    # Timeframe'i string'e çevir
                    if 'timeframe' in row_dict and row_dict['timeframe'] is not None:
                        row_dict['timeframe'] = str(row_dict['timeframe'])
                    
                    # entry_time'ı datetime nesnesine çevir
                    if 'entry_time' in row_dict and row_dict['entry_time'] is not None:
                        if isinstance(row_dict['entry_time'], str):
                            row_dict['entry_time'] = datetime.fromisoformat(row_dict['entry_time'])
                    
                    # result_time'ı da kontrol et
                    if 'result_time' in row_dict and row_dict['result_time'] is not None:
                        if isinstance(row_dict['result_time'], str):
                            row_dict['result_time'] = datetime.fromisoformat(row_dict['result_time'])
                    
                    signal = TrackedSignalModel(**row_dict)
                    self.active_signals[row['signal_id']] = signal
                except Exception as signal_error:
                    logger.warning(f"⚠️ Sinyal yüklenirken hata (atlanıyor): {row.get('signal_id', 'Unknown')} - {signal_error}")
                    continue
                    
            logger.info(f"✅ {len(self.active_signals)} aktif sinyal yüklendi.")
        except Exception as e:
            logger.error(f"❌ Aktif sinyaller yüklenirken hata: {e}", exc_info=True)
            self._repair_csv_file()

    def _repair_csv_file(self):
        """Bozuk CSV dosyasını onarır veya yeniden oluşturur."""
        try:
            logger.warning("🔧 CSV dosyası onarılıyor...")
            
            # Yedek oluştur
            if os.path.exists(self.signals_file):
                backup_path = f"{self.signals_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(self.signals_file, backup_path)
                logger.info(f"📁 CSV yedeklendi: {backup_path}")
            
            # Yeni dosya oluştur
            self._recreate_csv_file(self.signals_file, SIGNAL_FIELDS)
            self.active_signals = {}
            logger.info("✅ CSV dosyası onarıldı ve aktif sinyaller temizlendi.")
            
        except Exception as e:
            logger.error(f"❌ CSV onarımı sırasında hata: {e}")

    def _update_signals_csv(self):
        try:
            with open(self.signals_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=SIGNAL_FIELDS, extrasaction='ignore')
                writer.writeheader()
                writer.writerows([s.dict() for s in self.active_signals.values()])
            logger.debug(f"Aktif sinyaller CSV güncellendi: {len(self.active_signals)} sinyal.")
        except Exception as e:
            logger.error(f"Aktif sinyaller CSV güncellenirken hata: {e}", exc_info=True)

    def check_active_signals(self, current_prices: Dict[str, float], all_timeframe_data: Dict[str, Any]):
        completed_trades = []
        logger.info(f"🔍 Aktif sinyal kontrolü başlıyor: {len(self.active_signals)} sinyal, {len(current_prices)} fiyat")
        
        for signal_id, signal in list(self.active_signals.items()):
            if not signal.symbol or signal.symbol not in current_prices:
                logger.warning(f"⚠️ {signal_id}: Sembol ({signal.symbol}) için güncel fiyat bulunamadı")
                continue

            current_price = current_prices[signal.symbol]
            old_status = signal.status
            logger.debug(f"📊 {signal_id}: Mevcut durum={old_status}, Güncel fiyat={current_price}, Giriş fiyatı={signal.entry_price}")

            new_status = self._evaluate_signal_status(signal, current_price, all_timeframe_data)
            
            if new_status != signal.status:
                logger.info(f"🔄 {signal_id}: Durum değişti {old_status} → {new_status}")
                signal.status = new_status
                
                # Status değişikliğini CSV'ye kaydet
                self._update_signals_csv()
                
                if new_status not in ['PENDING_ENTRY', 'ACTIVE', 'TP1_HIT', 'TP1.5_HIT', 'TP2_HIT', 'TP3_HIT', 'TRAILING_PROFIT']:
                    logger.info(f"✅ {signal_id}: İşlem tamamlandı - {new_status}")
                    self._handle_trade_completion(signal, current_price)
                    completed_trades.append(signal.dict())
                    del self.active_signals[signal_id]
            else:
                logger.debug(f"📊 {signal_id}: Durum değişmedi ({old_status})")

        if completed_trades:
            logger.info(f"📈 {len(completed_trades)} işlem tamamlandı")
            self._update_metrics(completed_trades=completed_trades)
        else:
            logger.debug("📊 Hiçbir işlem tamamlanmadı")

    def _evaluate_signal_status(self, signal: TrackedSignalModel, current_price: float, all_timeframe_data: Dict[str, Any]) -> str:
        """
        Yeni hibrit ticaret yönetimi mantığını uygular.
        1. 0.8R'de BE.
        2. BE sonrası 0.5R'ye düşerse kısmi kâr al.
        3. Kalan pozisyonu Supertrend ile takip et.
        """
        # PENDING_ENTRY durumu
        if signal.status == 'PENDING_ENTRY':
            if self._has_price_reached_entry(signal.direction, current_price, signal.entry_price):
                logger.info(f"✅ GİRİŞ: {signal.signal_id} aktif oldu @ {format_price_standard(signal.entry_price)}")
                if self.alert_manager:
                    self.alert_manager.send_alert(f"✅ {signal.symbol} işlemi aktif oldu! Giriş: {format_price_standard(signal.entry_price)}", "info")
                return 'ACTIVE'
            if (datetime.now() - signal.entry_time).total_seconds() > self.entry_timeout_hours * 3600:
                logger.warning(f"⏰ TIMEOUT: {signal.signal_id} zaman aşımına uğradı.")
                return 'TIMEOUT'
            return 'PENDING_ENTRY'

        # SL kontrolü her zaman en yüksek önceliğe sahiptir
        if (signal.direction.upper() == 'LONG' and current_price <= signal.sl_price) or \
           (signal.direction.upper() == 'SHORT' and current_price >= signal.sl_price):
            logger.info(f"🛑 SL HIT: {signal.signal_id} stop oldu @ {format_price_standard(current_price)}")
            return "SL"

        # Aktif işlem yönetimi
        current_r = abs(current_price - signal.entry_price) / signal.initial_risk if signal.initial_risk > 0 else 0
        logger.debug(f"📊 {signal.signal_id}: R={current_r:.2f}, Status={signal.status}, BE={signal.is_breakeven}")

        # 1. Break-Even Mantığı
        if not signal.is_breakeven and current_r >= 0.8:
            signal.is_breakeven = True
            old_sl = signal.sl_price
            signal.sl_price = signal.entry_price
            logger.info(f"🛡️ BREAKEVEN: {signal.signal_id} stopu girişe çekildi {format_price_standard(old_sl)} → {format_price_standard(signal.sl_price)}")
            if self.alert_manager:
                self.alert_manager.send_alert(f"🛡️ {signal.symbol}: Stop, giriş seviyesine çekildi.", "info")
            return 'BREAKEVEN' # Yeni durum

        # 2. Kısmi Kâr Alma Mantığı (Sadece BE olduktan sonra ve bir kez)
        if signal.is_breakeven and not signal.is_partial_profit_hit and current_r <= 0.5:
            signal.is_partial_profit_hit = True
            signal.partial_profit_price = current_price
            logger.info(f"💰 PARTIAL PROFIT: {signal.signal_id} 0.5R'de kısmi kar aldı ({current_r:.2f}R).")
            if self.alert_manager:
                risk_as_percentage = (signal.initial_risk / signal.entry_price) * 100
                profit_pct = risk_as_percentage * current_r
                self.alert_manager.send_alert(f"💰 {signal.symbol} işlemi 0.5R'de kısmi kâr aldı! Kâr: ~%{profit_pct:.2f} ({current_r:.2f}R)", "info")
            return 'PARTIAL_PROFIT_HIT'

        # 3. Trailing Stop Mantığı (BE veya Kısmi Kâr sonrası)
        if signal.status in ['BREAKEVEN', 'PARTIAL_PROFIT_HIT']:
            supertrend_data = all_timeframe_data.get(signal.symbol, {}).get('supertrend')
            if supertrend_data and hasattr(supertrend_data, 'supertrend_level') and supertrend_data.supertrend_level:
                new_sl = supertrend_data.supertrend_level
                is_sl_improvable = (signal.direction.upper() == 'LONG' and new_sl > signal.sl_price) or \
                                   (signal.direction.upper() == 'SHORT' and new_sl < signal.sl_price)
                if is_sl_improvable:
                    old_sl = signal.sl_price
                    signal.sl_price = new_sl
                    logger.info(f"📈 TRAILING: {signal.signal_id} stop Supertrend'e güncellendi: {format_price_standard(old_sl)} → {format_price_standard(new_sl)}")
                    if self.alert_manager:
                        self.alert_manager.send_alert(f"📈 {signal.symbol} için Stop Loss Supertrend'e güncellendi: {format_price_standard(new_sl)}", "info")
            else:
                logger.debug(f"⚠️ {signal.signal_id}: Trailing için Supertrend verisi mevcut değil.")

        return signal.status # Mevcut durumu koru

    def _handle_trade_completion(self, signal: TrackedSignalModel, result_price: float):
        signal.result_time = datetime.now()
        signal.result_price = result_price
        signal.profit_percentage = self._calculate_profit(signal.direction, signal.entry_price, result_price)
        
        # --- ÇÖZÜM: Nihai R değerini hesapla ve kaydet ---
        if signal.initial_risk > 0:
            if signal.direction.upper() == 'LONG':
                signal.exit_r_value = (result_price - signal.entry_price) / signal.initial_risk
            else: # SHORT
                signal.exit_r_value = (signal.entry_price - result_price) / signal.initial_risk
        else:
            signal.exit_r_value = 0.0
        # --- Bitiş ---
        
        with open(self.results_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=SIGNAL_FIELDS, extrasaction='ignore')
            if f.tell() == 0: writer.writeheader()
            writer.writerow(signal.dict())
        
        logger.info(f"İşlem tamamlandı: {signal.signal_id}, Sonuç: {signal.status}, Kar: {signal.profit_percentage:.2f}%, R Değeri: {signal.exit_r_value:.2f}R")
        if signal.symbol:
            self.start_cooldown(signal.symbol)

    def _update_metrics(self, completed_trades: List = None, new_signal: bool = False, signal_data: Dict[str, Any] = None):
        metrics = self.get_performance_metrics()
        if new_signal and signal_data:
            metrics['total_signals'] = metrics.get('total_signals', 0) + 1
        if completed_trades:
            for trade in completed_trades:
                metrics['completed_trades'] = metrics.get('completed_trades', 0) + 1
                status = trade.get("status")
                profit = trade.get("profit_percentage", 0.0)
                if status in ["TP1_HIT", "TP1.5_HIT", "TP2_HIT", "TP3", "PIVOT_SUCCESS", "TRAILING_STOP_HIT", "TRAILING_PROFIT_HIT"]:
                    metrics['successful_trades'] = metrics.get('successful_trades', 0) + 1
                elif status == "SL":
                    metrics['failed_trades'] = metrics.get('failed_trades', 0) + 1
                metrics['total_profit_percentage'] = metrics.get('total_profit_percentage', 0.0) + profit
        
        completed_count = metrics.get('completed_trades', 0)
        if completed_count > 0:
            metrics['success_rate'] = (metrics.get('successful_trades', 0) / completed_count) * 100
            metrics['avg_profit_percentage'] = metrics.get('total_profit_percentage', 0.0) / completed_count
        
        metrics['last_updated'] = datetime.now().isoformat()
        try:
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=4)
        except Exception as e:
            logger.error(f"Metrikler güncellenirken hata: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except (IOError, json.JSONDecodeError):
            pass
        return self._get_default_metrics()

    def _get_default_metrics(self) -> Dict[str, Any]:
        return {"total_signals": 0, "completed_trades": 0, "successful_trades": 0, "failed_trades": 0, "success_rate": 0.0, "avg_profit_percentage": 0.0, "total_profit_percentage": 0.0, "last_updated": datetime.now().isoformat()}

    def get_active_signals_summary(self, current_prices: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        summary_list = []
        for s in self.active_signals.values():
            pnl_percentage = 0.0
            # --- ÇÖZÜM: Anlık fiyat varsa PNL'yi hesapla ---
            if current_prices and s.symbol in current_prices:
                current_price = current_prices[s.symbol]
                pnl_percentage = self._calculate_profit(s.direction, s.entry_price, current_price)
            # --- Bitiş ---
    
            summary_list.append({
                'signal_id': s.signal_id,
                'symbol': s.symbol,
                'direction': s.direction,
                'status': s.status,
                'pnl_percentage': pnl_percentage # <-- YENİ ALAN
            })
        return summary_list
    
    def has_active_signal_for_symbol(self, symbol: str) -> bool:
        return any(s.symbol == symbol for s in self.active_signals.values())

    def start_cooldown(self, symbol: str, minutes: int = 15):
        self.signal_cooldown[symbol] = datetime.now() + timedelta(minutes=minutes)
        logger.info(f"❄️ {symbol} için {minutes}dk soğuma periyodu başlatıldı.")

    def start_potential_signal_cooldown(self, symbol: str, minutes: int = 10):
        """Potansiyel sinyal tespit edildiğinde kısa süreli cooldown başlatır"""
        self.signal_cooldown[symbol] = datetime.now() + timedelta(minutes=minutes)
        logger.info(f"⏳ {symbol} için potansiyel sinyal cooldown'u başlatıldı ({minutes}dk).")

    def is_on_cooldown(self, symbol: str) -> bool:
        if symbol in self.signal_cooldown:
            if datetime.now() < self.signal_cooldown[symbol]:
                return True
            del self.signal_cooldown[symbol]
        return False
        
    def _has_price_reached_entry(self, direction: Optional[str], current_price: float, entry_price: Optional[float]) -> bool:
        if not entry_price or not direction: 
            logger.warning(f"⚠️ Giriş kontrolü için eksik veri: entry_price={entry_price}, direction={direction}")
            return False
            
        tolerance = 0.002  # %0.2 tolerans (artırıldı)
        direction_lower = direction.lower()
        
        if 'long' in direction_lower or 'bull' in direction_lower:
            threshold = entry_price * (1 - tolerance)
            reached = current_price >= threshold
            logger.debug(f"📈 LONG giriş kontrolü: {current_price} >= {threshold} = {reached}")
            return reached
        elif 'short' in direction_lower or 'bear' in direction_lower:
            threshold = entry_price * (1 + tolerance)
            reached = current_price <= threshold
            logger.debug(f"📉 SHORT giriş kontrolü: {current_price} <= {threshold} = {reached}")
            return reached
        
        logger.warning(f"⚠️ Bilinmeyen yön: {direction}")
        return False

    def _calculate_profit(self, direction: Optional[str], entry_price: Optional[float], exit_price: Optional[float]) -> float:
        if not entry_price or entry_price == 0 or not exit_price or not direction: return 0.0
        direction_lower = direction.lower()
        if 'long' in direction_lower or 'bull' in direction_lower:
            return ((exit_price - entry_price) / entry_price) * 100
        elif 'short' in direction_lower or 'bear' in direction_lower:
            return ((entry_price - exit_price) / entry_price) * 100
        return 0.0

    def _update_signal_tracking(self):
        """Aktif sinyal takibini günceller."""
        current_count = len(self.active_signals)
        current_symbols = set(signal.symbol for signal in self.active_signals.values() if signal.symbol)
        
        self.last_signal_count = current_count
        self.last_signal_symbols = current_symbols

    def _load_pending_setups(self) -> Dict[str, Any]:
        """Beklemedeki kurulumları JSON dosyasından yükler."""
        try:
            if os.path.exists(self.pending_setups_file):
                with open(self.pending_setups_file, 'r', encoding='utf-8') as f:
                    setups = json.load(f)
                    # Timestamp'leri datetime nesnesine çevir
                    for symbol, setup in setups.items():
                        if 'lock_time' in setup and isinstance(setup['lock_time'], str):
                            setup['lock_time'] = datetime.fromisoformat(setup['lock_time'])
                    logger.info(f"✅ {len(setups)} adet beklemedeki kurulum yüklendi.")
                    return setups
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"❌ Beklemedeki kurulumlar dosyası okunamadı: {e}. Boş başlatılıyor.")
        return {}

    def _save_pending_setups(self):
        """Mevcut beklemedeki kurulumları JSON dosyasına kaydeder."""
        try:
            setups_to_save = self._json_ready(self.pending_setups)
            with open(self.pending_setups_file, 'w', encoding='utf-8') as f:
                json.dump(setups_to_save, f, indent=4)
        except IOError as e:
            logger.error(f"❌ Beklemedeki kurulumlar dosyaya kaydedilemedi: {e}")

    def apply_bos_retracement_lock(self, symbol: str, bos_direction: str, target_poi_zone: Dict, bos_timestamp: Optional[Any] = None):
        """Bir BOS sonrası geri çekilme beklentisi için bir kilit uygular."""
        lock_data = {
            'type': 'BOS_RETRACEMENT',
            'expected_direction': bos_direction.lower(),
            'target_poi': target_poi_zone,
            'lock_time': datetime.now(),
            'bos_direction': bos_direction,
            'bos_timestamp': bos_timestamp
        }
        self.pending_setups[symbol] = lock_data
        self._save_pending_setups()
        
        log_message = f"🔒 [BOS Kilidi Aktif] {symbol} için {bos_direction.upper()} yönlü trend devamı bekleniyor."
        logger.warning(log_message)
        
        if self.alert_manager:
            poi_top = target_poi_zone.get('super_poi_top', 0)
            poi_bottom = target_poi_zone.get('super_poi_bottom', 0)
            poi_factors = ", ".join(target_poi_zone.get('confluent_factors', []))
            
            alert_title = f"📈 {symbol} - Yapı Kırılımı (BOS) ve Beklenti"
            alert_message = (
                f"**{symbol}** paritesinde **{bos_direction.upper()}** yönlü bir yapı kırılımı (BOS) tespit edildi.\n\n"
                f"Sistem, fiyatin aşağıdaki POI (Point of Interest) bölgesine geri çekilmesini ve ardından trend yönünde devam etmesini bekliyor:\n\n"
                f"- **Hedef POI:** {format_price_standard(poi_bottom)} - {format_price_standard(poi_top)}\n"
                f"- **İçerik:** {poi_factors}\n\n"
                f"Bu bölgeden **{bos_direction.upper()}** yönlü bir tepki (LTF MSS) aranacak."
            )
            self.alert_manager.send_alert(f"{alert_title}\n\n{alert_message}", "info")

    def get_active_lock(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Belirli bir sembol için aktif bir kilit olup olmadığını kontrol eder ve döndürür."""
        lock = self.pending_setups.get(symbol)
        if lock:
            lock_age_hours = (datetime.now() - lock['lock_time']).total_seconds() / 3600
            if lock_age_hours > 8:
                logger.info(f"⏳ [BOS Kilidi Kaldırıldı] {symbol} için zaman aşımı.")
                self.release_lock(symbol)
                return None
            return lock
        return None

    def release_lock(self, symbol: str):
        """Bir sembol üzerindeki kilidi kaldırır."""
        if symbol in self.pending_setups:
            del self.pending_setups[symbol]
            self._save_pending_setups()
            logger.info(f"🔓 [BOS Kilidi Kaldırıldı] {symbol} için beklenti durumu sona erdi.")