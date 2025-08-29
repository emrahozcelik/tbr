import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys

# Gerekli modülleri import et
from data_loader import DataLoader
from bybit_client import BybitClient
from schemas import TBRAnalysisModel, SupertrendAnalysisModel, FinalSignalModel
from custom_supertrend import CustomSupertrend

# Loglamayı ayarla
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# --- Backtest Ayarları ---
SYMBOLS_TO_TEST = ["ADAUSDT", "ARBUSDT", "AVAXUSDT", "BNBUSDT", "CRVUSDT", "DOGEUSDT", "DOTUSDT", "ENAUSDT", "HYPEUSDT", "LINKUSDT", "NEARUSDT", "ONDOUSDT", "PEOPLEUSDT", "POPCATUSDT", "RENDERUSDT", "SOLUSDT", "SUIUSDT", "ICPUSDT", "TONUSDT", "TRBUSDT", "TRXUSDT", "VIRTUALUSDT", "WIFUSDT", "XLMUSDT", "XRPUSDT", "NOTUSDT", "TIAUSDT"]
TIMEFRAME_TO_TEST = "15"
BACKTEST_START_DATE = "2025-07-10"
BACKTEST_END_DATE = None
INITIAL_BALANCE_USD = 250
RISK_PERCENTAGE_PER_TRADE = 3.0 # Kasanın %1'i ile işlem yap
# Sabit risk kullanmak için RISK_PERCENTAGE_PER_TRADE = 0 yapın ve aşağıdakini ayarlayın
# RISK_PER_TRADE_USD = 10
# -------------------------

class OptimizedTBRBacktester:
    """
    Optimize edilmiş TBR ve Supertrend stratejisi backtester.
    Bakiye yönetimi ve sembol bazında performans raporlaması içerir.
    """

    def __init__(self, initial_balance: float, risk_percentage: float, risk_per_trade_usd: float = 0):
        self.bybit_client = BybitClient()
        self.data_loader = DataLoader(bybit_client=self.bybit_client)
        self.supertrend_params = {'length': 10, 'multiplier': 3.3}
        
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_percentage = risk_percentage
        self.risk_per_trade_usd = risk_per_trade_usd

        self.custom_supertrend = CustomSupertrend(
            atr_period=self.supertrend_params['length'],
            multiplier=self.supertrend_params['multiplier'],
            change_atr=True
        )
        
        logging.info(f"Optimize edilmiş Backtester başlatıldı. Bakiye: ${initial_balance}, Risk: {risk_percentage}% / işlem")

    def run(self, symbols: list, timeframe: str, start_date: str, end_date: str = None, max_data_iterations: int = 999):
        end_date_str = end_date or datetime.now().strftime('%Y-%m-%d')
        logging.info(f"\n{'='*20} Backtest Başlatılıyor: {', '.join(symbols)} | {timeframe} | {start_date} -> {end_date_str} {'='*20}")

        all_data = self._fetch_data(symbols, timeframe, start_date, end_date_str, max_data_iterations)
        
        if not all_data:
            logging.error("Hiçbir sembol için veri çekilemedi. Backtest durduruluyor.")
            return

        all_signals = []
        all_data_with_indicators = {} # İndikatörlü verileri saklamak için sözlük

        for symbol, df in all_data.items():
            if df.empty:
                logging.warning(f"[{symbol}] için veri boş. Atlanıyor.")
                continue

            logging.info(f"[{symbol}] {len(df)} mum verisiyle analiz ediliyor...")
            df_with_indicators = self._calculate_all_indicators(df)
            all_data_with_indicators[symbol] = df_with_indicators # İndikatörlü DF'i sakla

            tbr_signals = self._detect_tbr_patterns_vectorized(df_with_indicators)
            filtered_signals = self._apply_supertrend_filter(tbr_signals, df_with_indicators)
            
            for _, signal in filtered_signals.iterrows():
                signal_dict = signal.to_dict()
                signal_dict['symbol'] = symbol
                all_signals.append(signal_dict)

        # Tüm sinyalleri zamana göre sırala
        all_signals.sort(key=lambda x: x['timestamp'])
        logging.info(f"Tüm sembollerden toplam {len(all_signals)} adet sinyal bulundu ve sıralandı.")

        # Trade simülasyonu
        trade_results = []
        # active_trades sözlüğü artık sembolün meşgul olduğu son zamanı tutacak
        active_trades_until = {}

        for signal in all_signals:
            symbol = signal['symbol']
            signal_time = signal['timestamp']

            # Eğer sembol hala meşgulse (önceki işlem bitmediyse), bu sinyali atla
            if symbol in active_trades_until and signal_time < active_trades_until[symbol]:
                continue

            df_with_indicators = all_data_with_indicators[symbol]
            try:
                # Sinyalin oluştuğu barın bir sonrası ile simülasyonu başlat
                start_loc = df_with_indicators.index.get_loc(signal_time) + 1
            except KeyError:
                continue
            
            future_df = df_with_indicators.iloc[start_loc:]
            if len(future_df) < 10:
                continue

            trade_result = self._simulate_trade_dynamic(future_df, signal)

            if trade_result:
                # Sembolün meşgul olacağı son zamanı güncelle
                exit_timestamp = trade_result['exit_timestamp']
                active_trades_until[symbol] = exit_timestamp
                
                trade_result['symbol'] = symbol
                trade_results.append(trade_result)
                
                logging.info(
                    f"İŞLEM KAYDI: Sembol={symbol:<12} | "
                    f"Sonuç={trade_result['result']:<8} | "
                    f"R={trade_result['r_value']:.2f} | "
                    f"PnL=${trade_result['profit_usd']:<12,.2f} | "
                    f"Yeni Bakiye=${trade_result['balance']:<15,.2f}"
                )
        
        self._report(trade_results)

    def _fetch_data(self, symbols: list[str], timeframe: str, start_date_str: str, end_date_str: str, max_iterations: int) -> dict[str, pd.DataFrame]:
        """Belirtilen tarih aralığı için birden fazla sembolün verisini çeker."""
        try:
            start_time = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_time = datetime.strptime(end_date_str, '%Y-%m-%d')
        except ValueError:
            logging.error("Tarih formatı hatalı. Lütfen 'YYYY-MM-DD' formatını kullanın.")
            return {}

        logging.info(f"Veri çekiliyor... Başlangıç: {start_time.strftime('%Y-%m-%d')}, Bitiş: {end_time.strftime('%Y-%m-%d')}")
        
        start_timestamp_ms = int(start_time.timestamp() * 1000)
        end_timestamp_ms = int(end_time.timestamp() * 1000)

        # Sembol listesi string ise, virgülle ayırarak listeye çevir
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(',')]

        return self.data_loader.get_historical_data_for_backtest(
            symbols=symbols,
            timeframe=timeframe,
            start_timestamp=start_timestamp_ms,
            end_timestamp=end_timestamp_ms,
            max_iterations=max_iterations
        )

    def _calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tüm gerekli indikatörleri tek seferde hesaplar. Sadece CustomSupertrend kullanılır."""
        df_copy = df.copy()
        
        # Canlı sistemle %100 uyumlu Custom Supertrend'i hesapla
        up, dn, trend, supertrend = self.custom_supertrend.calculate_supertrend(
            df_copy['high'], df_copy['low'], df_copy['close']
        )
        
        # Sonuçları standart kolon adlarıyla DataFrame'e ekle
        length = self.supertrend_params['length']
        multiplier = self.supertrend_params['multiplier']
        
        df_copy[f'SUPERTd_{length}_{multiplier}'] = trend
        df_copy[f'SUPERTl_{length}_{multiplier}'] = up
        df_copy[f'SUPERTs_{length}_{multiplier}'] = dn
        
        return df_copy

    def _detect_tbr_patterns_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized TBR pattern detection - çok daha hızlı!
        """
        # Numpy arrays kullanarak hızlı hesaplama
        o = df['open'].values
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        
        # Bullish TBR koşulları (vectorized)
        bullish_cond1 = c[:-2] < o[:-2]  # close[2] < open[2]
        bullish_cond2 = l[1:-1] < l[:-2]  # low[1] < low[2]
        bullish_cond3 = h[1:-1] < h[:-2]  # high[1] < high[2]
        bullish_cond4 = c[1:-1] < o[1:-1]  # close[1] < open[1]
        bullish_cond5 = c[2:] > o[2:]  # close > open
        bullish_cond6 = h[2:] > h[:-2]  # high > high[2]
        
        bullish_pattern = (bullish_cond1 & bullish_cond2 & bullish_cond3 & 
                          bullish_cond4 & bullish_cond5 & bullish_cond6)
        
        # Bearish TBR koşulları (vectorized)
        bearish_cond1 = c[:-2] > o[:-2]  # close[2] > open[2]
        bearish_cond2 = h[1:-1] > h[:-2]  # high[1] > high[2]
        bearish_cond3 = l[1:-1] > l[:-2]  # low[1] > low[2]
        bearish_cond4 = c[1:-1] > o[1:-1]  # close[1] > open[1]
        bearish_cond5 = c[2:] < o[2:]  # close < open
        bearish_cond6 = l[2:] < l[:-2]  # low < low[2]
        
        bearish_pattern = (bearish_cond1 & bearish_cond2 & bearish_cond3 & 
                          bearish_cond4 & bearish_cond5 & bearish_cond6)
        
        # Enhanced variant kontrolü (vectorized)
        bullish_enhanced = bullish_pattern & (c[2:] > h[:-2])  # close > high[i-2]
        bearish_enhanced = bearish_pattern & (c[2:] < l[:-2])  # close < low[i-2]
        
        # Sonuçları DataFrame olarak hazırla
        signals_data = []
        
        # Bullish sinyaller
        for idx in np.where(bullish_enhanced)[0]:
            actual_idx = idx + 2  # Offset düzeltmesi
            signals_data.append({
                'index': actual_idx,
                'direction': 'long',
                'entry_price': h[actual_idx - 2], # GİRİŞ: 1. mumun high seviyesi
                'pattern_high': h[actual_idx - 2],
                'pattern_low': l[actual_idx - 2],
                'pattern_level': h[actual_idx - 2],
                'timestamp': df.index[actual_idx]
            })
        
        # Bearish sinyaller
        for idx in np.where(bearish_enhanced)[0]:
            actual_idx = idx + 2  # Offset düzeltmesi
            signals_data.append({
                'index': actual_idx,
                'direction': 'short',
                'entry_price': l[actual_idx - 2], # GİRİŞ: 1. mumun low seviyesi
                'pattern_high': h[actual_idx - 2],
                'pattern_low': l[actual_idx - 2],
                'pattern_level': l[actual_idx - 2],
                'timestamp': df.index[actual_idx]
            })
        
        if not signals_data:
            return pd.DataFrame()
            
        return pd.DataFrame(signals_data).sort_values('index')

    def _apply_supertrend_filter(self, signals_df: pd.DataFrame, main_df: pd.DataFrame) -> pd.DataFrame:
        """Supertrend filtresi uygular."""
        if signals_df.empty:
            return signals_df
        
        st_direction_col = f'SUPERTd_{self.supertrend_params["length"]}_{self.supertrend_params["multiplier"]}'
        st_long_col = f'SUPERTl_{self.supertrend_params["length"]}_{self.supertrend_params["multiplier"]}'
        st_short_col = f'SUPERTs_{self.supertrend_params["length"]}_{self.supertrend_params["multiplier"]}'
        
        filtered_signals = []
        
        for _, signal in signals_df.iterrows():
            idx = signal['index']
            direction = signal['direction']
            
            # İlgili bardaki Supertrend yönünü al
            st_direction = main_df.iloc[idx][st_direction_col]
            supertrend_trend = 'long' if st_direction == 1 else 'short'
            
            # TBR ile Supertrend uyumunu kontrol et
            if direction == supertrend_trend:
                # Sinyal verisine o anki ST seviyelerini de ekleyelim
                signal_with_st = signal.copy()
                signal_with_st[st_long_col] = main_df.iloc[idx][st_long_col]
                signal_with_st[st_short_col] = main_df.iloc[idx][st_short_col]
                filtered_signals.append(signal_with_st)
        
        return pd.DataFrame(filtered_signals) if filtered_signals else pd.DataFrame()

    def _simulate_trade_dynamic(self, future_df: pd.DataFrame, signal: dict) -> dict:
        """
        Bakiye yönetimi entegre edilmiş dinamik ticaret simülasyonu.
        """
        entry_price = signal['entry_price']
        direction = signal['direction']

        # Dinamik veya sabit riski hesapla
        if self.risk_percentage > 0:
            risk_in_usd = self.balance * (self.risk_percentage / 100)
        else:
            risk_in_usd = self.risk_per_trade_usd
        
        if risk_in_usd <= 0: return None

        stop_loss = self.calculate_stop_loss(signal, entry_price, direction)
        if stop_loss is None: return None
            
        initial_risk_price = abs(entry_price - stop_loss)
        if initial_risk_price == 0: return None

        # Simülasyon Değişkenleri
        current_sl = stop_loss
        max_favorable_r = 0.0
        is_breakeven = False
        is_partial_profit_hit = False # Sadece loglama ve raporlama için

        st_long_col = f'SUPERTl_{self.supertrend_params["length"]}_{self.supertrend_params["multiplier"]}'
        st_short_col = f'SUPERTs_{self.supertrend_params["length"]}_{self.supertrend_params["multiplier"]}'

        # Ana Simülasyon Döngüsü
        for i in range(len(future_df)):
            bar = future_df.iloc[i]
            
            # Stop loss kontrolü (öncelikli)
            if (direction == 'long' and bar['low'] <= current_sl) or \
               (direction == 'short' and bar['high'] >= current_sl):
                exit_price = current_sl
                final_r = (exit_price - entry_price) / initial_risk_price if direction == 'long' else (entry_price - exit_price) / initial_risk_price
                
                exit_reason = "initial_stop"
                if is_breakeven:
                    exit_reason = "breakeven_stop" if current_sl == entry_price else "trailing_profit_stop"

                profit_usd = final_r * risk_in_usd
                self.balance += profit_usd

                return {
                    "result": "loss" if final_r <= 0 else "win", "exit": exit_price,
                    "r_value": final_r, "exit_reason": exit_reason, "max_favorable_r": max_favorable_r,
                    "profit_usd": profit_usd, "balance": self.balance, "exit_timestamp": bar.name
                }

            # Maksimum R takibi
            current_high_r = (bar['high'] - entry_price) / initial_risk_price if direction == 'long' else (entry_price - bar['low']) / initial_risk_price
            max_favorable_r = max(max_favorable_r, current_high_r)

            # 1. Break-Even Mantığı
            if not is_breakeven and max_favorable_r >= 0.8:
                is_breakeven = True
                new_sl = entry_price
                if (direction == 'long' and new_sl > current_sl) or \
                   (direction == 'short' and new_sl < current_sl):
                    current_sl = new_sl
                    logging.debug(f"BE: {bar.name} -> Stop girişe çekildi: {current_sl:.5f}")

            # 2. Kısmi Kâr Alma Anını Tespit Etme (Sadece Raporlama için)
            current_close_r = (bar['close'] - entry_price) / initial_risk_price if direction == 'long' else (entry_price - bar['close']) / initial_risk_price
            if is_breakeven and not is_partial_profit_hit and current_close_r <= 0.5:
                is_partial_profit_hit = True
                logging.debug(f"PARTIAL PROFIT HIT (LOG): {bar.name} -> Fiyat BE sonrası 0.5R altına düştü.")
            
            # 3. Trailing Stop Mantığı (Sadece BE sonrası aktif)
            if is_breakeven:
                supertrend_level = bar[st_long_col] if direction == 'long' else bar[st_short_col]
                if pd.notna(supertrend_level):
                    # SL sadece yukarı/aşağı hareket edebilir, asla kötüleşemez
                    if (direction == 'long' and supertrend_level > current_sl) or \
                       (direction == 'short' and supertrend_level < current_sl):
                        current_sl = supertrend_level
                        logging.debug(f"Trailing: {bar.name} -> ST takip ediliyor: {current_sl:.5f}")

        # Veri biterse, son barın kapanış fiyatıyla pozisyonu kapat
        final_bar = future_df.iloc[-1]
        exit_price = final_bar['close']
        final_r = (exit_price - entry_price) / initial_risk_price if direction == 'long' else (entry_price - exit_price) / initial_risk_price
        
        profit_usd = final_r * risk_in_usd
        self.balance += profit_usd
            
        return {
            "result": "closed_at_end", "exit": exit_price, "r_value": final_r,
            "exit_reason": "end_of_data", "max_favorable_r": max_favorable_r,
            "profit_usd": profit_usd, "balance": self.balance, "exit_timestamp": final_bar.name
        }

    def calculate_stop_loss(self, signal, entry_price, direction):
        """
        Sadece Supertrend tabanlı, min/max risk kuralları içeren stop loss hesaplama.
        - Sadece geçerli bir Supertrend seviyesi varsa işlem yapılır.
        - Supertrend yoksa veya geçersizse, işlem atlanır (None döner).
        - Supertrend risk seviyesi %1'den az ise, SL %1 riske ayarlanır (Minimum Risk).
        - Supertrend risk seviyesi %2.2'den fazla ise, SL %2.2 riske ayarlanır (Maksimum Risk).
        """
        min_risk_pct = 0.01  # Minimum %1 risk
        max_risk_pct = 0.022 # Maksimum %2.2 risk
        buffer_pct = 0.0007  # Fiyatın SL'e dokunmasını önlemek için küçük bir tampon

        st_long_col = f'SUPERTl_{self.supertrend_params["length"]}_{self.supertrend_params["multiplier"]}'
        st_short_col = f'SUPERTs_{self.supertrend_params["length"]}_{self.supertrend_params["multiplier"]}'

        timestamp_val = signal['timestamp']
        if isinstance(timestamp_val, (int, np.int64)):
            # Tamsayı ise, pandas Timestamp nesnesine geri dönüştür
            timestamp_obj = pd.to_datetime(timestamp_val)
        else:
            timestamp_obj = timestamp_val
        
        timestamp_str = timestamp_obj.strftime('%Y-%m-%d %H:%M')

        logging.debug(f"\n--- SL HESAPLAMA BAŞLADI: {timestamp_str} ({direction}) ---")
        logging.debug(f"Giriş Fiyatı: {entry_price:.5f}")

        # 1. Supertrend Varlığını ve Geçerliliğini Kontrol Et
        supertrend_level = signal[st_long_col] if direction == 'long' else signal[st_short_col]

        if pd.isna(supertrend_level):
            logging.warning(f"[{timestamp_str}] Supertrend verisi mevcut değil (NaN). İşlem için SL belirlenemedi.")
            return None

        candidate_sl = (supertrend_level - (entry_price * buffer_pct)) if direction == 'long' else (supertrend_level + (entry_price * buffer_pct))
        
        is_st_valid = (direction == 'long' and candidate_sl < entry_price) or \
                      (direction == 'short' and candidate_sl > entry_price)

        if not is_st_valid:
            logging.warning(f"[{timestamp_str}] Supertrend seviyesi ({candidate_sl:.5f}) giriş fiyatına ({entry_price:.5f}) göre geçersiz. İşlem için SL belirlenemedi.")
            return None

        # 2. Geçerli Supertrend Seviyesini Min/Max Risk Kurallarına Göre Ayarla
        logging.debug(f"Geçerli Aday SL (Supertrend): {candidate_sl:.5f}")
        candidate_risk_pct = abs(entry_price - candidate_sl) / entry_price
        logging.debug(f"Aday SL Riski (Supertrend): {candidate_risk_pct:.4%}")

        final_sl = None
        
        if candidate_risk_pct < min_risk_pct:
            final_sl = entry_price * (1 - min_risk_pct) if direction == 'long' else entry_price * (1 + min_risk_pct)
            logging.debug(f"KARAR: Supertrend riski ({candidate_risk_pct:.4%}) minimumdan ({min_risk_pct:.2%}) düşük. SL, {min_risk_pct:.2%}'ye sabitlendi.")
        
        elif candidate_risk_pct > max_risk_pct:
            final_sl = entry_price * (1 - max_risk_pct) if direction == 'long' else entry_price * (1 + max_risk_pct)
            logging.debug(f"KARAR: Supertrend riski ({candidate_risk_pct:.4%}) maksimumdan ({max_risk_pct:.2%}) yüksek. SL, {max_risk_pct:.2%}'ye sabitlendi.")
            
        else:
            final_sl = candidate_sl
            logging.debug(f"KARAR: Supertrend riski ({candidate_risk_pct:.4%}) aralık içinde. Supertrend SL kullanıldı.")

        logging.debug(f"NİHAİ STOP LOSS: {final_sl:.5f}")
        return final_sl

    def _report(self, results: list):
        """Gelişmiş, bakiye ve sembol bazlı performans raporu oluşturur."""
        if not results:
            logging.warning("Raporlanacak sonuç bulunamadı.")
            return

        df = pd.DataFrame(results)
        df['max_favorable_r'] = df['max_favorable_r'].fillna(0)
        total_trades = len(df)

        # 1. Bakiye Büyüme Raporu
        logging.info(f"\n--- 📈 Bakiye Büyüme Raporu 📈 ---")
        total_profit_usd = df['profit_usd'].sum()
        balance_growth_pct = (total_profit_usd / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        logging.info(f"Başlangıç Bakiyesi: ${self.initial_balance:,.2f}")
        logging.info(f"Bitiş Bakiyesi:     ${self.balance:,.2f}")
        logging.info(f"Toplam Net Kâr/Zarar: ${total_profit_usd:,.2f}")
        logging.info(f"Bakiye Büyümesi:      {balance_growth_pct:.2f}%")
        logging.info("-" * 40)

        # 2. Genel Performans İstatistikleri
        win_rate = (df['r_value'] > 0).sum() / total_trades * 100 if total_trades > 0 else 0
        total_r = df['r_value'].sum()
        avg_r = df['r_value'].mean() if total_trades > 0 else 0
        
        winning_trades = df[df['r_value'] > 0]
        losing_trades = df[df['r_value'] <= 0]
        avg_win_r = winning_trades['r_value'].mean() if not winning_trades.empty else 0
        avg_loss_r = losing_trades['r_value'].mean() if not losing_trades.empty else 0
        profit_factor = winning_trades['profit_usd'].sum() / abs(losing_trades['profit_usd'].sum()) if abs(losing_trades['profit_usd'].sum()) > 0 else float('inf')

        logging.info("--- 📊 Genel Performans 📊 ---")
        logging.info(f"Toplam İşlem Sayısı: {total_trades}")
        logging.info(f"Kazanma Oranı: {win_rate:.2f}%")
        logging.info(f"Toplam Net R Değeri: {total_r:.2f}R")
        logging.info(f"Ortalama R/İşlem: {avg_r:.3f}R")
        logging.info(f"Ortalama Kazanan R: {avg_win_r:.3f}R")
        logging.info(f"Ortalama Kaybeden R: {avg_loss_r:.3f}R")
        logging.info(f"Profit Factor: {profit_factor:.2f}")
        logging.info("-" * 40)

        # 3. Sembol Bazında Performans
        symbol_stats = df.groupby('symbol').agg(
            total_profit_usd=('profit_usd', 'sum'),
            trade_count=('symbol', 'size'),
            win_rate=('r_value', lambda x: (x > 0).sum() / len(x) * 100)
        ).sort_values('total_profit_usd', ascending=False)

        logging.info("--- 🏆 Sembol Bazında Performans 🏆 ---")
        logging.info(f"{'Sembol':<15} | {'Toplam Kâr':>15} | {'İşlem Sayısı':>15} | {'Kazanma Oranı':>15}")
        logging.info(f"{'-'*15} | {'-'*15} | {'-'*15} | {'-'*15}")
        for symbol, stats in symbol_stats.iterrows():
            logging.info(f"{symbol:<15} | ${stats['total_profit_usd']:>14,.2f} | {stats['trade_count']:>15} | {stats['win_rate']:>14.2f}%")
        logging.info("-" * 40)

        # 4. Diğer İstatistikler (Seriler, Çıkış Nedenleri vb.)
        win_streak, loss_streak, max_win_streak, max_loss_streak = 0, 0, 0, 0
        for r in df['r_value']:
            if r > 0:
                win_streak += 1; loss_streak = 0
            else:
                loss_streak += 1; win_streak = 0
            max_win_streak = max(max_win_streak, win_streak)
            max_loss_streak = max(max_loss_streak, loss_streak)

        logging.info("--- ⚙️ Ek İstatistikler ⚙️ ---")
        logging.info(f"En Uzun Kazanma Serisi: {max_win_streak}")
        logging.info(f"En Uzun Kaybetme Serisi: {max_loss_streak}")
        
        exit_reasons = df['exit_reason'].value_counts(normalize=True) * 100
        logging.info("Çıkış Nedenleri Dağılımı:")
        for reason, percentage in exit_reasons.items():
            logging.info(f"  - {reason.replace('_', ' ').title()}: {percentage:.2f}%")

        trailing_profit_trades = df[df['exit_reason'] == 'trailing_profit_stop']
        if not trailing_profit_trades.empty:
            logging.info("Supertrend Takip Kârı İstatistikleri:")
            logging.info(f"  - Kapanan İşlem Sayısı: {len(trailing_profit_trades)}")
            logging.info(f"  - Ortalama Kâr (R): {trailing_profit_trades['r_value'].mean():.2f}R")
        
        logging.info("=" * 42)

if __name__ == "__main__":
    bt = OptimizedTBRBacktester(
        initial_balance=INITIAL_BALANCE_USD,
        risk_percentage=RISK_PERCENTAGE_PER_TRADE,
        # risk_per_trade_usd=RISK_PER_TRADE_USD # Sabit risk için bu satırı açın
    )
    
    bt.run(
        symbols=SYMBOLS_TO_TEST,
        timeframe=TIMEFRAME_TO_TEST,
        start_date=BACKTEST_START_DATE,
        end_date=BACKTEST_END_DATE,
        max_data_iterations=999 # Tüm veriyi kullan
    )