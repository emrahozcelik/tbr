from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime

class TBRAnalysisModel(BaseModel):
    """TBR analizi çıktısı için Pydantic modeli. Artık tek bir sinyali temsil eder."""
    symbol: str = Field(..., description="Sinyal sembolü")
    direction: str = Field(..., description="Sinyal yönü (Long/Short)")
    price: float = Field(..., description="Sinyalin oluştuğu anki kapanış fiyatı (3. mum)")
    entry_price: float = Field(..., description="İşleme giriş fiyatı (genellikle 3. mumun kapanışı)")
    pattern_level: float = Field(..., description="Formasyonun referans seviyesi (Long için orta mumun high'ı, Short için low'u)")
    pattern_high: float = Field(..., description="TBR formasyonunun tepe noktası (Stop-Loss için)")
    pattern_low: float = Field(..., description="TBR formasyonunun dip noktası (Stop-Loss için)")
    timestamp: datetime = Field(..., description="Sinyal zamanı")
    timeframe: str = Field(..., description="Sinyal zaman aralığı")

    @validator('direction')
    def validate_direction(cls, v):
        if v.lower() not in ['long', 'short']:
            raise ValueError(f"Geçersiz direction: {v}. 'Long' veya 'Short' olmalı.")
        return v.lower()

class SupertrendAnalysisModel(BaseModel):
    """Supertrend analizi çıktısı için Pydantic modeli."""
    symbol: str
    timeframe: str
    trend: str
    supertrend_level: Optional[float] = Field(None, description="Supertrend indikatörünün sayısal seviyesi")

class FinalSignalModel(BaseModel):
    """
    Nihai, teyit edilmiş ve Telegram'a gönderilmeye hazır sinyal modeli.
    """
    symbol: str = Field(..., description="İşlem sembolü")
    signal_type: str = Field("TBR_Supertrend", description="Sinyal tipi")
    direction: str = Field(..., description="İşlem yönü (Long/Short)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Sinyal oluşturma zamanı")
    
    primary_entry: float = Field(..., description="Ana giriş fiyatı")
    stop_loss: float = Field(..., description="Başlangıç stop loss seviyesi")
    initial_risk: float = Field(..., description="Başlangıçtaki risk miktarı (fiyat farkı)")
    tp1: float = Field(..., description="Take Profit 1 (Artık sadece referans için)")
    
    # Teyit için ek bilgiler
    strategy_used: str = Field("TBR + Supertrend Confirmation", description="Kullanılan strateji")
    htf_trend: Optional[str] = Field(None, description="Teyit için kullanılan Supertrend yönü")
    positive_factors: List[str] = Field(default_factory=lambda: ["TBR Pattern", "Supertrend Confirmation"], description="Pozitif faktörler")

    @validator('direction')
    def validate_direction_final(cls, v):
        if v.lower() not in ['long', 'short']:
            raise ValueError(f"Geçersiz direction: {v}. 'Long' veya 'Short' olmalı.")
        return v.lower()

class TrackedSignalModel(BaseModel):
    """StatsTracker tarafından aktif olarak takip edilen bir sinyalin veri modeli."""
    signal_id: str
    symbol: str
    timeframe: str
    entry_time: datetime
    entry_price: float
    direction: str
    status: str = "PENDING_ENTRY"
    
    # Dinamik Takip için Alanlar
    sl_price: float  # Bu alan artık dinamik olarak güncellenecek
    initial_risk: float # R hesaplamaları için
    is_breakeven: bool = False
    is_partial_profit_hit: bool = False
    partial_profit_price: Optional[float] = None
    
    # Orijinal Sinyal Bilgileri (opsiyonel)
    confirmations: Optional[str] = None
    pattern_name: Optional[str] = None
    initial_pivots: Optional[str] = None
    
    # Sonuç Alanları
    result_time: Optional[datetime] = None
    result_price: Optional[float] = None
    profit_percentage: Optional[float] = None
    exit_r_value: Optional[float] = None # <-- YENİ ALAN
    
    # Eski TP alanları (uyumluluk için, artık kullanılmayacak)
    tp_price: Optional[float] = None
    tp1_price: Optional[float] = None
    tp1_5_price: Optional[float] = None
    tp2_price: Optional[float] = None
    tp3_price: Optional[float] = None
