"""
Custom exceptions for the trading automaton.
"""

class AutomatonError(Exception):
    """Base class for exceptions in this application."""
    pass

class InvalidDataError(AutomatonError):
    """Raised when input data is invalid or missing."""
    pass

class APIConnectionError(AutomatonError):
    """Raised for errors related to API connections."""
    pass

class ConfigurationError(AutomatonError):
    """Raised for configuration-related errors."""
    pass

class CalculationError(AutomatonError):
    """Raised during an error in a technical calculation."""
    pass

# Hata Seviyesi Sınıflandırması
class CriticalAnalysisError(AutomatonError):
    """
    CRITICAL: Sistem tamamen durmalı (pipeline'ı durdur)
    - Veri bağlantısı kesilmesi
    - Config hatası
    - Temel sistem bileşeni hatası
    """
    pass

class RecoverableAnalysisError(AutomatonError):
    """
    RECOVERABLE: Default/fallback değer kullan, analizi devam ettir
    - Geçici API hatası
    - Eksik veri (fallback ile çözülebilir)
    - Opsiyonel hesaplama hatası

    Not: keyword argümanları (context) kabul eder ve log/diagnostic için saklar.
    """
    def __init__(self, message: str = "", **context):
        self.message = message
        self.context = context or {}
        super().__init__(message)

    def __str__(self):
        base = super().__str__()
        if self.context:
            try:
                ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            except Exception:
                ctx_str = str(self.context)
            return f"{base} [{ctx_str}]"
        return base

class SilentAnalysisError(AutomatonError):
    """
    SILENT: Sadece log, analizi devam ettir
    - Enhancement hesaplamaları
    - Opsiyonel özellikler
    - Görsel/raporlama hataları
    """
    pass


# Pydantic-Specific Exceptions
class PydanticValidationError(RecoverableAnalysisError):
    """
    Pydantic model validation hatası.
    RECOVERABLE seviyesinde - fallback değerlerle devam edilebilir.
    """
    def __init__(self, model_name: str, field_errors: str, original_data: dict = None):
        self.model_name = model_name
        self.field_errors = field_errors
        self.original_data = original_data or {}
        super().__init__(f"Pydantic validation failed for {model_name}: {field_errors}")


class SignalValidationError(PydanticValidationError):
    """
    Sinyal validation özel hatası.
    Sinyal-specific context bilgileri içerir.
    """
    def __init__(self, signal_type: str, field_errors: str, symbol: str = None, original_data: dict = None):
        self.signal_type = signal_type
        self.symbol = symbol
        context = f"Signal: {signal_type}"
        if symbol:
            context += f", Symbol: {symbol}"
        super().__init__(context, field_errors, original_data)


class AnalyzerOutputValidationError(PydanticValidationError):
    """
    Analizör çıktı validation hatası.
    Analizör-specific context bilgileri içerir.
    """
    def __init__(self, analyzer_name: str, field_errors: str, symbol: str = None, timeframe: str = None, original_data: dict = None):
        self.analyzer_name = analyzer_name
        self.symbol = symbol
        self.timeframe = timeframe
        context = f"Analyzer: {analyzer_name}"
        if symbol:
            context += f", Symbol: {symbol}"
        if timeframe:
            context += f", Timeframe: {timeframe}"
        super().__init__(context, field_errors, original_data)


class SchemaRegistryError(ConfigurationError):
    """
    Schema registry ile ilgili hatalar.
    CRITICAL seviyesinde - sistem konfigürasyon hatası.
    """
    def __init__(self, key: str, message: str):
        self.key = key
        super().__init__(f"Schema registry error for key '{key}': {message}")