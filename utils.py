from typing import Optional
from loguru import logger

def format_price_standard(price: Optional[float]) -> str:
    """
    Farklı büyüklükteki kripto para fiyatlarını standart formatta biçimlendirir.
    
    Args:
        price: Biçimlendirilecek fiyat değeri
        
    Returns:
        str: Biçimlendirilmiş fiyat değeri
    """
    if price is None:
        return "N/A"
    
    try:
        # Fiyatın büyüklüğüne göre hassasiyeti belirle
        if price >= 1000:
            return f"{price:.2f}"
        elif price >= 1:
            return f"{price:.4f}"
        else:
            return f"{price:.6f}"
            
    except (TypeError, ValueError) as e:
        logger.warning(f"Fiyat formatlanırken hata oluştu: {price} - Hata: {e}")
        return str(price)