FROM python:3.11.9-slim

WORKDIR /app

# Zaman dilimini ayarla
ENV TZ=Europe/Istanbul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Gerekli paketleri yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Uygulama kodunu kopyala
COPY . .

# Uygulama portunu aç
EXPOSE 5041

# Uygulamayı başlat
CMD ["python", "backtest_tbr.py"]
