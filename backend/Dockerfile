# 1. Temel imaj (hafif ve güncel Python)
FROM python:3.12-slim-bullseye

# 2. Çalışma dizini oluştur
WORKDIR /app

# 3. Sistem bağımlılıkları (OpenCV ve ffmpeg için)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 4. requirements.txt dosyasını kopyala ve pip ile yükle
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Tüm uygulama kodunu kopyala
COPY . .

# 6. Video kayıt klasörünü oluştur ve yazılabilir yap
RUN mkdir -p processed_videos && chmod -R 777 processed_videos

# (Opsiyonel) Ultralytics model cache klasörü (otomatik oluşur genelde)
# RUN mkdir -p /root/.cache/ultralytics/weights

# 7. Uvicorn ile uygulamayı başlat (modüler yapıda artık main.py kullanılıyor)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
