# Dockerfile for FastAPI Backend

# Python'ın resmi hafif sürümünü temel imaj olarak kullan
# Python 3.12-slim-bullseye, daha küçük bir imaj boyutu için idealdir.
FROM python:3.12-slim-bullseye

# Çalışma dizinini ayarla
WORKDIR /app

# Bağımlılıkları kopyala ve yükle
# requirements.txt dosyasını kopyala
COPY requirements.txt .

# Gerekli sistem bağımlılıklarını yükle (OpenCV için)
# Bu adım, OpenCV'nin düzgün çalışması için gerekli olan kütüphaneleri sağlar.
# build-essential: Derleme araçları
# libgl1-mesa-glx, libglib2.0-0: OpenCV'nin GUI dışı işlevleri için gerekli GL ve Glib kütüphaneleri
# ffmpeg: Video işleme için (OpenCV'nin video okuma/yazma yetenekleri için)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Python bağımlılıklarını yükle
# --no-cache-dir: pip'in önbellek dizini oluşturmasını engeller, imaj boyutunu küçültür.
# --upgrade pip: pip'i en son sürüme günceller.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Uygulama kodunu kopyala
# main.py ve diğer Python dosyalarını /app dizinine kopyala
COPY . .

# PROCESSED_VIDEOS DİZİNİNİ OLUŞTUR VE YAZMA İZİNLERİNİ AYARLA
# Bu adım, uygulamanın video dosyalarını kaydedebilmesi için gereklidir.
# mkdir -p: Dizin yoksa oluşturur.
# chmod -R 777: Dizine ve altındaki tüm dosyalara tam okuma/yazma/çalıştırma izni verir.
# Bu, geliştirme ortamı için kolay bir çözümdür. Üretim ortamında daha kısıtlı izinler (örn. 755) ve
# uygulamanın çalıştığı kullanıcıya sahiplik vermek (chown) daha güvenlidir.

# YOLO modellerinin indirileceği dizini oluştur (isteğe bağlı, Ultralytics otomatik indirir)
# RUN mkdir -p /root/.cache/ultralytics/weights

# Uygulama çalıştığında Uvicorn'u başlat
# 0.0.0.0 adresinde dinlemesi, Docker konteynerinin dışarıdan erişilebilir olmasını sağlar.
# --host 0.0.0.0: Tüm ağ arayüzlerinde dinle.
# --port 8000: 8000 numaralı portta dinle.
CMD ["uvicorn", "islem_motoru:app", "--host", "0.0.0.0", "--port", "8000"]
