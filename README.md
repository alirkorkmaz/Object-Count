
# 🧠 YOLO Tabanlı Nesne Sayım Sistemi

Gerçek zamanlı video işleme, nesne tespiti, takip ve sayım yapabilen bir sistemdir. Kullanıcı, özel veya hazır YOLO modellerini seçerek video yükleyebilir, nesne sınıflarını filtreleyebilir, takip algoritması belirleyebilir ve video üzerinde canlı sayım gerçekleştirebilir.

![Gif](./assets/video.gif)

---

## 🚀 Özellikler

- ✅ YOLOV5, YOLOv8, YOLOv9, YOLOv10, YOLOv11 gibi modellerle uyumlu
- ✅ Özel `.pt` model yükleme desteği
- ✅ Class filtreleme (tek tek seç, tümünü seç, tümünü kaldır)
- ✅ Takip algoritması seçimi (ByteTrack, BoT-SORT)
- ✅ Çizgi tabanlı geçiş sayımı
- ✅ Video işleme sonrası sonuç videosu izleme
- ✅ Son 10 sayımı ve geçmiş logları görme
- ✅ WebSocket üzerinden canlı veri güncelleme

---

## 🧱 Proje Mimarisi

```
.
├── backend/              # FastAPI + Ultralytics + Tracking
│   ├── api/
│   ├── services/
│   ├── utils/
│   └── main.py
│
├── frontend/             # React arayüz
│   ├── src/
│   ├── public/
│   └── Dockerfile
│
├── docker-compose.yml    # Tüm sistemi ayağa kaldırır
└── README.md             # Bu dosya
```

---

## ⚙️ Kurulum

### 1. Gerekli Araçlar

- Docker
- Docker Compose

### 2. Projeyi Klonla

```bash
git clone https://github.com/kullanici_adi/Object-Count.git
cd Object-Count
```

### 3. Docker ile Servisleri Başlat

```bash
docker compose up --build
```

Frontend → `http://localhost:3000`  
Backend API → `http://localhost:8000/docs`

---

## 🖼️ Kullanım

1. **Video yükle** (.mp4, .avi)
2. **Model seç:** hazır YOLO modeli veya özel `.pt` dosyası
3. **Takip algoritması seç**
4. **Sınıfları filtrele**
5. **Sayımı başlat** → çizim otomatik yapılır, sonuç videosu gösterilir

