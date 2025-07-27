# islem_motoru.py veya main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response # Response import edildi
import cv2
from ultralytics import YOLO
import os
import shutil
import json
import asyncio
import time
import uuid
from datetime import datetime

# Veritabanı import'ları (database klasörünüzün backend klasörü içinde olduğundan emin olun!)
from database.config import database, create_db_tables, connect_db, disconnect_db
from database.models import DetectionRecord, OverallCount
from sqlalchemy.sql import select, insert, update

app = FastAPI(
    title="YOLOv8 Video Counter API",
    description="Video üzerinde dinamik nesne tespiti, takibi, canlı sayım ve işlenmiş video kaydı ile veritabanı entegrasyonu yapan FastAPI uygulaması",
    version="0.1.0",
)

# --- CORS Middleware Ekle ---
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Uygulama Başlangıcı ve Kapanışı İçin Olay Dinleyicileri ---
@app.on_event("startup")
async def startup_event():
    await connect_db()
    await create_db_tables()

@app.on_event("shutdown")
async def shutdown_event():
    await disconnect_db()

# --- Global değişkenler ve model/tracker yükleme ---
CUSTOM_MODELS_DIR = "/app/custom_models" # Docker içinde /app/custom_models

if not os.path.exists(CUSTOM_MODELS_DIR):
    os.makedirs(CUSTOM_MODELS_DIR)
    print(f"Özel modeller dizini oluşturuldu: {CUSTOM_MODELS_DIR}")


SUPPORTED_YOLO_MODELS = {
    "yolov8n": "yolov8n.pt", "yolov8s": "yolov8s.pt", "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt", "yolov8x": "yolov8x.pt",
    "yolov5n": "yolov5n.pt", "yolov5s": "yolov5s.pt", "yolov5m": "yolov5m.pt",
    "yolov5l": "yolov5l.pt", "yolov5x": "yolov5x.pt",
    "yolov9c": "yolov9c.pt", "yolov9e": "yolov9e.pt", "yolov9t": "yolov9t.pt",
    "yolov10n": "yolov10n.pt", "yolov10s": "yolov10s.pt", "yolov10m": "yolov10m.pt",
    "yolov10l": "yolov10l.pt", "yolov10x": "yolov10x.pt",
    "yolo11n": "yolo11n.pt", "yolo11s": "yolo11s.pt", "yolo11m": "yolo11m.pt",
    "yolo11l": "yolo11l.pt", "yolo11x": "yolo11x.pt",
    "yolo12n": "yolo12n.pt", "yolo12s": "yolo12s.pt", "yolo12m": "yolo12m.pt",
    "yolo12l": "yolo12l.pt", "yolo12x": "yolo12x.pt",
}

SUPPORTED_TRACKERS = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
}

loaded_model = None
current_model_name = None

def load_yolo_model(model_identifier: str):
    global loaded_model, current_model_name

    if model_identifier in SUPPORTED_YOLO_MODELS:
        model_path = SUPPORTED_YOLO_MODELS[model_identifier]
        model_type = "Standard"
    elif os.path.exists(os.path.join(CUSTOM_MODELS_DIR, model_identifier)):
        model_path = os.path.join(CUSTOM_MODELS_DIR, model_identifier)
        model_type = "Custom"
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen veya bulunamayan YOLO modeli: {model_identifier}. "
                   f"Desteklenenler: {list(SUPPORTED_YOLO_MODELS.keys())} veya '{CUSTOM_MODELS_DIR}' dizinindeki .pt dosyaları."
        )

    if current_model_name == model_identifier and loaded_model is not None:
        print(f"Model '{model_identifier}' (tip: {model_type}) zaten yüklü.")
        return loaded_model

    print(f"Model '{model_identifier}' (tip: {model_type}) yükleniyor...")
    try:
        loaded_model = YOLO(model_path)
        current_model_name = model_identifier
        print(f"YOLO modeli '{model_identifier}' başarıyla yüklendi.")
        return loaded_model
    except Exception as e:
        print(f"YOLO modeli '{model_identifier}' yüklenirken bir hata oluştu: {e}")
        loaded_model = None
        current_model_name = None
        raise HTTPException(status_code=500, detail=f"Model '{model_identifier}' yüklenemedi: {e}")

# --- Sayım Mantığı Fonksiyonları ---
def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0: return 0
    return 1 if val > 0 else 2

def on_segment(p, q, r):
    if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        return True
    return False

def check_line_crossing(point1, point2, line_p1, line_p2):
    o1 = orientation(line_p1, line_p2, point1)
    o2 = orientation(line_p1, line_p2, point2)
    o3 = orientation(point1, point2, line_p1)
    o4 = orientation(point1, point2, line_p2)
    if o1 != 0 and o2 != 0 and o3 != 0 and o4 != 0 and o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(line_p1, point1, line_p2): return True
    if o2 == 0 and on_segment(line_p1, point2, line_p2): return True
    if o3 == 0 and on_segment(point1, line_p1, point2): return True
    if o4 == 0 and on_segment(point1, line_p2, point2): return True
    return False

# --- WebSocket Bağlantı Yöneticisi ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Yeni WebSocket bağlantısı: {websocket.client.host}:{websocket.client.port}")
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"WebSocket bağlantısı kesildi: {websocket.client.host}:{websocket.client.port}")
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                self.active_connections.remove(connection)
                print(f"Kopuk WebSocket bağlantısı temizlendi.")
            except Exception as e:
                print(f"Mesaj gönderirken hata: {e}")
                self.active_connections.remove(connection)

manager = ConnectionManager()

@app.get("/")
async def read_root():
    return {"message": "YOLOv8 Video Counter API'ye Hoş Geldiniz! API dokümantasyonu için /docs adresini ziyaret edin, canlı sayım için /ws/video-count WebSocket bağlantısını kullanın."}

@app.get("/models", summary="Desteklenen standart YOLO modellerini listele")
async def get_supported_models():
    return {"supported_models": list(SUPPORTED_YOLO_MODELS.keys())}

@app.get("/trackers", summary="Desteklenen takip algoritmalarını listele")
async def get_supported_trackers():
    return {"supported_trackers": list(SUPPORTED_TRACKERS.keys())}

@app.get("/custom-models", summary="Kendi eğitilmiş YOLO modellerini listele")
async def get_custom_models():
    models_list = []
    if os.path.exists(CUSTOM_MODELS_DIR) and os.path.isdir(CUSTOM_MODELS_DIR):
        for filename in os.listdir(CUSTOM_MODELS_DIR):
            if filename.endswith(".pt"):
                models_list.append(filename)
    else:
        print(f"Uyarı: {CUSTOM_MODELS_DIR} dizini bulunamadı veya bir dizin değil.")
    return {"custom_models": models_list}

@app.post("/upload-model/", summary="Kendi eğitilmiş YOLO modelini yükle")
async def upload_model(model_file: UploadFile = File(...)):
    if not model_file.filename.endswith(".pt"):
        raise HTTPException(status_code=400, detail="Sadece .pt uzantılı model dosyaları yüklenebilir.")

    file_location = os.path.join(CUSTOM_MODELS_DIR, model_file.filename)

    try:
        with open(file_location, "wb") as f:
            shutil.copyfileobj(model_file.file, f)
        print(f"Model başarıyla yüklendi: {file_location}")
        return JSONResponse(status_code=200, content={"message": f"Model '{model_file.filename}' başarıyla yüklendi."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model yüklenirken bir hata oluştu: {e}")

@app.get("/model-classes/{model_name}", summary="Belirtilen modelin sınıf listesini döndürür")
async def get_model_classes(model_name: str):
    try:
        model_instance = load_yolo_model(model_name)
        if model_instance and hasattr(model_instance, 'names'):
            class_list = [{"id": int(k), "name": v} for k, v in model_instance.names.items()]
            return JSONResponse(status_code=200, content={"classes": class_list})
        else:
            raise HTTPException(status_code=404, detail="Model bulunamadı veya sınıf bilgisi yok.")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model sınıfları çekilirken hata: {e}")

@app.websocket("/ws/video-count")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            print(f"WebSocket'ten gelen mesaj: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket hatası: {e}")
        manager.disconnect(websocket)

@app.get("/counts", summary="Tüm genel sayım kayıtlarını listele")
async def get_all_overall_counts():
    query = select(OverallCount)
    records = await database.fetch_all(query)
    return [dict(r) for r in records]

@app.get("/detections", summary="Tüm bireysel algılama kayıtlarını listele")
async def get_all_detection_records():
    query = select(DetectionRecord)
    records = await database.fetch_all(query)
    return [dict(r) for r in records]

# --- YENİ EKLENECEK ENDPOINT: Son 10 Sayım Videosu Verisi ---
@app.get("/last-10-counts", summary="Veritabanından son 10 video sayım kaydını çeker")
async def get_last_10_overall_counts():
    try:
        # 'id' sütununa göre azalan sırada sırala ve ilk 10 kaydı al
        # Alternatif olarak 'start_time' veya 'end_time' da kullanılabilir.
        # Bu örnekte en son eklenen kayıtları almak için 'id' yeterlidir.
        query = select(OverallCount).order_by(OverallCount.id.desc()).limit(10)
        records = await database.fetch_all(query)

        # Sorgu sonuçlarını dictionary listesine dönüştür
        # Eğer datetime nesneleri JSON serileştirme hatası verirse, onları string'e çevirmek gerekebilir.
        # Örneğin: record["start_time"].isoformat() if record["start_time"] else None
        return [
            {
                "id": r.id,
                "video_name": r.video_name,
                "model_used": r.model_used,
                "tracker_used": r.tracker_used,
                "final_count": r.final_count,
                "start_time": r.start_time.isoformat() if r.start_time else None,
                "end_time": r.end_time.isoformat() if r.end_time else None,
                "line_coordinates": r.line_coordinates,
                "processed_video_path": r.processed_video_path
            } for r in records
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Son 10 sayım verisi çekilirken hata oluştu: {e}")


# islem_motoru.py
#from fastapi.responses import Response # Response import edildi

@app.get("/processed-videos/{filename}", summary="Kaydedilen işlenmiş videoyu sunar")
async def get_processed_video(filename: str):
    video_path = os.path.join("processed_videos", filename)

    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="İşlenmiş video bulunamadı.")

    try:
        # Dosyayı ikili (binary) modda okuyun
        with open(video_path, "rb") as video_file:
            video_content = video_file.read()
        
        # İçeriği 'video/mp4' medya tipiyle döndürün. Bu başlık, tarayıcıya oynatmasını söyler.
        return Response(content=video_content, media_type="video/mp4") 
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video okunurken hata oluştu: {e}")


@app.post("/process-video/")
async def process_video(
    video_file: UploadFile = File(...),
    model_name: str = Form("yolov8n"),
    tracker_name: str = Form("bytetrack"),
    line_coordinates: str = Form("[[0,0],[0,0]]"),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.7),
    selected_class_ids: str = Form("[]")
):
    try:
        current_processing_model = load_yolo_model(model_name)
    except HTTPException as e:
        raise e

    if tracker_name not in SUPPORTED_TRACKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen takip algoritması: {tracker_name}. Desteklenenler: {list(SUPPORTED_TRACKERS.keys())}"
        )
    selected_tracker_config = SUPPORTED_TRACKERS[tracker_name]

    try:
        selected_class_ids_list = json.loads(selected_class_ids)
        selected_class_ids_list = [int(cls_id) for cls_id in selected_class_ids_list]
        print(f"Seçilen Sınıf ID'leri: {selected_class_ids_list}")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Seçilen sınıf ID'leri geçersiz formatta: {e}. Beklenen format: JSON array of integers.")

    try:
        parsed_line = json.loads(line_coordinates)
        if not (isinstance(parsed_line, list) and len(parsed_line) == 2 and
                all(isinstance(p, list) and len(p) == 2 and all(isinstance(coord, (int, float)) for coord in p) for p in parsed_line)):
            raise ValueError("Geçersiz çizgi koordinat formatı.")

        line_p1 = (int(parsed_line[0][0]), int(parsed_line[0][1]))
        line_p2 = (int(parsed_line[1][0]), int(parsed_line[1][1]))

        is_line_defined = not (line_p1 == (0,0) and line_p2 == (0,0))

    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Çizgi koordinatları geçersiz formatta: {e}. Beklenen format: [[x1,y1],[x2,y2]]")

    temp_video_path = f"temp_{uuid.uuid4().hex}_{video_file.filename}"
    output_video_filename = f"processed_{uuid.uuid4().hex}_{video_file.filename}"
    output_video_path = os.path.join("processed_videos", output_video_filename)

    os.makedirs("processed_videos", exist_ok=True)

    initial_overall_count_id = None

    try:
        with open(temp_video_path, "wb") as f:
            shutil.copyfileobj(video_file.file, f)

        video_processing_start_time = time.time()

        overall_count_data = {
            "video_name": video_file.filename,
            "model_used": model_name,
            "tracker_used": tracker_name,
            "final_count": 0,
            "start_time": datetime.fromtimestamp(video_processing_start_time),
            "line_coordinates": line_coordinates
        }
        query = insert(OverallCount).values(overall_count_data)
        initial_overall_count_id = await database.execute(query)

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            print(f"HATA: Video dosyası açılamadı: {temp_video_path}. Desteklenmeyen format veya bozuk dosya.")
            raise HTTPException(status_code=400, detail="Video dosyası açılamadı.")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video bilgisi: Genişlik={frame_width}, Yükseklik={frame_height}, FPS={fps}, Toplam Kare={total_frames}")

        # Codec'i değiştirme stratejisi: mp4v, XVID, MJPG sırasıyla dene
        CODECS_TO_TRY = ['mp4v', 'XVID', 'MJPG']
        out = None
        used_codec = None

        for codec_str in CODECS_TO_TRY:
            fourcc = cv2.VideoWriter_fourcc(*codec_str)
            try:
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
                if out.isOpened():
                    used_codec = codec_str
                    print(f"VideoWriter '{used_codec}' codec'i ile başarıyla açıldı: {output_video_path}")
                    break # Başarılı olursa döngüden çık
                else:
                    print(f"UYARI: VideoWriter '{codec_str}' codec'i ile açılamadı. Deneniyor...")
                    if out: out.release() # Açılmayan out nesnesini serbest bırak
            except Exception as e:
                print(f"UYARI: VideoWriter '{codec_str}' codec'i ile başlatılırken hata: {e}. Deneniyor...")
                if out: out.release() # Hata veren out nesnesini serbest bırak
            out = None # Bir sonraki deneme için out'u sıfırla

        if not out:
            raise HTTPException(status_code=500, detail="VideoWriter hiçbir desteklenen codec ile başlatılamadı. Lütfen sunucu günlüklerini kontrol edin.")

        total_count = 0
        object_line_states = {}
        crossing_lines = [ (line_p1, line_p2) ] if is_line_defined else []
        last_positions = {}

        frame_counter = 0
        UPDATE_FREQUENCY = 5

        async def generate_frames():
            nonlocal total_count, frame_counter

            while True:
                ret, frame = cap.read()

                if not ret:
                    print(f"Video sonuna ulaşıldı (ret=False). Toplam işlenen kare: {frame_counter}")
                    break

                results = current_processing_model.track(
                    frame,
                    persist=True,
                    tracker=selected_tracker_config,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose = True,
                )
                
                # Başlangıçta annotated_frame orijinal karenin bir kopyası olsun
                annotated_frame = frame.copy()

                # YOLOv8'in plot metodunu kullanarak otomatik çizim yap
                # Sadece sonuçlar varsa plot et
                if results and len(results) > 0 and results[0].boxes and results[0].boxes.data.shape[0] > 0:
                    try:
                        temp_annotated_frame = results[0].plot(
                            boxes=True,
                            conf=True,
                            labels=True,
                        )
                        # Plot metodu farklı boyut veya tip döndürebileceği için kontrol edelim
                        if temp_annotated_frame.shape[1] == frame_width and \
                           temp_annotated_frame.shape[0] == frame_height and \
                           temp_annotated_frame.dtype == 'uint8' and \
                           temp_annotated_frame.shape[2] == 3: # RGB/BGR 3 kanal
                            annotated_frame = temp_annotated_frame
                        else:
                            print(f"UYARI: YOLO plot metodu beklenmeyen boyutta/tipte kare döndürdü. "
                                  f"Beklenen: ({frame_width}, {frame_height}, uint8, 3), Alınan: {temp_annotated_frame.shape}, {temp_annotated_frame.dtype}")
                            # Boyut uyumsuzluğu durumunda yeniden boyutlandırma
                            annotated_frame = cv2.resize(temp_annotated_frame, (frame_width, frame_height))
                            # Tip uyumsuzluğu durumunda dönüştürme (genellikle 0-255 aralığında kalır)
                            if annotated_frame.dtype != 'uint8':
                                annotated_frame = annotated_frame.astype('uint8')

                    except Exception as e:
                        print(f"UYARI: YOLO plot metodunda hata oluştu: {e}. Orijinal kare kullanılacak.")
                        # Hata durumunda annotated_frame zaten frame.copy() olduğu için bir şey yapmaya gerek yok.
                
                # Sayım mantığı ve ek manuel çizimler
                current_frame_tracked_ids = set()
                for r in results:
                    if r.boxes and r.boxes.id is not None:
                        for i, box in enumerate(r.boxes):
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls = int(box.cls[0])

                            if selected_class_ids_list and cls not in selected_class_ids_list:
                                continue

                            track_id = int(box.id.item())
                            label = current_processing_model.names[cls]
                            current_frame_tracked_ids.add(track_id)

                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2
                            current_position = (cx, cy)

                            if is_line_defined:
                                for j, (lp1, lp2) in enumerate(crossing_lines):
                                    line_key = (track_id, j)

                                    if line_key not in object_line_states:
                                        object_line_states[line_key] = {'crossed': False}

                                    if track_id in last_positions:
                                        prev_pos = last_positions[track_id]

                                        if check_line_crossing(prev_pos, current_position, lp1, lp2):
                                            if not object_line_states[line_key]['crossed']:
                                                total_count += 1
                                                object_line_states[line_key]['crossed'] = True
                                                print(f"Nesne ID {track_id} çizgiyi geçti. Toplam Sayım: {total_count}")

                                                count_data = {
                                                    "total_count": total_count,
                                                    "event": "object_counted",
                                                    "object_id": track_id,
                                                    "object_label": label,
                                                    "timestamp": time.time()
                                                }
                                                await manager.broadcast(json.dumps(count_data))

                                                detection_record_data = {
                                                    "video_name": video_file.filename,
                                                    "model_used": model_name,
                                                    "tracker_used": tracker_name,
                                                    "object_id": track_id,
                                                    "object_label": label,
                                                    "timestamp": datetime.fromtimestamp(time.time()),
                                                    "current_total_count": total_count,
                                                }
                                                query = insert(DetectionRecord).values(detection_record_data)
                                                await database.execute(query)

                                                # Sayılan nesneyi vurgula
                                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3) # Kırmızı ve daha kalın
                                                cv2.putText(annotated_frame, f"ID:{track_id} {label} COUNTED!", (x1, y1 - 20),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    last_positions[track_id] = current_position

                # --- Kaybolan nesnelerin durumunu sıfırla ---
                keys_to_reset_crossed_state = []
                for (t_id, l_idx), state in object_line_states.items():
                    if state['crossed'] and t_id not in current_frame_tracked_ids:
                        keys_to_reset_crossed_state.append((t_id, l_idx))

                for k_key in keys_to_reset_crossed_state:
                    object_line_states[k_key]['crossed'] = False
                    print(f"Nesne ID {k_key[0]} görünmüyor, çizgiden geçiş durumu sıfırlandı.")

                keys_to_remove_from_last_pos = [t_id for t_id in last_positions if t_id not in current_frame_tracked_ids]
                for k_id in keys_to_remove_from_last_pos:
                    del last_positions[k_id]

                # Çizgiyi ve sayım bilgisini annotated_frame üzerine çizdirin
                if is_line_defined:
                    cv2.line(annotated_frame, line_p1, line_p2, (0, 255, 255), 2) # Sarı
                    cv2.putText(annotated_frame, f"Total Count: {total_count}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Debugging: annotated_frame'in boyutlarını ve tipini kontrol edelim
                # Bu çıktılar, kayıt sorununun kökenini anlamamızda kritik.
                if frame_counter % (fps // 2 + 1) == 0: # Saniyede yaklaşık 2 kez kontrol et
                    print(f"Debug [Kare {frame_counter}]: annotated_frame boyutu: {annotated_frame.shape}, tipi: {annotated_frame.dtype}")
                    if annotated_frame.shape[1] != frame_width or annotated_frame.shape[0] != frame_height:
                        print(f"UYARI: annotated_frame boyutu VideoWriter başlangıç boyutlarından farklı! "
                              f"Beklenen: ({frame_width}, {frame_height}), Alınan: ({annotated_frame.shape[1]}, {annotated_frame.shape[0]})")
                    if annotated_frame.dtype != 'uint8':
                        print(f"UYARI: annotated_frame veri tipi 'uint8' değil! Alınan: {annotated_frame.dtype}")
                    if len(annotated_frame.shape) < 3 or annotated_frame.shape[2] != 3: # Renkli olması bekleniyorsa
                         print(f"UYARI: annotated_frame kanal sayısı 3 değil! Alınan: {annotated_frame.shape[2] if len(annotated_frame.shape) == 3 else 'N/A (tek kanal)'}")
                
                # `out` nesnesi tanımlanmış ve açılmışsa kareyi yaz
                if out and out.isOpened():
                    try:
                        print(f"DEBUG: writing frame. shape: {annotated_frame.shape}, dtype: {annotated_frame.dtype}, channels: {annotated_frame.shape[2] if len(annotated_frame.shape) > 2 else 'N/A'}")
                        write_success = out.write(annotated_frame)
                        if not write_success:
                            print(f"HATA: Kare {frame_counter} VideoWriter'a yazılırken başarısız oldu! (Yazma hatası)")
                    except Exception as e:
                        print(f"HATA: Kare {frame_counter} VideoWriter'a yazılırken istisna oluştu: {e}")
                else:
                    if frame_counter % (fps // 2 + 1) == 0:
                        print(f"UYARI [Kare {frame_counter}]: VideoWriter nesnesi kapalı veya başlatılmamış. Kare kaydedilmiyor.")

                frame_counter += 1
                if frame_counter % UPDATE_FREQUENCY == 0:
                    general_count_data = {
                        "total_count": total_count,
                        "event": "general_update",
                        "timestamp": time.time(),
                        "processed_frames": frame_counter,
                        "total_frames": total_frames
                    }
                    await manager.broadcast(json.dumps(general_count_data))

                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                if not ret:
                    continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            # Döngü bittiğinde kaynakları serbest bırak
            cap.release()

            if out:
                out.release()
                print(f"İşlenmiş video kaydedildi: {output_video_path} (Codec: {used_codec})")
                await manager.broadcast(json.dumps({
                    "event": "processed_video_saved",
                    "path": output_video_path, # Backend dosya yolu
                    "total_count": total_count,
                    "video_name": video_file.filename,
                    "processed_video_url": f"http://127.0.0.1:8000/processed-videos/{output_video_filename}" # Frontend'in kullanacağı URL
                }))

            final_count_data = {
                "total_count": total_count,
                "event": "video_ended",
                "timestamp": time.time(),
                "video_name": video_file.filename,
                "processed_video_url": f"http://127.0.0.1:8000/processed-videos/{output_video_filename}" # Frontend'in kullanacağı URL
            }
            await manager.broadcast(json.dumps(final_count_data))

            if initial_overall_count_id is not None:
                update_query = (
                    update(OverallCount)
                    .where(OverallCount.id == initial_overall_count_id) 
                    .values(
                        final_count=total_count,
                        end_time=datetime.fromtimestamp(time.time()),
                        processed_video_path=output_video_path # Veritabanına backend dosya yolu kaydediliyor
                    )
                )
                await database.execute(update_query)
                print(f"Genel sayım kaydı (ID: {initial_overall_count_id}) güncellendi. Son Sayım: {total_count}")

            await asyncio.sleep(1)

    finally:
        # Geçici video dosyası silme satırı 
        
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    # `StreamingResponse` video işleme sırasında canlı akışı sağlar.
    # İşleme bittiğinde, bu akış doğal olarak kapanır.
    # Frontend'in bu kapanışı algılayıp yeni videoyu (WebSocket ile gelen URL'den) oynatması gerekir.
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")