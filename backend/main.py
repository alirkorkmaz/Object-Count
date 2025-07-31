from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocket, WebSocketDisconnect
from database.config import connect_db, disconnect_db, create_db_tables
from detection_routes import router as detection_router
from websocket_manager import manager


# uygulama başlatırken bilgi eklendi
# Swagger UI altında dökümantasyon sağlayacağı için önemlidir
app = FastAPI(
    title="YOLO Video Counter API",
    description="Video üzerinde nesne sayımı ve video işleme servisi",
    version="0.1.0"
)

# CORS ayarları
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

# Veritabanı bağlantıları
@app.on_event("startup")
async def startup_event():
    await connect_db()
    await create_db_tables()

@app.on_event("shutdown")
async def shutdown_event():
    await disconnect_db()

# Route'ları tanıt
app.include_router(detection_router)

# Ana endpoint
@app.get("/")
async def root():
    return {
        "message": "YOLO Video Counter API'ye Hoş Geldiniz!",
        "docs": "/docs",
        "live_ws": "/ws/video-count"
    }



@app.websocket("/ws/video-count")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Gelen mesaj: {data}")
            await manager.broadcast(f"Sunucudan mesaj: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("WebSocket bağlantısı kesildi.")
