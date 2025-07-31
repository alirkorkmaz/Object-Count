from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from database.config import database
from database.models import DetectionRecord, OverallCount
from sqlalchemy.sql import select, insert
from model_manager import load_yolo_model, extract_class_names, SUPPORTED_YOLO_MODELS, SUPPORTED_TRACKERS, CUSTOM_MODELS_DIR
from video_service import process_video_stream
import os
import shutil
import json

router = APIRouter()

@router.get("/models")
async def get_supported_models():
    return {"supported_models": list(SUPPORTED_YOLO_MODELS.keys())}

@router.get("/trackers")
async def get_supported_trackers():
    return {"supported_trackers": list(SUPPORTED_TRACKERS.keys())}

@router.get("/custom-models")
async def get_custom_models():
    models_list = []
    if os.path.exists(CUSTOM_MODELS_DIR):
        for filename in os.listdir(CUSTOM_MODELS_DIR):
            if filename.endswith(".pt"):
                models_list.append(filename)
    return {"custom_models": models_list}

@router.post("/upload-model/")
async def upload_model(model_file: UploadFile = File(...)):
    if not model_file.filename.endswith(".pt"):
        raise HTTPException(status_code=400, detail="Sadece .pt uzantılı model dosyaları yüklenebilir.")
    
    file_location = os.path.join(CUSTOM_MODELS_DIR, model_file.filename)
    try:
        with open(file_location, "wb") as f:
            shutil.copyfileobj(model_file.file, f)
        return JSONResponse(status_code=200, content={"message": f"Model '{model_file.filename}' yüklendi."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model yüklenirken hata: {e}")

@router.get("/model-classes/{model_name}")
async def get_model_classes(model_name: str):
    model_instance = load_yolo_model(model_name)
    class_list = extract_class_names(model_instance)
    if class_list:
        return {"classes": class_list}
    raise HTTPException(status_code=400, detail="Modelde sınıf isimleri bulunamadı.")

@router.get("/processed-videos/{filename}")
async def get_processed_video(filename: str):
    video_path = os.path.join("processed_videos", filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video bulunamadı.")
    return FileResponse(path=video_path, media_type="video/mp4", filename=filename)

@router.get("/last-10-counts")
async def get_last_10_overall_counts():
    try:
        query = select(OverallCount).order_by(OverallCount.id.desc()).limit(10)
        records = await database.fetch_all(query)
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
        raise HTTPException(status_code=500, detail=f"Hata oluştu: {e}")

@router.post("/process-video/")
async def process_video_endpoint(
    video_file: UploadFile = File(...),
    model_name: str = Form("yolov8n"),
    tracker_name: str = Form("bytetrack"),
    line_coordinates: str = Form("[[0,0],[0,0]]"),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.7),
    selected_class_ids: str = Form("[]")
):
    try:
        selected_class_ids_list = json.loads(selected_class_ids)
        parsed_line = json.loads(line_coordinates)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Girdi hatası: {e}")

    return await process_video_stream(
        video_file, model_name, tracker_name,
        selected_class_ids_list, parsed_line,
        conf_threshold, iou_threshold
    )
