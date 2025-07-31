import os
from fastapi import HTTPException
from ultralytics import YOLO

# Özel modellerin yükleneceği dizin (Docker için uygun yol)
CUSTOM_MODELS_DIR = "/app/custom_models"

if not os.path.exists(CUSTOM_MODELS_DIR):
    os.makedirs(CUSTOM_MODELS_DIR)

# Hazır model adları ve yolları
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

# Desteklenen takip algoritmaları
SUPPORTED_TRACKERS = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
}

# Bellekte tutulan aktif model
loaded_model = None
current_model_name = None

def load_yolo_model(model_identifier: str):
    global loaded_model, current_model_name

    if model_identifier in SUPPORTED_YOLO_MODELS:
        model_path = SUPPORTED_YOLO_MODELS[model_identifier]
    elif os.path.exists(os.path.join(CUSTOM_MODELS_DIR, model_identifier)):
        model_path = os.path.join(CUSTOM_MODELS_DIR, model_identifier)
    else:
        raise HTTPException(status_code=400, detail=f"Model bulunamadı: {model_identifier}")

    if current_model_name == model_identifier and loaded_model is not None:
        return loaded_model

    try:
        loaded_model = YOLO(model_path)
        current_model_name = model_identifier
        return loaded_model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model yükleme hatası: {e}")

def extract_class_names(model_instance):
    # 1. YOLO tarzı .names varsa
    if hasattr(model_instance, "names") and isinstance(model_instance.names, dict):
        return [{"id": int(k), "name": v} for k, v in model_instance.names.items()]

    # 2. model.data['names'] varsa
    if hasattr(model_instance, "model") and hasattr(model_instance.model, "names"):
        return [{"id": int(k), "name": v} for k, v in model_instance.model.names.items()]

    # 3. class_names gibi alanlar varsa
    for attr in dir(model_instance):
        if "name" in attr.lower() and isinstance(getattr(model_instance, attr, None), (list, dict)):
            raw = getattr(model_instance, attr)
            if isinstance(raw, list):
                return [{"id": i, "name": name} for i, name in enumerate(raw)]
            elif isinstance(raw, dict):
                return [{"id": int(k), "name": v} for k, v in raw.items()]

    return None
