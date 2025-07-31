import os
import cv2
import json
import uuid
import time
import shutil
import asyncio
from datetime import datetime
from fastapi import UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.sql import insert, update
from database.config import database
from database.models import DetectionRecord, OverallCount
from model_manager import load_yolo_model, SUPPORTED_TRACKERS
from websocket_manager import manager
from utils import check_line_crossing

async def process_video_stream(
    video_file: UploadFile,
    model_name: str,
    tracker_name: str,
    selected_class_ids: list,
    line_coordinates: list,
    conf_threshold: float,
    iou_threshold: float
):
    
    try:
        model = load_yolo_model(model_name)
    except HTTPException as e:
        raise e

    if tracker_name not in SUPPORTED_TRACKERS:
        raise HTTPException(status_code=400, detail="Geçersiz tracker adı")
    tracker_config = SUPPORTED_TRACKERS[tracker_name]

    try:
        line_p1 = tuple(map(int, line_coordinates[0]))
        line_p2 = tuple(map(int, line_coordinates[1]))
        is_line_defined = (line_p1 != (0, 0) or line_p2 != (0, 0))
    except:
        raise HTTPException(status_code=400, detail="Çizgi koordinatları geçersiz")

   
    temp_path = f"temp_{uuid.uuid4().hex}_{video_file.filename}"
    output_filename = f"processed_{uuid.uuid4().hex}_{video_file.filename}"
    output_path = os.path.join("processed_videos", output_filename)
    os.makedirs("processed_videos", exist_ok=True)

    initial_record_id = None

    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(video_file.file, f)

        # Veritabanına ilk kayıt
        start_time = datetime.now()
        query = insert(OverallCount).values({
            "video_name": video_file.filename,
            "model_used": model_name,
            "tracker_used": tracker_name,
            "final_count": 0,
            "start_time": start_time,
            "line_coordinates": json.dumps(line_coordinates)
        })
        initial_record_id = await database.execute(query)

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Video açılamadı")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Codec deneme
        CODECS = ["mp4v", "XVID", "MJPG"]
        out = None
        for codec in CODECS:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out.isOpened():
                break

        if not out:
            raise HTTPException(status_code=500, detail="VideoWriter başlatılamadı")

        # Sayım Değişkenleri
        total_count = 0
        last_positions = {}
        object_line_states = {}

        async def generate():
            nonlocal total_count

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.track(
                    frame,
                    persist=True,
                    tracker=tracker_config,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False
                )

                annotated = results[0].plot() if results else frame.copy()
                tracked_ids = set()

                for r in results:
                    if not r.boxes or not r.boxes.id is not None:
                        continue
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        if selected_class_ids and cls not in selected_class_ids:
                            continue

                        track_id = int(box.id.item())
                        label = model.names[cls]
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        current_pos = (cx, cy)
                        tracked_ids.add(track_id)

                        if is_line_defined:
                            key = (track_id, 0)
                            if key not in object_line_states:
                                object_line_states[key] = {"crossed": False}

                            if track_id in last_positions:
                                prev = last_positions[track_id]
                                if check_line_crossing(prev, current_pos, line_p1, line_p2):
                                    if not object_line_states[key]["crossed"]:
                                        total_count += 1
                                        object_line_states[key]["crossed"] = True
                                        await manager.broadcast(json.dumps({
                                            "event": "object_counted",
                                            "object_id": track_id,
                                            "object_label": label,
                                            "total_count": total_count
                                        }))

                                        det = {
                                            "video_name": video_file.filename,
                                            "model_used": model_name,
                                            "tracker_used": tracker_name,
                                            "object_id": track_id,
                                            "object_label": label,
                                            "timestamp": datetime.now(),
                                            "current_total_count": total_count,
                                        }
                                        await database.execute(insert(DetectionRecord).values(det))

                                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                        cv2.putText(annotated, f"{label} COUNTED", (x1, y1 - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        last_positions[track_id] = current_pos

                # Çizgi çiz
                if is_line_defined:
                    cv2.line(annotated, line_p1, line_p2, (0, 255, 255), 2)
                    cv2.putText(annotated, f"Count: {total_count}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                out.write(annotated)

                ret, buffer = cv2.imencode(".jpg", annotated)
                if not ret:
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

            cap.release()
            out.release()

            await database.execute(update(OverallCount).where(OverallCount.id == initial_record_id).values(
                final_count=total_count,
                end_time=datetime.now(),
                processed_video_path=output_path
            ))

            await manager.broadcast(json.dumps({
                "event": "video_ended",
                "total_count": total_count,
                "processed_video_url": f"http://127.0.0.1:8000/processed-videos/{output_filename}"
            }))

            await asyncio.sleep(1)

        return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
