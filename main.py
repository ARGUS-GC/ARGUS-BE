import os
import time
import json
import base64
import uuid
import threading
import psycopg2
import paho.mqtt.client as mqtt
import cv2
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from ultralytics import YOLO
import supervision as sv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from psycopg2.extras import RealDictCursor
from typing import Optional, List   

# ================= 설정 (Configuration) =================
IMAGE_DIR = "saved_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# DB 설정 (Docker Compose service name: db)
DB_CONFIG = {
    "host": "localhost",  # Docker 내부라면 "db", 로컬 테스트면 "localhost"
    "database": "argus_db",
    "user": "argus_user",
    "password": "argus_password",
    "port": "5432"
}

# AI 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FIRE_MODEL_PATH = 'models/fire_classifier_resnet.pth'
YOLO_MODEL_PATH = 'best.pt'  

# 클래스 ID 설정 (best.pt 기준)
CLASS_ID_HELMET = 0
CLASS_ID_NO_HELMET = 1

# [Task 4] 나홀로 작업 경고 설정
LONE_WORKER_LIMIT = 5  # 테스트를 위해 5초로 설정 (기존 180초)

app = FastAPI(title="Argus Server")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수 
latest_frame_bytes = None  # 대시보드 송출용 (JPEG Bytes)
latest_frame_cv = None     # AI 분석용 (OpenCV Image)
mqtt_client = None

# ================= AI 모델 로드 =================

# (1) 화재 모델 (ResNet)
def load_fire_model():
    print("🔥 화재 감지 모델 로딩 중...")
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    if os.path.exists(FIRE_MODEL_PATH):
        try:
            ckpt = torch.load(FIRE_MODEL_PATH, map_location=DEVICE)
            state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
            new_state = {k.replace('module.', ''): v for k, v in state.items()}
            model.load_state_dict(new_state, strict=False)
            model.to(DEVICE)
            model.eval()
            print("✅ 화재 모델 로드 완료")
            return model
        except Exception as e:
            print(f"❌ 화재 모델 로드 실패: {e}")
            return None
    return None

fire_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 모델 초기화
print("⏳ AI 모델 로딩 시작...")

# YOLO 모델을 best.pt로 로드
if os.path.exists(YOLO_MODEL_PATH):
    yolo_model = YOLO(YOLO_MODEL_PATH) 
    print(f"✅ PPE 모델({YOLO_MODEL_PATH}) 로드 완료")
else:
    print(f"⚠️ {YOLO_MODEL_PATH} 없음. 기본 yolov8n.pt 로드 (PPE 감지 불가)")
    yolo_model = YOLO('yolov8n.pt') 

fire_model = load_fire_model()    # 화재 감지용
print("✅ AI 시스템 준비 완료")

# DB 연결 함수
def get_db_connection():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"❌ DB 접속 실패: {e}")
        return None

# ================= [Thread 1] MQTT 수신 (영상 받기) =================
def on_message(client, userdata, msg):
    global latest_frame_bytes, latest_frame_cv
    
    try:
        topic = msg.topic
        if topic.endswith("/stream"):
            payload = json.loads(msg.payload.decode('utf-8'))
            img_data = base64.b64decode(payload['img_base64'])
            
            # 1) 대시보드 송출용
            latest_frame_bytes = img_data
            
            # 2) AI 분석용 (OpenCV 변환)
            nparr = np.frombuffer(img_data, np.uint8)
            latest_frame_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    except Exception as e:
        print(f"⚠️ MQTT 수신 에러: {e}")

mqtt_client = mqtt.Client()
mqtt_client.on_message = on_message
mqtt_client.connect("localhost", 1883, 60)
mqtt_client.subscribe("argus/#")
mqtt_client.loop_start()

# ================= [Thread 2] AI 분석 루프 (백그라운드) =================
def ai_processing_loop():
    global latest_frame_cv
    
    # ----------------------------------------------------
    # [설정] Task 4: 나홀로 작업 구역 (PolygonZone) - 좌표 업데이트
    # ----------------------------------------------------
    polygons = [
        np.array([[48, 222], [328, 220], [306, 612], [48, 652]]),      # Zone 1
        np.array([[600, 100], [1000, 100], [1000, 500], [600, 500]]),  # Zone 2
        np.array([[1100, 100], [1500, 100], [1500, 500], [1100, 500]]) # Zone 3
    ]
    zones = []
    zone_annotators = []
    zone_timers = [None] * len(polygons)

    # Supervision 설정
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    for polygon in polygons:
        zone = sv.PolygonZone(polygon=polygon, triggering_anchors=[sv.Position.CENTER])
        zones.append(zone)
        zone_annotators.append(
            sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED, thickness=2, text_scale=0.8)
        )

    last_alert_time = 0
    
    print("🚀 AI 분석 엔진 가동 (PPE + Lone Worker + Fire)")
    
    while True:
        if latest_frame_cv is None:
            time.sleep(0.1)
            continue
            
        try:
            frame = latest_frame_cv.copy()
            height, width, _ = frame.shape
            detected_events = [] 
            
            # ==========================================================
            # [Step 1] YOLO 추론 & 트래킹 (best.pt)
            # ==========================================================
            # imgsz=640 등 옵션 추가 가능
            results = yolo_model.track(frame, persist=True, verbose=False, conf=0.5)
            
            # Supervision Detections 변환
            detections = sv.Detections.from_ultralytics(results[0])

            if detections.tracker_id is not None:
                # ------------------------------------------------------
                # [Task A] 전역 헬멧 미착용 감지 (Global No Helmet)
                # ------------------------------------------------------
                # 클래스 ID 필터링: Helmet(0) or No Helmet(1)
                valid_detections = detections[
                    (detections.class_id == CLASS_ID_HELMET) | 
                    (detections.class_id == CLASS_ID_NO_HELMET)
                ]

                # 미착용자 필터링
                no_helmet_detections = valid_detections[valid_detections.class_id == CLASS_ID_NO_HELMET]
                no_helmet_count = len(no_helmet_detections)

                if no_helmet_count > 0:
                    detected_events.append("NO_HELMET")
                    cv2.putText(frame, f"WARNING: NO HELMET ({no_helmet_count})", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                # ------------------------------------------------------
                # [Task B] 나홀로 작업 감지 로직 (Zone Logic)
                # ------------------------------------------------------
                for i, zone in enumerate(zones):
                    # 구역 내 사람(Helmet + No Helmet 모두 포함) 감지
                    is_inside = zone.trigger(detections=valid_detections)
                    zone_person_count = is_inside.sum()
                    
                    status_text = "OK"
                    
                    if zone_person_count == 1:
                        if zone_timers[i] is None:
                            zone_timers[i] = time.time()
                        
                        elapsed = time.time() - zone_timers[i]
                        if elapsed > LONE_WORKER_LIMIT:
                            detected_events.append(f"LONE_WORKER_ZONE_{i+1}")
                            status_text = f"ALARM! {int(elapsed)}s"
                        else:
                            status_text = f"WARN {int(elapsed)}s"
                    else:
                        zone_timers[i] = None # 0명이나 2명 이상이면 리셋
                        
                    # 구역 그리기
                    zone_annotators[i].annotate(scene=frame)
                    
                    # 텍스트 표시
                    text_pos = (polygons[i][0][0], polygons[i][0][1] - 10)
                    cv2.putText(frame, f"Z{i+1}: {zone_person_count}P {status_text}", 
                                text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # ------------------------------------------------------
                # [시각화] 바운딩 박스 & 라벨
                # ------------------------------------------------------
                labels = [
                    f"{yolo_model.model.names[class_id]} {confidence:.2f}"
                    for class_id, confidence
                    in zip(valid_detections.class_id, valid_detections.confidence)
                ]
                frame = box_annotator.annotate(scene=frame, detections=valid_detections)
                frame = label_annotator.annotate(scene=frame, detections=valid_detections, labels=labels)

            # ==========================================================
            # [Task C] 화재 감지 (ResNet Sliding Window)
            # ==========================================================
            if fire_model is not None:
                # (기존 코드 유지)
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                batch_tensors = []
                batch_coords = []
                
                for y in range(0, height - 256 + 1, 128):
                    for x in range(0, width - 256 + 1, 128):
                        patch = pil_img.crop((x, y, x + 256, y + 256))
                        batch_tensors.append(fire_preprocess(patch))
                        batch_coords.append((x, y))
                
                if batch_tensors:
                    batch_input = torch.stack(batch_tensors).to(DEVICE)
                    with torch.no_grad():
                        outputs = fire_model(batch_input)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        scores, preds = torch.max(probs, 1)

                        for i in range(len(preds)):
                            if preds[i] == 1 and scores[i] > 0.8: 
                                detected_events.append("FIRE")
                                fx, fy = batch_coords[i]
                                cv2.rectangle(frame, (fx, fy), (fx+256, fy+256), (0, 0, 255), 2)
                                cv2.putText(frame, "FIRE", (fx, fy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # ==========================================================
            # [종합] 이벤트 발생 시 DB 저장 & 로봇 명령
            # ==========================================================
            if detected_events:
                current_time = time.time()
                # 5초 쿨다운 (중복 알림 방지)
                if current_time - last_alert_time > 5.0: 
                    last_alert_time = current_time
                    unique_events = list(set(detected_events))
                    print(f"🚨 위험 감지: {unique_events}")
                    
                    # 1. 파일 저장
                    timestamp_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    img_filename = f"{timestamp_id}.jpg"
                    save_path = os.path.join(IMAGE_DIR, img_filename)
                    cv2.imwrite(save_path, frame)
                    
                    # 2. 로봇 명령 우선순위 결정
                    cmd_reason = "INTRUDER"
                    if "FIRE" in unique_events: cmd_reason = "FIRE"
                    elif any("LONE_WORKER" in e for e in unique_events): cmd_reason = "LONE_WORKER"
                    elif "NO_HELMET" in unique_events: cmd_reason = "PPE_VIOLATION"
                    
                    payload = {"command": "DISPATCH", "target_zone": "Zone_A", "reason": str(unique_events)}
                    if mqtt_client:
                        mqtt_client.publish("argus/robot/command", json.dumps(payload))
                    
                    # 3. DB 저장 (통합)
                    conn = get_db_connection()
                    if conn:
                        try:
                            cur = conn.cursor()
                            sql = """
                                INSERT INTO safety_logs 
                                (device_id, source_type, event_type, image_path, detail_info, created_at) 
                                VALUES (%s, %s, %s, %s, %s, NOW())
                            """
                            cur.execute(sql, ("SERVER_AI", "CCTV", cmd_reason, save_path, json.dumps({"events": unique_events})))
                            conn.commit()
                        except Exception as db_err:
                            print(f"DB Error: {db_err}")
                        finally:
                            conn.close()
            
            time.sleep(0.05) 

        except Exception as e:
            print(f"AI Loop Error: {e}")
            time.sleep(1)

threading.Thread(target=ai_processing_loop, daemon=True).start()

# ================= [Thread 3] FastAPI (대시보드 송출) =================                   

@app.get("/video_feed")
def video_feed():
    def iter_frames():
        while True:
            if latest_frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_frame_bytes + b'\r\n')
            time.sleep(0.04) 
    return StreamingResponse(iter_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/v1/devices")
def get_devices():
    conn = get_db_connection()
    if not conn: return []
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT *, 
        CASE WHEN last_heartbeat < NOW() - INTERVAL '1 minute' THEN 'OFFLINE' ELSE status END as calculated_status
        FROM device_status
    """)
    rows = cur.fetchall()
    conn.close()
    return rows

@app.get("/api/v1/logs")
def get_logs(source: Optional[str] = None, limit: int = 20):
    conn = get_db_connection()
    if not conn: return []
    cur = conn.cursor(cursor_factory=RealDictCursor)
    query = "SELECT * FROM safety_logs"
    params = []
    if source:
        query += " WHERE source_type = %s"
        params.append(source)
    query += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)
    cur.execute(query, tuple(params))
    rows = cur.fetchall()
    conn.close()
    return rows

@app.get("/api/v1/logs/{log_id}/image")
def get_log_image(log_id: int):
    conn = get_db_connection()
    if not conn: raise HTTPException(status_code=500, detail="DB Error")
    cur = conn.cursor()
    cur.execute("SELECT image_path FROM safety_logs WHERE id = %s", (log_id,))
    res = cur.fetchone()
    conn.close()
    if res and res[0] and os.path.exists(res[0]):
        return FileResponse(res[0])
    else:
        return {"error": "Image not found"}

@app.get("/api/v1/dashboard/stats")
def get_stats():
    conn = get_db_connection()
    if not conn: return {"active_alerts": 0, "system_status": "ERROR"}
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM safety_logs WHERE created_at > CURRENT_DATE")
    today_alerts = cur.fetchone()[0]
    conn.close()
    return {"active_alerts": today_alerts, "system_status": "ONLINE"}

class RobotCommand(BaseModel):
    command: str
    target_zone: Optional[str] = None

@app.post("/api/v1/robot/command")
def send_robot_command(cmd: RobotCommand):
    payload = {"action": cmd.command, "target": cmd.target_zone, "timestamp": str(datetime.now())}
    if mqtt_client:
        mqtt_client.publish("argus/robot/command", json.dumps(payload))
    
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            sql = "INSERT INTO safety_logs (device_id, source_type, event_type, image_path, detail_info, created_at) VALUES (%s, %s, %s, %s, %s, NOW())"
            cur.execute(sql, ("SERVER", "ADMIN", "COMMAND", "", json.dumps(payload)))
            conn.commit()
        finally:
            conn.close()
    return {"status": "Command Sent", "payload": payload}