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

# ================= 설정 (Configuration) =================
IMAGE_DIR = "saved_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# DB 설정 (Docker Compose service name: db)
DB_CONFIG = {
    "host": "db",  # Docker 내부라면 "db", 로컬 테스트면 "localhost"
    "database": "argus_db",
    "user": "argus_user",
    "password": "argus_password",
    "port": "5432"
}

# AI 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FIRE_MODEL_PATH = 'models/fire_classifier_resnet.pth'

# [Task 4] 나홀로 작업 경고 설정
LONE_WORKER_LIMIT = 180  # 3분 (초 단위)

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
yolo_model = YOLO('yolov8n.pt')   # 사람 감지용
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
        # 1. CCTV가 보낸 영상 데이터 수신
        if topic.endswith("/stream"):
            payload = json.loads(msg.payload.decode('utf-8'))
            
            # Base64 디코딩
            img_data = base64.b64decode(payload['img_base64'])
            
            # 1) 대시보드 송출용 (Bytes 그대로 저장 -> 빠름)
            latest_frame_bytes = img_data
            
            # 2) AI 분석용 (OpenCV 포맷으로 변환 -> 무거움, 필요할 때만 갖다 씀)
            # 매번 변환하면 느려질 수 있으므로, AI 스레드에서 변환하는 게 낫지만
            # 여기서는 최신 프레임을 즉시 갱신해둠.
            nparr = np.frombuffer(img_data, np.uint8)
            latest_frame_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    except Exception as e:
        print(f"⚠️ MQTT 수신 에러: {e}")

mqtt_client = mqtt.Client()
mqtt_client.on_message = on_message
mqtt_client.connect("localhost", 1883, 60) # Mosquitto 주소
mqtt_client.subscribe("argus/#")   # 모든 기기의 스트림 구독
mqtt_client.start_loop() # 별도 스레드에서 실행

# ================= [Thread 2] AI 분석 루프 (백그라운드) =================
def ai_processing_loop():
    global latest_frame_cv
    
    # ----------------------------------------------------
    # [설정] Task 1: 라인 카운팅 (LineZone)
    # ----------------------------------------------------
    START = sv.Point(50, 200)   # ⚠️ 실제 CCTV 좌표에 맞게 수정 필요
    END = sv.Point(600, 200)    # ⚠️ 실제 CCTV 좌표에 맞게 수정 필요
    
    line_zone = sv.LineZone(start=START, end=END, triggering_anchors=[sv.Position.BOTTOM_CENTER])
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=0.8)
    box_annotator = sv.BoxAnnotator(thickness=2)

    # ----------------------------------------------------
    # Task 4: 나홀로 작업 구역 (PolygonZone)
    # ----------------------------------------------------
    polygons = [
        np.array([[48, 222], [328, 220], [306, 612], [48, 652]]) # Zone 1
    ]
    zones = []
    zone_annotators = []
    zone_timers = [None] * len(polygons)

    for polygon in polygons:
        zone = sv.PolygonZone(polygon=polygon, triggering_anchors=[sv.Position.CENTER])
        zones.append(zone)
        zone_annotators.append(sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED, thickness=2))

    # 상태 변수
    last_alert_time = 0
    
    print("🚀 AI 분석 엔진 가동 (YOLO + ResNet + Supervision)")
    
    while True:
        if latest_frame_cv is None:
            time.sleep(0.1)
            continue
            
        try:
            frame = latest_frame_cv.copy()
            height, width, _ = frame.shape
            
            detected_events = [] 
            
            # ==========================================================
            # [Step 1] YOLO 추론 & 트래킹 (한 번 실행해서 모두 공유)
            # ==========================================================
            # persist=True로 ID 유지 (Supervision에 필수)
            results = yolo_model.track(frame, persist=True, verbose=False, conf=0.5, classes=[0]) # 0=person
            
            # Supervision Detections 변환
            detections = sv.Detections.from_ultralytics(results[0])
            if detections.tracker_id is not None:
                detections = detections[detections.class_id == 0] # 사람만 필터링

                # ------------------------------------------------------
                # [Task 1] 라인 카운팅 로직
                # ------------------------------------------------------
                line_zone.trigger(detections=detections)
                
                # 라인 그리기
                line_annotator.annotate(frame=frame, line_counter=line_zone)
                
                # 사람 수 업데이트 (In - Out)
                current_people = line_zone.in_count - line_zone.out_count
                cv2.putText(frame, f"Count: {current_people}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # ------------------------------------------------------
                # [Task 4] 나홀로 작업 감지 로직
                # ------------------------------------------------------
                for i, zone in enumerate(zones):
                    is_inside = zone.trigger(detections=detections)
                    zone_person_count = is_inside.sum()
                    
                    status_text = "OK"
                    
                    if zone_person_count == 1:
                        if zone_timers[i] is None:
                            zone_timers[i] = time.time()
                        
                        elapsed = time.time() - zone_timers[i]
                        if elapsed > LONE_WORKER_LIMIT:
                            detected_events.append("LONE_WORKER")
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
                
                # 사람 박스 그리기
                frame = box_annotator.annotate(scene=frame, detections=detections)


            # ==========================================================
            # [Task 2] 화재 감지 (ResNet Sliding Window)
            # ==========================================================
            if fire_model is not None:
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                batch_tensors = []
                batch_coords = []
                
                # 윈도우 생성
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
                            if preds[i] == 1 and scores[i] > 0.8: # Threshold 0.8
                                detected_events.append("FIRE")
                                fx, fy = batch_coords[i]
                                cv2.rectangle(frame, (fx, fy), (fx+256, fy+256), (0, 0, 255), 2)
                                cv2.putText(frame, "FIRE", (fx, fy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


            # ==========================================================
            # [종합] 이벤트 발생 시 처리 (DB 저장 & 로봇 명령)
            # ==========================================================
            if detected_events:
                current_time = time.time()
                if current_time - last_alert_time > 5.0: # 5초 쿨다운
                    last_alert_time = current_time
                    unique_events = list(set(detected_events))
                    print(f"🚨 위험 감지: {unique_events}")
                    
                    # 1. 파일 저장
                    timestamp_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    img_filename = f"{timestamp_id}.jpg"
                    save_path = os.path.join(IMAGE_DIR, img_filename)
                    cv2.imwrite(save_path, frame)
                    
                    # 2. 로봇 명령
                    cmd_reason = "INTRUDER"
                    if "FIRE" in unique_events: cmd_reason = "FIRE"
                    elif "LONE_WORKER" in unique_events: cmd_reason = "LONE_WORKER"
                    
                    payload = {"command": "DISPATCH", "target_zone": "Zone_A", "reason": str(unique_events)}
                    mqtt_client.publish("aragog/robot/command", json.dumps(payload))
                    
                    # 3. DB 저장
                    conn = get_db_connection()
                    if conn:
                        cur = conn.cursor()
                        sql = "INSERT INTO safety_logs (device_id, source_type, event_type, image_path, detail_info, created_at) VALUES (%s, %s, %s, %s, %s, NOW())"
                        cur.execute(sql, ("SERVER_AI", "CCTV", cmd_reason, save_path, json.dumps({"events": unique_events})))
                        conn.commit()
                        conn.close()
            
            time.sleep(0.1) # 루프 주기 조절

        except Exception as e:
            print(f"AI Loop Error: {e}")
            time.sleep(1)

threading.Thread(target=ai_processing_loop, daemon=True).start()

# ================= [Thread 3] FastAPI (대시보드 송출) =================

# 1. [기능 D-C01] 실시간 영상 스트리밍
# 설명: MQTT 스레드가 업데이트하는 latest_frame_bytes를 가져와 송출
@app.get("/video_feed")
def video_feed():
    def iter_frames():
        while True:
            if latest_frame_bytes: # 전역 변수 참조
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_frame_bytes + b'\r\n')
            time.sleep(0.04) # 약 25 FPS 제한 (대시보드 부하 방지)
            
    return StreamingResponse(iter_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# 2. [기능 D-G01] 기기 상태 조회 (Heartbeat 기반)
@app.get("/api/v1/devices")
def get_devices():
    conn = get_db_connection()
    if not conn: return [] # DB 연결 실패 시 빈 리스트 반환
    
    cur = conn.cursor(cursor_factory=RealDictCursor)
    # 마지막 통신이 1분 지났으면 OFFLINE으로 간주
    cur.execute("""
        SELECT *, 
        CASE 
            WHEN last_heartbeat < NOW() - INTERVAL '1 minute' THEN 'OFFLINE'
            ELSE status 
        END as calculated_status
        FROM device_status
    """)
    rows = cur.fetchall()
    conn.close()
    return rows

# 3. [기능 D-ARC01] 위험 로그 조회 (필터링 지원)
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

# 4. [기능 D-ARC01] 로그 이미지 다운로드
@app.get("/api/v1/logs/{log_id}/image")
def get_log_image(log_id: int):
    conn = get_db_connection()
    if not conn: raise HTTPException(status_code=500, detail="DB Error")
    
    cur = conn.cursor()
    cur.execute("SELECT image_path FROM safety_logs WHERE id = %s", (log_id,))
    res = cur.fetchone()
    conn.close()
    
    # 파일이 실제로 존재하는지 확인 후 전송
    if res and res[0] and os.path.exists(res[0]):
        return FileResponse(res[0])
    else:
        return {"error": "Image not found"}

# 5. [기능 D-T01] 대시보드 통계 (오늘 발생한 알림 수)
@app.get("/api/v1/dashboard/stats")
def get_stats():
    conn = get_db_connection()
    if not conn: return {"active_alerts": 0, "system_status": "ERROR"}
    
    cur = conn.cursor()
    # 오늘 날짜 이후 발생한 로그 카운트
    cur.execute("SELECT COUNT(*) FROM safety_logs WHERE created_at > CURRENT_DATE")
    today_alerts = cur.fetchone()[0]
    conn.close()
    
    return {"active_alerts": today_alerts, "system_status": "ONLINE"}

# 6. [기능 D-CMD] 로봇 제어 명령 (WPF -> Server -> Robot)
class RobotCommand(BaseModel):
    command: str          # 예: "DISPATCH", "RETURN"
    target_zone: Optional[str] = None

@app.post("/api/v1/robot/command")
def send_robot_command(cmd: RobotCommand):
    # 1. 명령 패킷 생성
    payload = {
        "action": cmd.command,
        "target": cmd.target_zone,
        "timestamp": str(datetime.now())
    }
    
    # 2. MQTT로 로봇에게 명령 전달 (Fire-and-forget)
    if mqtt_client:
        mqtt_client.publish("aragog/robot/command", json.dumps(payload))
        print(f"🎮 Command Sent: {payload}")
    else:
        print("⚠️ MQTT Client Not Ready")

    # 3. DB에 명령 이력 저장 (History Log)
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            sql = """
                INSERT INTO safety_logs (device_id, source_type, event_type, image_path, detail_info, created_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
            """
            # detail_info에 명령 내용을 JSON으로 저장
            detail_json = json.dumps(payload)
            
            # device_id='SERVER', source='ADMIN', event='COMMAND' 로 구분해서 저장
            cur.execute(sql, ("SERVER", "ADMIN", "COMMAND", "", detail_json))
            conn.commit()
            print("✅ 명령 이력 DB 저장 완료")
        except Exception as e:
            print(f"❌ DB Log Error: {e}")
        finally:
            conn.close()
    
    return {"status": "Command Sent", "payload": payload}