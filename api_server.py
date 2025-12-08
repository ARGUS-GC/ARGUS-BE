import os
import time
import json
import base64
import uuid
import threading
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
import paho.mqtt.client as mqtt

# --- [설정] 파일 저장 경로 생성 ---
IMAGE_DIR = "saved_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

app = FastAPI(title="Aragog Industrial Safety System API")

# --- CORS 설정 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DB 설정 ---
DB_CONFIG = {
    "host": "localhost",
    "database": "argus_db",
    "user": "argus_user",
    "password": "argus_password",
    "port": "5432"
}

# --- 전역 변수: 실시간 스트리밍용 ---
latest_frame = None

def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"❌ DB Connection Failed: {e}")
        return None

# --- [Core] MQTT 메시지 처리 로직 ---
# 서버 계획서의 Data Flow 참조 
def on_message(client, userdata, msg):
    global latest_frame
    topic = msg.topic
    
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        
        # 1. 상태 업데이트 (Heartbeat) 
        # 토픽 예: aragog/robot/status, aragog/cctv/status
        if topic.endswith("/status"):
            handle_status_update(payload)

        # 2. 위험 이벤트 발생 (Event) 
        # 토픽 예: aragog/robot/event
        elif topic.endswith("/event"):
            handle_event_log(payload)
        
        # 3. 실시간 스트리밍 (기존 기능 유지 + CCTV)
        elif topic.endswith("/stream"):
             if 'img_base64' in payload:
                latest_frame = base64.b64decode(payload['img_base64'])

    except Exception as e:
        print(f"⚠️ MQTT Process Error: {e}")

def handle_status_update(data):
    """
    [기능 명세 D-G01] 시스템 상태 표시 및 Heartbeat 처리
    DB 테이블: device_status [cite: 77]
    """
    conn = get_db_connection()
    if not conn: return
    try:
        cur = conn.cursor()
        # 기기 상태 Upsert (없으면 Insert, 있으면 Update)
        sql = """
            INSERT INTO device_status (device_id, type, status, last_heartbeat, metrics)
            VALUES (%s, %s, %s, NOW(), %s)
            ON CONFLICT (device_id) 
            DO UPDATE SET 
                status = EXCLUDED.status,
                last_heartbeat = NOW(),
                metrics = EXCLUDED.metrics;
        """
        # metrics는 JSONB로 저장
        metrics_json = json.dumps(data.get("metrics", {}))
        cur.execute(sql, (data.get("device_id"), data.get("type"), "ONLINE", metrics_json))
        conn.commit()
    except Exception as e:
        print(f"Error updating status: {e}")
    finally:
        conn.close()

def handle_event_log(data):
    """
    [기능 명세 D-G03, Server Plan p.7] 위험 감지 시 로그 및 이미지 저장
    DB 테이블: safety_logs 
    """
    conn = get_db_connection()
    if not conn: return

    try:
        # 1. Base64 이미지 디코딩 및 파일 저장 
        img_filename = f"{uuid.uuid4()}.jpg"
        img_path = os.path.join(IMAGE_DIR, img_filename)
        
        if "img_base64" in data:
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(data["img_base64"]))
        else:
            img_path = None # 이미지가 없는 이벤트일 경우

        # 2. DB Insert [cite: 152]
        cur = conn.cursor()
        sql = """
            INSERT INTO safety_logs (device_id, source_type, event_type, image_path, detail_info, created_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
        """
        detail_info = json.dumps({
            "location": data.get("location_label"),
            "severity": data.get("severity")
        })
        
        cur.execute(sql, (
            data.get("device_id"),
            data.get("source_type"), # CCTV or ROBOT
            data.get("event_type"),  # FIRE, FALL 등
            img_path,
            detail_info
        ))
        conn.commit()
        print(f"🔥 Danger Log Saved: {data.get('event_type')} from {data.get('device_id')}")
        
    except Exception as e:
        print(f"Error saving event: {e}")
    finally:
        conn.close()

# --- MQTT 실행 ---
mqtt_client = mqtt.Client()
mqtt_client.on_message = on_message

try:
    mqtt_client.connect("localhost", 1883, 60)
    #  모든 aragog 관련 토픽 구독
    mqtt_client.subscribe("argus/#") 
    mqtt_client.loop_start()
    print("📡 MQTT Server Connected (Topic: argus/#)")
except Exception as e:
    print(f"⚠️ MQTT Connection Failed: {e}")


# --- API 구현 ---

# 1. [기능 D-C01] 실시간 영상 스트리밍 (MJPEG)
def generate_frames():
    while True:
        if latest_frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
        time.sleep(0.05) 

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# 2. [Server Plan p.3] 기기 상태 조회 API
# 기능: D-G01 시스템 상태 표시
@app.get("/api/v1/devices")
def get_devices():
    conn = get_db_connection()
    if not conn: raise HTTPException(status_code=500, detail="DB Error")
    
    cur = conn.cursor(cursor_factory=RealDictCursor)
    # 1분 이상 heartbeat 없으면 OFFLINE으로 간주하는 로직 포함 가능
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

# 3. [Server Plan p.3] 위험 로그 조회 API
# 기능: D-ARC01 사고 카드 리스트, 필터링
@app.get("/api/v1/logs")
def get_logs(source: Optional[str] = None, limit: int = 20):
    conn = get_db_connection()
    if not conn: raise HTTPException(status_code=500, detail="DB Error")
    
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

# 4. [Server Plan p.3] 증거 이미지 다운로드 API
@app.get("/api/v1/logs/{log_id}/image")
def get_log_image(log_id: int):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT image_path FROM safety_logs WHERE id = %s", (log_id,))
    res = cur.fetchone()
    conn.close()
    
    if res and res[0] and os.path.exists(res[0]):
        return FileResponse(res[0])
    else:
        return {"error": "Image not found"}

# 5. [기능 D-T01] 대시보드 통계 API
@app.get("/api/v1/dashboard/stats")
def get_stats():
    conn = get_db_connection()
    if not conn: return {}
    cur = conn.cursor()
    
    # 미해결 Active Alert 개수 (예시: 오늘 발생한 건수)
    cur.execute("SELECT COUNT(*) FROM safety_logs WHERE created_at > CURRENT_DATE")
    today_alerts = cur.fetchone()[0]
    
    conn.close()
    return {"active_alerts": today_alerts, "system_status": "ONLINE"}

# 6. [기능 D-CMD01~03] 로봇 제어 명령 API
# 명세서 F. 로봇 제어 명령 참조
class RobotCommand(BaseModel):
    command: str  # "start_patrol", "return_home", "stop"
    target_zone: Optional[str] = None

@app.post("/api/v1/robot/command")
def send_robot_command(cmd: RobotCommand):
    """
    WPF 클라이언트에서 요청을 받아 로봇에게 MQTT로 명령 전송
    """
    payload = {
        "action": cmd.command,
        "target": cmd.target_zone,
        "timestamp": str(datetime.now())
    }
    # 로봇이 구독하고 있는 커맨드 토픽으로 발행
    mqtt_client.publish("aragog/robot/command", json.dumps(payload))
    return {"status": "Command Sent", "payload": payload}