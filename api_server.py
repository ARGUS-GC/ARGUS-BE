from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from psycopg2.extras import RealDictCursor
import paho.mqtt.client as mqtt
import json
import base64
import time
import threading

app = FastAPI()

# --- CORS 설정 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 전역 변수: 최신 영상 프레임 저장소 ---
latest_frame = None

# --- MQTT 설정 (영상을 받는 기능) ---
def on_message(client, userdata, msg):
    global latest_frame
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        # Base64 -> 이미지 변환
        img_data = base64.b64decode(payload['img_base64'])
        latest_frame = img_data
    except Exception as e:
        pass

# MQTT 실행 (백그라운드)
mqtt_client = mqtt.Client()
mqtt_client.on_message = on_message
try:
    # 모스키토가 로컬(도커)에 있으므로 localhost로 접속
    mqtt_client.connect("localhost", 1883, 60)
    mqtt_client.subscribe("argus/cctv/stream")
    mqtt_client.loop_start()
    print("📡 영상 수신 준비 완료 (MQTT Connected)")
except Exception as e:
    print(f"⚠️ MQTT 연결 실패: {e}")

# --- DB 설정 ---
DB_CONFIG = {
    "host": "localhost",
    "database": "argus_db",
    "user": "argus_user",
    "password": "argus_password",
    "port": "5432"
}

def get_db_connection():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except:
        return None

# --- [핵심] 영상 스트리밍 주소 ---
def generate_frames():
    while True:
        if latest_frame:
            # MJPEG 포맷으로 실시간 전송
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
        time.sleep(0.05) 

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# --- 기존 로그 API ---
@app.get("/api/logs")
def read_logs():
    conn = get_db_connection()
    if not conn: return []
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM safety_logs ORDER BY id DESC LIMIT 10")
    rows = cur.fetchall()
    conn.close()
    return rows

@app.get("/api/status")
def read_status():
    return {"system_status": "NORMAL", "active_alerts": 0}
