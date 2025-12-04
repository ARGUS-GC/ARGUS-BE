import paho.mqtt.client as mqtt
import psycopg2
import json
import base64
import cv2
import numpy as np
from ultralytics import YOLO
import os

# DB 접속 정보 (Docker 내부 통신이라 호스트명이 'db')
DB_HOST = "db"
DB_NAME = "argus_db"
DB_USER = "argus_user"
DB_PASS = "argus_password"

# YOLO 모델 로드 (가벼운 Nano 버전)
print("⏳ YOLO 모델 로딩 중... (처음엔 다운로드 때문에 오래 걸림)")
model = YOLO('yolov8n.pt')
print("✅ 모델 로딩 완료!")

def get_db():
    return psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)

def on_connect(client, userdata, flags, rc):
    print(f"📡 MQTT 연결 성공 (코드: {rc})")
    client.subscribe("argus/cctv/stream")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode('utf-8'))

        # 1. Base64 -> 이미지 변환
        img_data = base64.b64decode(payload['img_base64'])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. YOLO 추론 (신뢰도 0.5 이상만)
        results = model(frame, conf=0.5, verbose=False)

        detected = []
        for r in results:
            for box in r.boxes:
                cls_name = model.names[int(box.cls[0])]
                # 사람이나 불이 보이면 추가
                if cls_name in ['person', 'fire', 'dog', 'cat']:
                    detected.append(cls_name)

        # 3. 감지된 게 있으면 DB 저장 & 로그 출력
        if detected:
            det_str = ", ".join(set(detected))
            print(f"🚨 [위험 감지] {det_str} 발견!")

            conn = get_db()
            cur = conn.cursor()
            sql = """
                INSERT INTO safety_logs (device_id, source_type, event_type, image_path, detail_info)
                VALUES (%s, %s, %s, %s, %s)
            """
            # 이미지 경로는 일단 비워둠 (나중에 S3/파일저장 추가 가능)
            # [수정됨] 이미지를 Base64 문자열 그대로 DB에 저장합니다.
            # 이렇게 하면 프론트엔드가 <img src="...">에 바로 넣어서 보여줄 수 있습니다.
            image_data = f"data:image/jpeg;base64,{payload['img_base64']}"

            cur.execute(sql, (payload['device_id'], 'CCTV', det_str.upper(), image_data, json.dumps(payload['detail_info'])))
            #cur.execute(sql, (payload['device_id'], 'CCTV', det_str.upper(), '', json.dumps(payload['detail_info'])))
            conn.commit()
            cur.close()
            conn.close()

    except Exception as e:
        print(f"❌ 에러 발생: {e}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# Docker 내부에서 Mosquitto 접속 시 호스트명은 'mosquitto'
client.connect("mosquitto", 1883, 60)
client.loop_forever()
