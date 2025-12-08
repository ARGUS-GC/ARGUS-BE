import cv2
import time
import json
import base64
import paho.mqtt.client as mqtt
from datetime import datetime

# 설정
MQTT_BROKER = "localhost"
VIDEO_SOURCE = 0  # 0: 웹캠, "video.mp4": 동영상 파일
DEVICE_ID = "test_cctv_01"

# MQTT 연결
client = mqtt.Client()
client.connect(MQTT_BROKER, 1883, 60)
client.loop_start()

cap = cv2.VideoCapture(VIDEO_SOURCE)

print(f"🎥 가상 CCTV({DEVICE_ID}) 시작... (종료: Ctrl+C)")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. 이미지 크기 줄이기 (전송 속도 위해)
        frame = cv2.resize(frame, (640, 480))

        # 2. Base64 인코딩
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # 3. 전송 (stream 토픽)
        payload = {
            "device_id": DEVICE_ID,
            "img_base64": img_base64,
            "timestamp": str(datetime.now())
        }
        client.publish("argus/cctv/stream", json.dumps(payload))
        
        # 4. 상태 전송 (status 토픽) - 3초에 한 번만 보내도 됨 (여기선 생략하거나 가끔 전송)
        
        time.sleep(0.05) # 약 20 FPS 조절

except KeyboardInterrupt:
    print("종료 중...")

cap.release()
client.loop_stop()