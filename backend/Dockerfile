FROM python:3.9-slim

# [변경점] libgl1-mesa-glx 대신 libgl1 사용, 줄바꿈 없이 한 줄로 작성하여 오류 방지
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

CMD ["python", "-u", "main.py"]
