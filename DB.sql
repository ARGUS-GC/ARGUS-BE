-- 1. 통합 기기 상태 테이블
CREATE TABLE IF NOT EXISTS device_status (
    device_id VARCHAR(50) PRIMARY KEY,
    type VARCHAR(20),      -- ROBOT, CCTV
    status VARCHAR(20),    -- ONLINE, OFFLINE
    last_heartbeat TIMESTAMP,
    metrics JSONB          -- 배터리, 위치 등
);

-- 2. 통합 위험 로그 테이블
CREATE TABLE IF NOT EXISTS safety_logs (
    id SERIAL PRIMARY KEY,
    device_id VARCHAR(50),
    source_type VARCHAR(20), -- ROBOT, CCTV
    event_type VARCHAR(20),  -- FIRE, FALL 등
    image_path TEXT,         -- 서버 로컬 파일 경로
    detail_info JSONB,       -- 좌표 등 추가 정보
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);