-- attendance.db 최초 생성 스크립트
-- 실행:
--   python -c "import sqlite3; c=sqlite3.connect('/mnt/nas/attendance.db'); c.executescript(open('init_db.sql').read()); c.close()"
-- 또는 utils/database.py 의 AttendanceDB() 인스턴스 생성 시 자동 실행됨.

-- ── 직원 테이블 ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS Employees (
    emp_id    TEXT    PRIMARY KEY,
    name      TEXT    NOT NULL,
    embedding BLOB    NOT NULL   -- 512-dim float32, tobytes() 직렬화
);

-- ── 근태 로그 테이블 ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS AttendanceLog (
    log_id  INTEGER  PRIMARY KEY AUTOINCREMENT,
    emp_id  TEXT     NOT NULL,
    date    TEXT     NOT NULL,   -- YYYY-MM-DD
    time    TEXT     NOT NULL,   -- HH:MM:SS
    state   TEXT     NOT NULL,   -- '출근' | '외출' | '복귀' | '퇴근'
    FOREIGN KEY (emp_id) REFERENCES Employees(emp_id)
);

-- ── 인덱스: 당일 조회 속도 최적화 ──────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_log_emp_date
    ON AttendanceLog(emp_id, date);

-- ── 사용 예시 쿼리 ───────────────────────────────────────────────
-- 오늘 특정 직원의 마지막 상태 확인:
--   SELECT state FROM AttendanceLog
--     WHERE emp_id='E001' AND date='2025-01-15'
--     ORDER BY log_id DESC LIMIT 1;
