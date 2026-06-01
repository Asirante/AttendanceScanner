"""SQLite 근태 DB + 상태 머신."""
import os
import sqlite3
from datetime import date, datetime
from typing import List, Optional

from . import DB_PATH

class AttendanceDB:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DB_PATH

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS Employees (
                emp_id    TEXT  PRIMARY KEY,
                name      TEXT  NOT NULL,
                embedding BLOB  NOT NULL    -- 512-dim float32 직렬화
            );
            CREATE TABLE IF NOT EXISTS AttendanceLog (
                log_id  INTEGER  PRIMARY KEY AUTOINCREMENT,
                emp_id  TEXT     NOT NULL,
                date    TEXT     NOT NULL,
                time    TEXT     NOT NULL,
                state   TEXT     NOT NULL,
                FOREIGN KEY (emp_id) REFERENCES Employees(emp_id)
            );
            CREATE INDEX IF NOT EXISTS idx_log_emp_date
                ON AttendanceLog(emp_id, date);
            """
        )
        self.conn.commit()

    TRANSITIONS = {
        None: ["출근"],
        "출근": ["외출", "퇴근"],
        "외출": ["복귀"],
        "복귀": ["외출", "퇴근"],
        "퇴근": [],
    }

    def get_last_state(self, emp_id: str) -> Optional[str]:
        today = date.today().isoformat()
        cur = self.conn.execute(
            "SELECT state FROM AttendanceLog "
            "WHERE emp_id=? AND date=? ORDER BY log_id DESC LIMIT 1",
            (emp_id, today),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def get_available_actions(self, emp_id: str) -> List[str]:
        """현재 상태 → 가능한 액션 리스트 반환."""
        return self.TRANSITIONS.get(self.get_last_state(emp_id), [])

    def log_action(self, emp_id: str, state: str):
        """유효성 검사 후 로그 INSERT."""
        valid = self.get_available_actions(emp_id)
        if state not in valid:
            raise ValueError(
                f"유효하지 않은 전이: {self.get_last_state(emp_id)} → {state}"
            )
        now = datetime.now()
        self.conn.execute(
            "INSERT INTO AttendanceLog (emp_id, date, time, state) VALUES (?,?,?,?)",
            (emp_id, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), state),
        )
        self.conn.commit()

    def get_employee_name(self, emp_id: str) -> str:
        cur = self.conn.execute(
            "SELECT name FROM Employees WHERE emp_id=?", (emp_id,)
        )
        row = cur.fetchone()
        return row[0] if row else emp_id

    def close(self):
        self.conn.close()
