"""utils 패키지 초기화."""
import os

try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except Exception:

    pass

DB_PATH = os.environ.get("NAS_DB_PATH", "/mnt/nas/attendance.db")
