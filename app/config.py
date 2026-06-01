"""앱 설정 관리 — DB 경로를 런타임에 설정/저장."""

import json
import os
import sys


def app_dir() -> str:
    """실행 파일(또는 스크립트)이 위치한 디렉토리."""
    if getattr(sys, "frozen", False):

        return os.path.dirname(sys.executable)

    return os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )


def resource_path(name: str) -> str:
    """번들된 리소스(아이콘 등) 경로. PyInstaller면 _MEIPASS, 아니면 프로젝트 루트."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    # 개발 환경 또는 exe 옆에 둔 경우
    return os.path.join(app_dir(), name)


CONFIG_PATH = os.path.join(app_dir(), "config.json")
DEFAULT_DB_NAME = "attendance.db"


def _default_db_path() -> str:
    return os.path.join(app_dir(), DEFAULT_DB_NAME)


def load_config() -> dict:
    """설정 로드. 파일이 없으면 기본값으로 새로 생성·저장한다."""
    existed = os.path.exists(CONFIG_PATH)
    if existed:
        try:
            with open(
                CONFIG_PATH, "r", encoding="utf-8"
            ) as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
            existed = False  # 깨진 파일이면 새로 쓰도록
    else:
        cfg = {}

    cfg.setdefault("db_path", _default_db_path())
    cfg.setdefault("sim_threshold", 0.85)
    cfg.setdefault("vote_n", 5)
    cfg.setdefault("use_fp16", True)
    cfg.setdefault("camera_index", 0)
    cfg.setdefault("attendance_mode", "simple")
    cfg.setdefault("cooldown_sec", 1.0)
    cfg.setdefault("auto_mode", True)
    cfg.setdefault("dwell_sec", 2.0)
    cfg.setdefault("auto_preview", True)
    cfg.setdefault("auto_popup", True)
    cfg.setdefault("popup_sec", 2.0)
    cfg.setdefault("tts_enabled", True)

    # 첫 실행(파일 없음/깨짐) 시 기본값으로 config.json 생성
    if not existed:
        try:
            save_config(cfg)
        except Exception:
            pass
    return cfg


def save_config(cfg: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def get_db_path() -> str:
    return load_config()["db_path"]


def set_db_path(path: str):
    cfg = load_config()
    cfg["db_path"] = path
    save_config(cfg)
