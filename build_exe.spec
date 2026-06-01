# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 빌드 스펙 (경량판) — reclassification 단계 교착 회피용.

기존 spec 은 torch/torchvision/facenet 을 collect_all 로 통째로 긁어와
파일이 1만 개를 넘기고, 그 분류 단계에서 멈추는 경우가 있다.
이 경량판은:
  - torch/torchvision 은 PyInstaller 내장 훅에 맡긴다(collect_all 안 함).
  - facenet_pytorch 는 코드+데이터만 모으고, cv2/PyQt5/openpyxl 도 표준 훅에 맡긴다.
  - 모델 가중치(vggface2)와 PyQt5 platforms 플러그인만 명시적으로 포함.

빌드 (Windows + ASCII 경로):
    rmdir /s /q build dist
    pyinstaller build_exe_light.spec --noupx
"""
import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# facenet_pytorch 만 가볍게 수집 (코드 + 데이터). torch 는 내장 훅이 처리.
hiddenimports = collect_submodules("facenet_pytorch")
# 음성 안내(TTS): pyttsx3 는 드라이버를 동적 import 하므로 명시 포함
hiddenimports += [
    "pyttsx3", "pyttsx3.drivers", "pyttsx3.drivers.sapi5",
    "win32com", "win32com.client", "pythoncom", "pywintypes",
]
datas = collect_data_files("facenet_pytorch")

# ── 앱 아이콘 포함 (창/작업표시줄 아이콘용) ───────────────────────
ICON_FILE = "favicon.ico"
if os.path.exists(ICON_FILE):
    datas.append((ICON_FILE, "."))
    print(f"[spec-light] 아이콘 포함: {ICON_FILE}")
else:
    print(f"[spec-light] 경고: {ICON_FILE} 없음 — 기본 아이콘으로 빌드됩니다.")

# ── 모델 가중치(vggface2) 포함 ────────────────────────────────────
def _find_torch_cache():
    cands = [
        os.path.join(os.path.expanduser("~"), ".cache", "torch", "checkpoints"),
        os.path.join(os.environ.get("USERPROFILE", ""), ".cache", "torch", "checkpoints"),
    ]
    for c in cands:
        if c and os.path.isdir(c):
            return c
    return None

_cache = _find_torch_cache()
if _cache:
    for fname in os.listdir(_cache):
        if fname.endswith((".pt", ".pth")):
            datas.append(
                (os.path.join(_cache, fname), os.path.join(".cache", "torch", "checkpoints"))
            )
    print(f"[spec-light] 모델 가중치 포함: {_cache}")
else:
    print("[spec-light] 경고: vggface2 가중치 캐시 없음 (온라인 PC면 첫 실행 시 다운로드).")

block_cipher = None

a = Analysis(
    ["run_app.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=["rthook_torch_home.py"],
    excludes=["tensorboard", "matplotlib", "expecttest"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AttendanceScanner",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=ICON_FILE if os.path.exists(ICON_FILE) else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AttendanceScanner",
)
