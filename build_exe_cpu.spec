# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec — CPU-only build (no NVIDIA GPU required).

Build inside a CPU-only venv (torch installed from the cpu index):
    python -m venv .venv_cpu
    .venv_cpu\Scripts\activate
    pip install -r requirements_cpu.txt
    rmdir /s /q build dist
    pyinstaller build_exe_cpu.spec

This is the same as the GPU spec except:
  - output is named AttendanceScanner_CPU (so it won't clash with the GPU build)
  - CUDA libraries are explicitly excluded (in case any linger on the system)
The app code itself is unchanged; face_engine auto-selects CPU when no GPU.
"""
import glob
import os
import sysconfig

from PyInstaller.utils.hooks import (
    collect_data_files, collect_submodules, collect_dynamic_libs,
)

# facenet_pytorch: collect code + data.
hiddenimports = collect_submodules("facenet_pytorch")

# torch: 강제 필터링 제거. 전체 서브모듈 수집
# (일부 모듈을 강제로 제외할 경우 torch._C 로딩 실패 에러 발생 가능성 방지)
hiddenimports += collect_submodules("torch")
hiddenimports += ["torch._C", "torchvision"]

# TTS: pyttsx3 imports its driver dynamically, so list it explicitly.
hiddenimports += [
    "pyttsx3", "pyttsx3.drivers", "pyttsx3.drivers.sapi5",
    "win32com", "win32com.client", "pythoncom", "pywintypes",
]

datas = collect_data_files("facenet_pytorch")
datas += collect_data_files("torch")

# OpenCV video I/O backends (camera) + torch/torchvision native DLLs.
# Without the cv2 libs the built exe can't open a webcam; without the torch
# libs torch._C fails to load.
datas += collect_data_files("cv2")
binaries = collect_dynamic_libs("cv2")
binaries += collect_dynamic_libs("torch")
binaries += collect_dynamic_libs("torchvision")

# --- OpenCV FFmpeg 플러그인 DLL 추가 배치 (동영상 "파일" 업로드 기능용) ---
# collect_dynamic_libs("cv2")만으로는 빌드된 exe에서 cv2.VideoCapture(파일경로)가
# 실패하는 경우가 있다 (webcam은 MSMF/DSHOW를 쓰므로 영향 없음 — 코드 쪽엔 이미
# MSMF 우선 시도 폴백을 넣어뒀지만, 일부 코덱/포맷은 FFmpeg DLL이 꼭 필요함).
# OpenCV가 이 DLL을 찾는 위치가 환경에 따라 달라서, exe와 같은 폴더(".")와
# cv2 패키지 폴더("cv2") 두 곳에 모두 복사해 넣어 이중으로 방지한다.
def _find_cv2_ffmpeg_dlls():
    site_packages = sysconfig.get_paths()["purelib"]
    return glob.glob(
        os.path.join(site_packages, "cv2", "opencv_videoio_ffmpeg*.dll")
    )

_ffmpeg_dlls = _find_cv2_ffmpeg_dlls()
if _ffmpeg_dlls:
    for _dll in _ffmpeg_dlls:
        binaries.append((_dll, "."))
        binaries.append((_dll, "cv2"))
        print(f"[spec-cpu] ffmpeg plugin included (2 locations): {os.path.basename(_dll)}")
else:
    print(
        "[spec-cpu] warning: opencv ffmpeg plugin DLL not found — "
        "video FILE upload may still fail in some formats (webcam recording is unaffected)."
    )

# --- App icon (window / taskbar) ---
ICON_FILE = "favicon.ico"
if os.path.exists(ICON_FILE):
    datas.append((ICON_FILE, "."))
    print(f"[spec-cpu] icon included: {ICON_FILE}")
else:
    print(f"[spec-cpu] warning: {ICON_FILE} not found - building with default icon.")

# --- Bundle vggface2 weights from torch cache ---
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
    print(f"[spec-cpu] model weights included: {_cache}")
else:
    print("[spec-cpu] warning: vggface2 weights cache not found (will download on first run if online).")

# Exclude only the external CUDA packages (not torch's own cuda submodules,
# which CPU torch still imports as stubs and must load cleanly). This keeps
# the build small without breaking torch import.
cuda_excludes = [
    "nvidia", "nvidia.cublas", "nvidia.cuda_runtime", "nvidia.cudnn",
    "nvidia.cufft", "nvidia.curand", "nvidia.cusolver", "nvidia.cusparse",
    "nvidia.nccl", "nvidia.nvtx", "triton",
]

# 수정됨: torch_excludes 변수 및 관련 로직 제거 

block_cipher = None

a = Analysis(
    ["run_app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=["rthook_torch_home.py"],
    excludes=["tensorboard", "matplotlib", "expecttest"] + cuda_excludes, # torch_excludes 제거
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
    name="AttendanceScanner_CPU",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # 배포용. 디버깅이 필요하면 True 로 바꿔 재빌드
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
    name="AttendanceScanner_CPU",
)