# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec — CPU-only build (no NVIDIA GPU required).

Build inside a CPU-only venv (torch installed from the cpu index):
    python -m venv .venv_cpu
    .venv_cpu\\Scripts\\activate
    pip install -r requirements_cpu.txt
    rmdir /s /q build dist
    pyinstaller build_exe_cpu.spec

This is the same as the GPU spec except:
  - output is named AttendanceScanner_CPU (so it won't clash with the GPU build)
  - CUDA libraries are explicitly excluded (in case any linger on the system)
The app code itself is unchanged; face_engine auto-selects CPU when no GPU.
"""
import os

from PyInstaller.utils.hooks import (
    collect_data_files, collect_submodules, collect_dynamic_libs,
)

# facenet_pytorch: collect code + data.
hiddenimports = collect_submodules("facenet_pytorch")

# torch: the built-in hook sometimes misses the C extension (torch._C),
# causing "NameError: name '_C' is not defined" at runtime. Collect torch
# submodules + DLLs, but DROP heavy unused subpackages (testing, distributed,
# onnx, quantization, benchmark...) — those add thousands of files and make the
# build extremely slow without being used by an inference-only app.
_TORCH_SKIP = (
    "torch.testing", "torch.distributed", "torch.onnx", "torch.quantization",
    "torch.ao", "torch.utils.benchmark", "torch.utils.tensorboard",
    "torch.utils.bottleneck", "torch.utils.data.datapipes", "torch.profiler",
    "torch.package", "torch._dynamo", "torch._inductor",
    "torch.utils.viz", "torch.utils.model_dump",
)
def _keep(mod):
    return not any(mod == s or mod.startswith(s + ".") for s in _TORCH_SKIP)

hiddenimports += [m for m in collect_submodules("torch") if _keep(m)]
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

# Heavy torch subpackages an inference-only app never uses. Excluding them
# cuts the file count (and build time) dramatically.
torch_excludes = [
    "torch.testing", "torch.distributed", "torch.onnx", "torch.quantization",
    "torch.ao", "torch.utils.benchmark", "torch.utils.tensorboard",
    "torch.utils.bottleneck", "torch.profiler", "torch.package",
    "torch._dynamo", "torch._inductor",
]

block_cipher = None

a = Analysis(
    ["run_app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=["rthook_torch_home.py"],
    excludes=["tensorboard", "matplotlib", "expecttest"] + cuda_excludes + torch_excludes,
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
