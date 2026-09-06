# PyInstaller spec for Jarvis.   Build with:
#     venv\Scripts\pyinstaller.exe packaging\jarvis.spec --noconfirm
#
# ONEDIR, deliberately not --onefile.
#   onefile extracts the ENTIRE bundle to a temp directory on every launch.
#   This bundle contains ggml-cuda.dll (703MB) and cublasLt64_12.dll (638MB),
#   so every start would copy ~1.5GB before showing a window -- tens of seconds,
#   every time. onedir starts immediately and lets you ship differential updates.
#
# MODELS ARE NOT BUNDLED.
#   The GGUF alone is 4.47GB. Bundling it would produce a ~6.5GB artifact that
#   cannot be updated incrementally and takes an hour to transfer. Instead the
#   app downloads models to a data directory on first run (scripts/download_model.py),
#   which also lets the user swap the model without a rebuild.
#
# Expected output: ~1.6GB in dist/Jarvis/  (dominated by CUDA kernels).
# Dropping to a CPU-only llama-cpp-python wheel would cut that to ~250MB, at the
# cost of ~34 tok/s falling to roughly 5-8.

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

datas = []
binaries = []
hiddenimports = []

# llama-cpp-python: the .dll files next to the package are the whole point.
binaries += collect_dynamic_libs("llama_cpp")
datas += collect_data_files("llama_cpp")

# CUDA runtime shipped as pip wheels (no CUDA Toolkit install required).
for pkg in ("nvidia.cublas", "nvidia.cuda_runtime", "nvidia.cuda_nvrtc"):
    try:
        binaries += collect_dynamic_libs(pkg)
    except Exception:
        pass

# CTranslate2 (faster-whisper) and onnxruntime ship their own native libs.
binaries += collect_dynamic_libs("ctranslate2")
binaries += collect_dynamic_libs("onnxruntime")

# openWakeWord keeps its pretrained .onnx models inside the package.
datas += collect_data_files("openwakeword", include_py_files=False)

# chromadb pulls a lot in dynamically; PyInstaller cannot see those statically.
hiddenimports += collect_submodules("chromadb")
hiddenimports += [
    "onnxruntime", "tokenizers", "pypika", "hnswlib",
    "faster_whisper", "ctranslate2", "openwakeword",
    "sounddevice", "soundfile", "networkx", "watchdog",
    "pypdf", "docx", "openpyxl", "jinja2",
]
datas += collect_data_files("chromadb")

a = Analysis(
    ["../main.py"],
    pathex=["../src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=["runtime_hook_cuda.py"],
    excludes=["torch", "transformers", "matplotlib", "pandas", "scipy.spatial.cKDTree"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="Jarvis",
    console=False,          # no console window for a tray/desktop app
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False,
    upx=False,              # UPX on 700MB CUDA DLLs is slow and gains little
    name="Jarvis",
)
