"""PyInstaller runtime hook: put the CUDA runtime DLLs on the search path.

In the venv this job is done by venv/Lib/site-packages/_cuda_dll_path.pth, but
a frozen app does not process .pth files at all -- so without this hook the
packaged build fails with the same misleading error the dev environment gave:

    Could not find module '...\\llama_cpp\\lib\\llama.dll'
    (or one of its dependencies)

llama_cpp loads llama.dll with winmode=0 (ctypes.RTLD_GLOBAL is 0 on Windows),
which selects the LEGACY LoadLibraryExW search order. That order ignores
directories registered via os.add_dll_directory but DOES search PATH, so PATH
is the mechanism that actually works here. Both are set anyway: add_dll_directory
covers other libraries (onnxruntime, ctranslate2) that load normally.
"""
import os
import sys
from pathlib import Path

_base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))

_dirs = [str(p) for p in [
    _base / "nvidia" / "cublas" / "bin",
    _base / "nvidia" / "cuda_runtime" / "bin",
    _base / "nvidia" / "cuda_nvrtc" / "bin",
    _base / "llama_cpp" / "lib",
] if p.is_dir()]

for _d in _dirs:
    try:
        os.add_dll_directory(_d)
    except (OSError, AttributeError):
        pass

if _dirs:
    os.environ["PATH"] = os.pathsep.join(_dirs) + os.pathsep + os.environ.get("PATH", "")
