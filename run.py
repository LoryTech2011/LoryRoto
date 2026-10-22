import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
PATHS = {
    "TMPDIR": RUNTIME / "tmp",
    "TEMP": RUNTIME / "tmp",
    "TMP": RUNTIME / "tmp",
    "PIP_CACHE_DIR": RUNTIME / "cache" / "pip",
    "TORCH_HOME": RUNTIME / "cache" / "torch",
    "HF_HOME": RUNTIME / "cache" / "huggingface",
    "HUGGINGFACE_HUB_CACHE": RUNTIME / "cache" / "huggingface" / "hub",
    "XDG_CACHE_HOME": RUNTIME / "cache" / "xdg",
    "MPLCONFIGDIR": RUNTIME / "cache" / "matplotlib",
    "PYTHONPYCACHEPREFIX": RUNTIME / "cache" / "pycache",
}

for variable, path in PATHS.items():
    if not path.resolve().is_relative_to(ROOT):
        raise RuntimeError(f"Refusing to use a path outside the application folder: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if not path.resolve().is_relative_to(ROOT):
        raise RuntimeError(f"Refusing to use a path outside the application folder: {path}")
    os.environ[variable] = str(path)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
sys.path.insert(0, str(ROOT / "src"))

from rotoscope.app import main

if __name__ == "__main__":
    raise SystemExit(main(ROOT))
