import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# compat first so `flash_attn` resolves to the shim even if the real package is installed
for p in (ROOT / "pipeline" / "compat", ROOT / "pipeline"):
    sys.path.insert(0, str(p))
