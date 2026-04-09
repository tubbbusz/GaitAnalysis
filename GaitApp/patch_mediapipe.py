"""
patch_mediapipe.py
Patches mediapipe/tasks/python/core/mediapipe_c_bindings.py inside the
active venv so that the 'free' symbol is found when running as a
PyInstaller bundle on Windows.

Run AFTER activating the venv and installing mediapipe, BEFORE pyinstaller.
"""

import sys
import os
import importlib.util
import re

# ── locate the file ────────────────────────────────────────────────────────
try:
    import mediapipe
    mp_root = os.path.dirname(mediapipe.__file__)
except ImportError:
    print("ERROR: mediapipe is not installed in the current environment.")
    sys.exit(1)

target = os.path.join(mp_root, "tasks", "python", "core", "mediapipe_c_bindings.py")

if not os.path.exists(target):
    print(f"ERROR: Cannot find {target}")
    sys.exit(1)

with open(target, "r", encoding="utf-8") as f:
    src = f.read()

# ── check if already patched ───────────────────────────────────────────────
PATCH_MARKER = "# [GAITAPP-PATCH]"
if PATCH_MARKER in src:
    print(f"Already patched: {target}")
    sys.exit(0)

# ── the patch ─────────────────────────────────────────────────────────────
# We insert a CRT pre-load right before the line that calls
#   _shared_lib.free.argtypes = ...
# so the symbol is available regardless of PyInstaller's load order.

OLD = "_shared_lib.free.argtypes = [ctypes.c_void_p]"

NEW = (
    f"{PATCH_MARKER} pre-load CRT so 'free' is resolvable in frozen builds\n"
    "  if getattr(_shared_lib, 'free', None) is None:\n"
    "    import ctypes.util as _cu\n"
    "    for _crt in ('ucrtbase', 'msvcrt', 'c'):\n"
    "      _p = _cu.find_library(_crt)\n"
    "      if _p:\n"
    "        try:\n"
    "          import ctypes as _ct\n"
    "          _crt_lib = _ct.CDLL(_p, use_last_error=True)\n"
    "          if hasattr(_crt_lib, 'free'):\n"
    "            _shared_lib.free = _crt_lib.free\n"
    "            break\n"
    "        except OSError:\n"
    "          pass\n"
    f"  {OLD}"
)

if OLD not in src:
    print(
        f"WARNING: Expected line not found in {target}.\n"
        "The mediapipe version may have changed. Skipping patch.\n"
        "You may need to apply the fix manually."
    )
    # Don't fail the build — maybe the version doesn't need the patch
    sys.exit(0)

patched = src.replace(OLD, NEW, 1)

# ── write back ────────────────────────────────────────────────────────────
with open(target, "w", encoding="utf-8") as f:
    f.write(patched)

print(f"Patched successfully: {target}")
sys.exit(0)
