# -*- mode: python ; coding: utf-8 -*-
import os
import mediapipe
from PyInstaller.utils.hooks import collect_all

datas = [
    ('novita.png',                '.'),
    ('pose_landmarker_full.task', '.'),
]
binaries = []
hiddenimports = []

# ── mediapipe ──────────────────────────────────────────────────────────────
tmp_ret = collect_all('mediapipe')
datas         += tmp_ret[0]
binaries      += tmp_ret[1]
hiddenimports += tmp_ret[2]

# Force-walk mediapipe and grab every native binary collect_all might miss
_mp_root   = os.path.dirname(mediapipe.__file__)
_mp_parent = os.path.dirname(_mp_root)
for _dirpath, _dirs, _files in os.walk(_mp_root):
    for _fname in _files:
        if _fname.endswith(('.dll', '.pyd', '.so')):
            _full = os.path.join(_dirpath, _fname)
            _rel  = os.path.relpath(os.path.dirname(_full), _mp_parent)
            # Avoid duplicates
            if not any(_full == b[0] for b in binaries):
                binaries.append((_full, _rel))

# ── opencv ─────────────────────────────────────────────────────────────────
tmp_ret = collect_all('cv2')
datas         += tmp_ret[0]
binaries      += tmp_ret[1]
hiddenimports += tmp_ret[2]

# ── hidden imports ─────────────────────────────────────────────────────────
hiddenimports += [
    'mediapipe.tasks.python.core.mediapipe_c_bindings',
    'mediapipe.tasks.python.vision.pose_landmarker',
    'mediapipe.python._framework_bindings',
    'scipy.signal',
    'scipy.signal._peak_finding_utils',
    'scipy.interpolate',
    'scipy.interpolate._fitpack_impl',
    'pandas',
    'matplotlib',
    'matplotlib.backends.backend_agg',
    'tkinter',
    'tkinter.filedialog',
    'tkinter.simpledialog',
]

a = Analysis(
    ['Gaitapp.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Gaitapp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
