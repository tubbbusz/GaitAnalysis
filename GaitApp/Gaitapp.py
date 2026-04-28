# standard imports
import atexit
import gc
import hashlib
import json
import os
import pickle
import shutil
import signal
import sys
import tempfile
import threading
import time
import urllib.request
from collections import OrderedDict, namedtuple
from datetime import datetime
from io import BytesIO

# third party imports
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import pyglet
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks
import tkinter as tk
from tkinter import filedialog, messagebox

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

def resource_path(filename):
    base = sys._MEIPASS if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, filename)

pyglet.font.add_file(resource_path('Coiny-Cyrillic.ttf'))

# analysis settings
SLOWMO_FPS    = 240
FILTER_CUTOFF = 6
FILTER_ORDER  = 3

# frame storage settings
SAVE_HEIGHT  = 540
JPEG_QUALITY = 65
CACHE_FRAMES = 96
DEBUG_DIRECTION_DIAGNOSTICS = False
DEBUG_LOG_JITTER = False
DEBUG_SKELETON_WIDTH = True
DEBUG_SKELETON_BASE_THICKNESS = 8.0
CACHE_SCHEMA_VERSION = 1

# pose model path
MODEL_PATH = resource_path("pose_landmarker_full.task")
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "pose_landmarker/pose_landmarker_full/float16/1/"
              "pose_landmarker_full.task")

_mp_bindings = None
_mp_lock = threading.Lock()


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading PoseLandmarker model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


def get_mediapipe_bindings():
    global _mp_bindings
    if _mp_bindings is not None:
        return _mp_bindings
    with _mp_lock:
        if _mp_bindings is None:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

            _mp_bindings = (mp, mp_python, PoseLandmarker, PoseLandmarkerOptions, RunningMode)
    return _mp_bindings


def _cache_root_dir():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "Clients", ".gait_cache")


def _cache_dir(cache_key):
    return os.path.join(_cache_root_dir(), cache_key)


def _butterworth_lowpass(cutoff, fs, order=4):
    nyquist = fs / 2.0
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def _butterworth_lowpass_filter(data, cutoff=6, fs=240, order=4):
    if len(data) < order + 1:
        return data
    b, a = _butterworth_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)


def _fix_jitter_outliers(df, max_frame_displacement=0.15, columns=None):
    if columns is None:
        columns = [c for c in df.columns if 'landmark_' in c and c.endswith(('_x', '_y', '_z'))]
    for col in columns:
        if col not in df.columns:
            continue
        diff = df[col].diff().abs()
        bad_frames = diff[diff > max_frame_displacement].index.tolist()
        if len(bad_frames) > 0:
            for frame_idx in bad_frames:
                if 0 < frame_idx < len(df) - 1:
                    before = df.iloc[frame_idx - 1][col]
                    after = df.iloc[frame_idx + 1][col]
                    df.at[frame_idx, col] = (before + after) / 2.0
    return df


def _fix_limb_swaps(df):
    landmark_pairs = [
        ('landmark_23_x', 'landmark_24_x'),
        ('landmark_25_x', 'landmark_26_x'),
        ('landmark_27_x', 'landmark_28_x'),
    ]
    for frame_idx in range(len(df)):
        for left_col, right_col in landmark_pairs:
            if left_col not in df.columns or right_col not in df.columns:
                continue
            left_x = df.iloc[frame_idx][left_col]
            right_x = df.iloc[frame_idx][right_col]
            if not pd.isna(left_x) and not pd.isna(right_x) and (right_x - left_x) < -0.05:
                if 0 < frame_idx < len(df) - 1:
                    before_lx = df.iloc[frame_idx - 1][left_col]
                    after_lx = df.iloc[frame_idx + 1][left_col]
                    before_rx = df.iloc[frame_idx - 1][right_col]
                    after_rx = df.iloc[frame_idx + 1][right_col]
                    if not pd.isna(before_lx) and not pd.isna(after_lx):
                        df.at[frame_idx, left_col] = 0.25 * before_lx + 0.50 * left_x + 0.25 * after_lx
                    if not pd.isna(before_rx) and not pd.isna(after_rx):
                        df.at[frame_idx, right_col] = 0.25 * before_rx + 0.50 * right_x + 0.25 * after_rx
    return df


def _detect_jittery_frames(pixel_landmarks, threshold=0.15):
    jittery_frames = set()
    if not pixel_landmarks or len(pixel_landmarks) < 2:
        return jittery_frames
    prev_landmarks = None
    for frame_idx, entry in enumerate(pixel_landmarks):
        if entry is None:
            prev_landmarks = None
            continue
        if isinstance(entry, tuple):
            _, current_lm = entry
        else:
            current_lm = entry
        if current_lm is None or len(current_lm) == 0:
            prev_landmarks = None
            continue
        if prev_landmarks is not None and len(prev_landmarks) == len(current_lm):
            max_displacement = 0.0
            for prev_lm, curr_lm in zip(prev_landmarks, current_lm):
                if prev_lm.visibility > 0.5 and curr_lm.visibility > 0.5:
                    dx = curr_lm.x - prev_lm.x
                    dy = curr_lm.y - prev_lm.y
                    displacement = np.sqrt(dx*dx + dy*dy)
                    max_displacement = max(max_displacement, displacement)
            if max_displacement > threshold:
                jittery_frames.add(frame_idx)
        prev_landmarks = current_lm
    return jittery_frames


def _video_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'size_bytes': os.path.getsize(video_path), 'frame_count': 0, 'fps': 0.0, 'width': 0, 'height': 0}
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {'size_bytes': os.path.getsize(video_path), 'frame_count': frame_count, 'fps': fps, 'width': width, 'height': height}


def _file_sha256(video_path, chunk_size=4 * 1024 * 1024):
    h = hashlib.sha256()
    with open(video_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _build_cache_key(scan_info, target_output_size):
    payload = {
        'schema': CACHE_SCHEMA_VERSION,
        'video_sha256': scan_info.get('video_sha256'),
        'video_meta': scan_info.get('video_meta'),
        'save_height': SAVE_HEIGHT,
        'jpeg_quality': JPEG_QUALITY,
        'filter_cutoff': FILTER_CUTOFF,
        'filter_order': FILTER_ORDER,
        'target_output_size': list(target_output_size) if target_output_size else None,
    }
    raw = json.dumps(payload, sort_keys=True).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _load_cached_video_result(cache_key):
    path = os.path.join(_cache_dir(cache_key), 'result.pkl')
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        if payload.get('schema') != CACHE_SCHEMA_VERSION:
            return None
        result = payload.get('result')
        if not isinstance(result, dict):
            return None
        return result
    except Exception:
        return None


def _save_cached_video_result(cache_key, cache_meta, result):
    cdir = _cache_dir(cache_key)
    os.makedirs(cdir, exist_ok=True)
    manifest_path = os.path.join(cdir, 'manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(cache_meta, f, indent=2)
    payload_path = os.path.join(cdir, 'result.pkl')
    with open(payload_path, 'wb') as f:
        pickle.dump({'schema': CACHE_SCHEMA_VERSION, 'result': result}, f, protocol=pickle.HIGHEST_PROTOCOL)


def _markup_cache_path(cache_key):
    return os.path.join(_cache_dir(cache_key), 'markup.json')


def _load_cached_markup(cache_key):
    path = _markup_cache_path(cache_key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        if payload.get('schema') != CACHE_SCHEMA_VERSION:
            return None
        step_frames = []
        for item in payload.get('step_frames', []):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                step_frames.append((int(item[0]), str(item[1])))
        return {'step_frames': step_frames}
    except Exception:
        return None


def _save_cached_markup(cache_key, step_frames):
    cdir = _cache_dir(cache_key)
    os.makedirs(cdir, exist_ok=True)
    payload = {
        'schema': CACHE_SCHEMA_VERSION,
        'step_frames': [[int(f), str(side)] for f, side in step_frames],
    }
    with open(_markup_cache_path(cache_key), 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def _clear_cached_markup(cache_key):
    path = _markup_cache_path(cache_key)
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


def _ui_settings_path():
    return os.path.join(_cache_root_dir(), 'ui_settings.json')


def _load_ui_settings():
    path = _ui_settings_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _save_ui_settings(data):
    try:
        os.makedirs(_cache_root_dir(), exist_ok=True)
        with open(_ui_settings_path(), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

# auto-crop helpers

_CROP_VIS_THRESH = 0.40
_CROP_TOP_IDX    = [0, 7, 8, 11, 12]   # head, ears, shoulders
_CROP_FOOT_IDX   = [27, 28, 29, 30, 31, 32]  # ankles, heels, toes
_CROP_TARGET_AR  = 4.0 / 3.0


def _crop_skeleton_stats(pixel_landmarks_list):
    """Compute tight skeleton bbox stats over all frames.
    pixel_landmarks_list is a list of (thumb, [SimpleLandmark...]) | (thumb, None)
    or just [SimpleLandmark...] | None entries.
    """
    heights = []; all_xs = []; all_ys = []; tight_xs = []; tight_ys = []
    per_frame_bbox = []

    for entry in pixel_landmarks_list:
        lms = entry[1] if isinstance(entry, tuple) else entry
        if lms is None:
            per_frame_bbox.append(None)
            continue
        vis = [lm for lm in lms if lm.visibility > _CROP_VIS_THRESH]
        if not vis:
            per_frame_bbox.append(None)
            continue
        fxs = [lm.x for lm in vis]; fys = [lm.y for lm in vis]
        per_frame_bbox.append((min(fxs), max(fxs), min(fys), max(fys)))
        tight_xs.extend(fxs); tight_ys.extend(fys)
        all_xs.extend(fxs);   all_ys.extend(fys)
        top_ys  = [lms[i].y for i in _CROP_TOP_IDX  if i < len(lms) and lms[i].visibility > _CROP_VIS_THRESH]
        foot_ys = [lms[i].y for i in _CROP_FOOT_IDX if i < len(lms) and lms[i].visibility > _CROP_VIS_THRESH]
        if top_ys and foot_ys:
            h = float(np.median(foot_ys) - np.median(top_ys))
            if h > 0.03:
                heights.append(h)

    if not heights or not all_xs:
        return None
    return dict(
        median_h=float(np.median(heights)),
        p95_h=float(np.percentile(heights, 95)),
        cx=float((np.min(all_xs) + np.max(all_xs)) / 2),
        cy=float((np.min(all_ys) + np.max(all_ys)) / 2),
        x_min=float(np.min(all_xs)), x_max=float(np.max(all_xs)),
        y_min=float(np.min(all_ys)), y_max=float(np.max(all_ys)),
        tight_x_min=float(np.min(tight_xs)), tight_x_max=float(np.max(tight_xs)),
        tight_y_min=float(np.min(tight_ys)), tight_y_max=float(np.max(tight_ys)),
        per_frame_bbox=per_frame_bbox,
    )

def _debug_crop_stats(landmarks, stats, video_path):
    """Print per-frame y_min to find what's pulling the crop up."""
    print(f"\n{'='*60}")
    print(f"CROP DEBUG: {os.path.basename(video_path)}")
    print(f"  tight_y_min={stats['tight_y_min']:.4f}  tight_y_max={stats['tight_y_max']:.4f}")
    print(f"  tight_x_min={stats['tight_x_min']:.4f}  tight_x_max={stats['tight_x_max']:.4f}")
    print(f"  Total frames: {len(landmarks)}")
    
    suspicious = []
    for fi, entry in enumerate(landmarks):
        if entry is None:
            continue
        lms = entry[1] if isinstance(entry, tuple) else entry
        if lms is None:
            continue
        vis = [lm for lm in lms if lm.visibility > _CROP_VIS_THRESH]
        if not vis:
            continue
        fys = [lm.y for lm in vis]
        frame_y_min = min(fys)
        if frame_y_min < stats['tight_y_min'] + 0.05:  # frames near top
            suspicious.append((fi, frame_y_min, 
                               [(i, lm.y, lm.visibility) for i, lm in enumerate(lms) 
                                if lm.visibility > _CROP_VIS_THRESH and lm.y < 0.15]))
    
    print(f"  Frames with landmarks near top (y < {stats['tight_y_min']+0.05:.3f}):")
    for fi, y_min, top_lms in suspicious[:20]:  # cap at 20
        print(f"    frame {fi:4d}: y_min={y_min:.4f}  top landmarks: {top_lms}")
    print(f"{'='*60}\n")

def _compute_auto_crop(stats, fw, fh, pad=0.05):
    if stats is None:
        return None
    x0 = max(0.0, stats['tight_x_min'] - pad)
    y0 = max(0.0, stats['tight_y_min'] - pad)
    x1 = min(1.0, stats['tight_x_max'] + pad)
    y1 = min(1.0, stats['tight_y_max'] + pad)
    w  = max(0.01, x1 - x0)
    h  = max(0.01, y1 - y0)
    print(f"  _compute_auto_crop → x0={x0:.4f} y0={y0:.4f} w={w:.4f} h={h:.4f}  (fw={fw} fh={fh})")
    return x0, y0, w, h

_CROP_TARGET_FILL = 0.70  # skeleton occupies this fraction of crop height

def _apply_target_fill(stats, x0, y0, w, h):
    skel_h = stats.get('p95_h', 0)
    if skel_h <= 0:
        return x0, y0, w, h
    target_h = max(0.01, min(1.0, skel_h / _CROP_TARGET_FILL))
    cy = stats['cy']
    new_y0 = cy - target_h / 2.0
    new_y1 = cy + target_h / 2.0
    if new_y0 < 0.0:
        new_y1 -= new_y0; new_y0 = 0.0
    if new_y1 > 1.0:
        new_y0 -= (new_y1 - 1.0); new_y1 = 1.0; new_y0 = max(0.0, new_y0)
    return x0, new_y0, w, max(0.01, new_y1 - new_y0)


def _compute_auto_crops_pair(stats_a, stats_b, fw_a, fh_a, fw_b, fh_b, pad=0.05):
    def _tight(stats):
        x0 = max(0.0, stats['tight_x_min'] - pad)
        y0 = max(0.0, stats['tight_y_min'] - pad)
        x1 = min(1.0, stats['tight_x_max'] + pad)
        y1 = min(1.0, stats['tight_y_max'] + pad)
        return x0, y0, max(0.01, x1 - x0), max(0.01, y1 - y0)

    if stats_a is None and stats_b is None:
        return None, None
    if stats_a is None:
        return None, _tight(stats_b)
    if stats_b is None:
        return _tight(stats_a), None

    return _tight(stats_a), _tight(stats_b)

def _apply_crop_rect(frame_bgr, crop):
    """Crop a BGR frame given (x0, y0, w, h) in normalised [0,1] coords."""
    if crop is None:
        return frame_bgr
    fh, fw = frame_bgr.shape[:2]
    x0, y0, cw, ch = crop
    px = int(x0 * fw);  py = int(y0 * fh)
    pw = max(1, int(cw * fw)); ph = max(1, int(ch * fh))
    px = min(px, max(0, fw - pw)); py = min(py, max(0, fh - ph))
    return frame_bgr[py:py + ph, px:px + pw]

class PoseLandmark:
    NOSE = 0
    LEFT_EYE_INNER = 1;  LEFT_EYE = 2;   LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4; RIGHT_EYE = 5;  RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7;        RIGHT_EAR = 8
    MOUTH_LEFT = 9;      MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11;  RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13;     RIGHT_ELBOW = 14
    LEFT_WRIST = 15;     RIGHT_WRIST = 16
    LEFT_PINKY = 17;     RIGHT_PINKY = 18
    LEFT_INDEX = 19;     RIGHT_INDEX = 20
    LEFT_THUMB = 21;     RIGHT_THUMB = 22
    LEFT_HIP = 23;       RIGHT_HIP = 24
    LEFT_KNEE = 25;      RIGHT_KNEE = 26
    LEFT_ANKLE = 27;     RIGHT_ANKLE = 28
    LEFT_HEEL = 29;      RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31; RIGHT_FOOT_INDEX = 32

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),
    (23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]

DRAW_THICKNESS      = 8
DEFAULT_SKELETON_THICKNESS = 4.5
USE_WORLD_LANDMARKS = False
SKELETON_LINE_COL   = (0, 0, 255)

JOINT_NAME_TO_LANDMARK = {
    'left_hip':    PoseLandmark.LEFT_HIP,
    'right_hip':   PoseLandmark.RIGHT_HIP,
    'left_knee':   PoseLandmark.LEFT_KNEE,
    'right_knee':  PoseLandmark.RIGHT_KNEE,
    'left_ankle':  PoseLandmark.LEFT_ANKLE,
    'right_ankle': PoseLandmark.RIGHT_ANKLE,
}

LANDMARK_TO_JOINT_NAME = {v: k for k, v in JOINT_NAME_TO_LANDMARK.items()}

JOINT_COLORS_MPL = {
    'left_hip':    '#c0392b', 'left_knee':   '#d35400', 'left_ankle':  '#8e44ad',
    'right_hip':   '#2471a3', 'right_knee':  '#1a5276', 'right_ankle': '#148f77',
}

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)

JOINT_COLORS_BGR = {k: hex_to_bgr(v) for k, v in JOINT_COLORS_MPL.items()}
C_RIGHT = '#4a90d9'
C_LEFT  = '#e8913a'

BG      = "#f0f0f0"
BG2     = "#d6d6d6"
BG3     = "#c8c8c8"
BG_VID  = "#d8d8d8"
BG_PLOT = "#dbdbdb"
BG_INIT = "#c0c0c0"

ACCENT  = "#3a083a"
TEXT    = "#1a1a1a"
SUBTEXT = "#4a4a4a"
GREEN   = '#27ae60'
RED     = '#c0392b'

C_V1      = "#4a1a44"
C_V2      = "#2e6b40"
C_CURSOR  = '#ff4444'
C_OUTLIER = '#555555'
C_NORM    = '#888888'

NORMATIVE_GAIT = {
    "hip": {"mean": np.array([
        30,29.506506685311212,29.01229856655923,28.516660839680867,28.018878700612923,
        27.51823734529221,27.014021969655534,26.505517769639706,25.99200994118153,25.47278368021781,
        24.947123895956047,24.413935007804735,23.870990861489325,23.31585627706569,22.74609607458976,
        22.15927507411743,21.552958095704597,20.924709959407174,20.27209548528106,19.592679493382157,
        18.884026803766375,18.143702236489613,17.369270611607774,16.558296749176765,15.708345469252489,
        14.816981591890844,13.88176993714774,12.900275325079075,11.870062575740757,10.788696509188691,
        9.653799953645802,8.467483850702372,7.239466805631681,5.980204342273305,4.700151984466822,
        3.4097652560518084,2.119499680867847,0.8398107827545127,-0.41885,-1.64601,-2.83124,
        -3.96407,-5.03404,-6.0307,-6.9436,-7.76228,-8.47628,-9.07515,-9.54843,-9.88566,
        -10.0767,-10.118,-10.0125,-9.76377,-9.37503,-8.84966,-8.19103,-7.40248,-6.48737,
        -5.44907,-4.29124,-3.02239,-1.65474,-0.2006,1.3277196644497558,2.9179054948001246,
        4.557645419714394,6.234627242087145,7.936538764812967,9.651067790786447,11.36590212290217,
        13.068729564054731,14.747237917138705,16.389114985048685,17.98204857067926,19.51372647692502,
        20.971836506680543,22.344066462840424,23.618104148299242,24.781637365951592,25.823944482924336,
        26.74605913012349,27.554283682474498,28.25494536746892,28.85437141259832,29.358889045354253,
        29.77483,30.10850798371201,30.366263744296955,30.55442,30.679303985736794,30.747242921574824,
        30.76456403748034,30.737594560944906,30.672661719460088,30.576092740517456,30.454214851608565,
        30.313355280224986,30.159841253858275,30
    ]), "lower": None, "upper": None},
    "knee": {"mean": np.array([
        175,174.3658,173.7341,173.1075,172.4885,171.8797,171.2834,170.7021,170.138,169.5936,
        169.0709,168.5721,168.0992,167.6541,167.2385,166.8542,166.5027,166.1854,165.9036,165.6584,
        165.4508,165.2815,165.1513,165.0606,165.0098,164.9988,165.0276,165.0958,165.2028,165.3477,
        165.5292,165.7455,165.9946,166.2735,166.5789,166.9063,167.2504,167.6046,167.9609,168.3099,
        168.6402,168.9387,169.1899,169.3768,169.4799,169.4781,169.349,169.069,168.6146,167.963,
        166.7759,165.0369,163.0528,160.8189,158.3382,155.6223,152.6924,149.5803,146.328,142.9879,
        139.6216,136.2981,133.0921,130.0813,127.3429,124.9508,122.9722,121.4644,120.4721,120.0252,
        120.1373,120.8049,122.0079,123.7101,125.8616,128.4007,131.2572,134.3555,137.6176,140.967,
        144.3308,147.6426,150.8446,153.8883,156.7357,159.3596,161.7424,163.8763,165.7615,167.405,
        168.8196,170.0218,171.0308,171.8674,172.5528,173.1076,173.5515,173.9026,174.1769,174.389
    ]), "lower": None, "upper": None},
    "ankle": {"mean": np.array([
        0,-0.747,-1.48,-2.182,-2.839,-3.437,-3.959,-4.392,-4.72,-4.928,-5.001,-4.93,
        -4.723,-4.393,-3.949,-3.404,-2.769,-2.055,-1.275,-0.438,0.442,1.355,2.29,3.235,
        4.178,5.108,6.013,6.883,7.705,8.469,9.163,9.775,10.294,10.708,11.006,11.178,
        11.21,11.092,10.812,10.36,9.723,8.903,7.92,6.793,5.541,4.185,2.743,1.237,
        -0.315,-1.893,-3.477,-5.047,-6.584,-8.068,-9.479,-10.797,-12.004,-13.078,-14.001,
        -14.752,-15.313,-15.684,-15.878,-15.907,-15.787,-15.53,-15.15,-14.664,-14.085,
        -13.429,-12.71,-11.943,-11.144,-10.326,-9.505,-8.694,-7.91,-7.166,-6.477,-5.859,
        -5.325,-4.891,-4.57,-4.378,-4.328,-4.435,-4.714,-5.177,-5.84,-6.717,-7.822,
        -9.17,-10.774,-12.649,-14.808,-17.267,-20.039,-23.139,-26.582,-30.381
    ]), "lower": None, "upper": None},
}
NORMATIVE_GAIT['hip']['mean']   = NORMATIVE_GAIT['hip']['mean'][:100]
NORMATIVE_GAIT['ankle']['mean_base'] = NORMATIVE_GAIT['ankle']['mean'][:100].copy()
NORMATIVE_GAIT['ankle']['offset'] = 120.0

def _apply_ankle_normative_offset(offset_deg):
    NORMATIVE_GAIT['ankle']['offset'] = offset_deg
    m = np.array(NORMATIVE_GAIT['ankle']['mean_base'] + offset_deg)
    m = np.nan_to_num(m, nan=np.nanmean(m))
    sd = np.std(m)
    se = sd / np.sqrt(len(m))
    NORMATIVE_GAIT['ankle']['mean'] = m
    NORMATIVE_GAIT['ankle']['lower'] = np.asarray(m - se)
    NORMATIVE_GAIT['ankle']['upper'] = np.asarray(m + se)

_apply_ankle_normative_offset(NORMATIVE_GAIT['ankle']['offset'])
for jt in NORMATIVE_GAIT:
    m  = np.array(NORMATIVE_GAIT[jt]["mean"])
    m = np.nan_to_num(m, nan=np.nanmean(m))
    sd = np.std(m)
    se = sd / np.sqrt(len(m))
    NORMATIVE_GAIT[jt]["mean"] = m
    NORMATIVE_GAIT[jt]["lower"] = np.asarray(m - se)
    NORMATIVE_GAIT[jt]["upper"] = np.asarray(m + se)
NORMATIVE_X = np.linspace(0, 100, 100)


SimpleLandmark = namedtuple('SimpleLandmark', ['x', 'y', 'visibility'])

GRAY_BGR = (128, 128, 128)

def _compute_cycle_rmse(y_values, reference_y, max_cycle_length):
    if len(y_values) < 2 or reference_y is None or len(reference_y) == 0:
        return 0.0
    try:
        t = np.linspace(0, 1, len(y_values))
        y_resampled = interp1d(t, y_values)(np.linspace(0, 1, max_cycle_length))
        if len(reference_y) != max_cycle_length:
            t_ref = np.linspace(0, 1, len(reference_y))
            reference_resampled = interp1d(t_ref, reference_y)(np.linspace(0, 1, max_cycle_length))
        else:
            reference_resampled = reference_y
        rmse = float(np.sqrt(np.mean((y_resampled - reference_resampled) ** 2)))
        cycle_range = np.nanmax(y_resampled) - np.nanmin(y_resampled)
        ref_range = np.nanmax(reference_resampled) - np.nanmin(reference_resampled)
        max_range = max(cycle_range, ref_range)
        if max_range <= 0:
            return 0.0
        return (rmse / max_range) * 100.0
    except Exception:
        return 0.0


def draw_pose_landmarks_on_frame(frame_bgr, pixel_landmarks, joint_visibility=None, focus_side=None,
                                 skeleton_thickness=None, draw_jitter_red=False):
    h, w = frame_bgr.shape[:2]
    if skeleton_thickness is None:
        skeleton_thickness = DRAW_THICKNESS
    if skeleton_thickness <= 0:
        return frame_bgr

    visible_pts = []
    for lm in pixel_landmarks:
        if lm.visibility > 0.5:
            visible_pts.append((lm.x * w, lm.y * h))
    if visible_pts:
        xs = [p[0] for p in visible_pts]
        ys = [p[1] for p in visible_pts]
        subject_width_px = max(xs) - min(xs)
        subject_height_px = max(ys) - min(ys)
        subject_span_px = max(subject_width_px, subject_height_px)
        subject_scale = max(0.45, min(2.75, subject_span_px / 260.0))
    else:
        subject_scale = 1.0
    line_thickness = max(1, int(round(skeleton_thickness * subject_scale)))
    circle_radius  = max(1, int(round(skeleton_thickness * 0.5 * subject_scale)))
    default_line_col = GRAY_BGR if focus_side in ('left', 'right') else (200, 200, 200)

    def _side_matches(jname):
        if focus_side not in ('left', 'right'):
            return True
        return jname.startswith(focus_side + '_')

    def _resolve_color(landmark_idx, base_color):
        if draw_jitter_red:
            return (0, 0, 255)
        if landmark_idx not in LANDMARK_TO_JOINT_NAME:
            return GRAY_BGR if focus_side in ('left', 'right') else base_color
        jname = LANDMARK_TO_JOINT_NAME[landmark_idx]
        if not _side_matches(jname):
            return GRAY_BGR
        if joint_visibility is not None and not joint_visibility.get(jname, True):
            return GRAY_BGR
        return JOINT_COLORS_BGR[jname]

    for s, e in POSE_CONNECTIONS:
        if s < len(pixel_landmarks) and e < len(pixel_landmarks):
            ls, le = pixel_landmarks[s], pixel_landmarks[e]
            if ls.visibility > 0.5 and le.visibility > 0.5:
                if e in LANDMARK_TO_JOINT_NAME:
                    line_color = _resolve_color(e, default_line_col)
                elif s in LANDMARK_TO_JOINT_NAME:
                    line_color = _resolve_color(s, default_line_col)
                else:
                    line_color = default_line_col
                cv2.line(frame_bgr,
                         (int(ls.x*w), int(ls.y*h)),
                         (int(le.x*w), int(le.y*h)),
                         line_color, line_thickness)

    for idx, lm in enumerate(pixel_landmarks):
        if lm.visibility > 0.5:
            if idx in LANDMARK_TO_JOINT_NAME:
                joint_color = _resolve_color(idx, (255, 0, 255))
            else:
                joint_color = GRAY_BGR if focus_side in ('left', 'right') else (255, 0, 255)
            cv2.circle(frame_bgr, (int(lm.x*w), int(lm.y*h)), circle_radius, joint_color, -1)

    return frame_bgr


def midpoint_shoulder(landmarks):
    ls = landmarks[PoseLandmark.LEFT_SHOULDER]
    rs = landmarks[PoseLandmark.RIGHT_SHOULDER]
    return (
        (ls.x + rs.x) / 2,
        (ls.y + rs.y) / 2,
        (getattr(ls, 'z', 0) + getattr(rs, 'z', 0)) / 2,
    )


def determine_walking_direction(landmarks):
    try:
        rhy = landmarks[PoseLandmark.RIGHT_HEEL].y
        rty = landmarks[PoseLandmark.RIGHT_FOOT_INDEX].y
        return "left" if rhy < rty else "right"
    except Exception:
        return "right"


def calculate_angle(a, b, c, d=None, direction="left", joint_type="hip"):
    a = np.array(a[:2])
    b = np.array(b[:2])
    c = np.array(c[:2])

    ba = a - b
    bc = c - b
    base = np.degrees(np.arctan2(bc[1], bc[0]) - np.arctan2(ba[1], ba[0])) % 360

    if joint_type == "hip":
        angle = base - 180
    elif joint_type == "knee":
        angle = base
    elif joint_type == "ankle":
        angle = (360 - base) % 360

    if direction == "right" and joint_type == "hip":
        angle = ((360 - base) % 360) - 180
    if direction == "right" and joint_type == "knee":
        angle = (360 - base) % 360
    if direction == "right" and joint_type == "ankle":
        angle = base

    return angle


def detect_crop_region(video_path, needs_rotation, sample_count=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_count = min(sample_count, max(1, total))
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = 0, 0
    full_h, full_w = 0, 0
    for i in range(sample_count):
        target = int(total * (i + 1) / (sample_count + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if not ret:
            continue
        if needs_rotation:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        full_h, full_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        if coords is None:
            continue
        bx, by, bw, bh = cv2.boundingRect(coords)
        x_min = min(x_min, bx)
        y_min = min(y_min, by)
        x_max = max(x_max, bx + bw)
        y_max = max(y_max, by + bh)
    cap.release()
    if full_h == 0 or x_min == float('inf'):
        return None
    cw = x_max - x_min
    ch = y_max - y_min
    margin = 0.02
    left_frac   = x_min / full_w
    top_frac    = y_min / full_h
    right_frac  = (full_w - x_max) / full_w
    bottom_frac = (full_h - y_max) / full_h
    if max(left_frac, top_frac, right_frac, bottom_frac) < margin:
        return None
    return (x_min, y_min, cw, ch)


def pct_change(new, old):
    return 0.0 if old == 0 else (new - old) / old * 100.0


def detect_steps(angle_df, fps=240.0, min_step_time=0.4, refine_radius=5):
    for col in ("left_hip", "right_hip"):
        if col not in angle_df:
            raise ValueError(f"Column '{col}' not found")
    frame_nums = angle_df['frame_num'].to_numpy(dtype=int)

    def _refine(vals, p, r=refine_radius):
        L, R = max(0, p-r), min(len(vals)-1, p+r)
        return L + int(np.argmax(vals[L:R+1]))

    def _peaks_for(col):
        vals = angle_df[col].ffill().bfill().to_numpy(dtype=float)
        min_dist = int(min_step_time * fps)
        cand, props = find_peaks(vals, distance=min_dist, prominence=3.0)
        if cand.size == 0:
            return []
        if len(cand) >= 3:
            med_interval = np.median(np.diff(cand))
            filtered = [cand[0]]
            for i in range(1, len(cand)):
                if cand[i] - filtered[-1] >= med_interval * 0.5:
                    filtered.append(cand[i])
            cand = np.array(filtered)
        return [int(frame_nums[_refine(vals, p)]) for p in cand]

    steps = ([(f, "left")  for f in _peaks_for("left_hip")] +
             [(f, "right") for f in _peaks_for("right_hip")])
    steps.sort(key=lambda x: x[0])
    return steps


def _mad(vals):
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return 0.0
    med = np.median(a)
    return float(np.median(np.abs(a - med)))


def _bool_runs(mask):
    runs = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _peak_prominence_scores(vals, peaks):
    scores = {}
    if len(peaks) == 0:
        return scores
    for p in peaks:
        l0 = max(0, p - 6)
        r0 = min(len(vals), p + 7)
        baseline = np.median(vals[l0:r0]) if r0 > l0 else vals[p]
        scores[int(p)] = float(max(0.0, vals[p] - baseline))
    return scores


def detect_steps_robust(angle_df, depth_df=None, fps=240.0):
    if angle_df is None or angle_df.empty:
        return [], [], []
    req = ('frame_num', 'left_hip', 'right_hip')
    if any(c not in angle_df.columns for c in req):
        return [], [], []

    df = angle_df.copy()
    frame_nums = df['frame_num'].to_numpy(dtype=int)
    left_vals = df['left_hip'].ffill().bfill().to_numpy(dtype=float)
    right_vals = df['right_hip'].ffill().bfill().to_numpy(dtype=float)
    hip_mean = 0.5 * (left_vals + right_vals)
    hip_diff = left_vals - right_vals

    vel = np.abs(np.gradient(hip_mean))
    jerk = np.abs(np.gradient(vel))

    win = max(9, int(0.40 * fps))
    if win % 2 == 0:
        win += 1
    amp = pd.Series(np.abs(hip_diff)).rolling(win, center=True, min_periods=max(5, win // 4)).std().to_numpy()
    amp = np.nan_to_num(amp, nan=np.nanmedian(amp) if np.isfinite(np.nanmedian(amp)) else 0.0)

    amp_med = float(np.median(amp))
    amp_thr = max(float(np.percentile(amp, 20)), amp_med * 0.35)
    low_amp = amp < amp_thr

    vel_thr = float(np.median(vel) + 4.5 * max(_mad(vel), 1e-6))
    jerk_thr = float(np.median(jerk) + 6.0 * max(_mad(jerk), 1e-6))
    dyn_bad = (vel > vel_thr) | (jerk > jerk_thr)

    turn_bad = np.zeros(len(df), dtype=bool)
    if isinstance(depth_df, pd.DataFrame) and not depth_df.empty:
        z_cols = ['joint_11', 'joint_12', 'joint_23', 'joint_24']
        if all(c in depth_df.columns for c in z_cols) and 'frame_num' in depth_df.columns:
            dep = depth_df[['frame_num'] + z_cols].dropna().copy()
            if not dep.empty:
                dep = dep.sort_values('frame_num')
                yaw = 0.5 * ((dep['joint_11'] - dep['joint_12']) + (dep['joint_23'] - dep['joint_24']))
                yaw = yaw.to_numpy(dtype=float)
                yaw_frame = dep['frame_num'].to_numpy(dtype=int)
                yaw_interp = np.interp(frame_nums, yaw_frame, yaw)
                yaw_rate = np.abs(np.gradient(yaw_interp))
                yr_thr = float(np.median(yaw_rate) + 5.0 * max(_mad(yaw_rate), 1e-6))
                ymag = np.abs(yaw_interp)
                ymag_thr = float(np.percentile(ymag, 25))
                turn_bad = (yaw_rate > yr_thr) | (ymag < ymag_thr * 0.55)

    bad = low_amp | dyn_bad | turn_bad

    smooth_win = max(5, int(0.30 * fps))
    if smooth_win % 2 == 0:
        smooth_win += 1
    smooth_bad = pd.Series(bad.astype(float)).rolling(smooth_win, center=True, min_periods=1).mean().to_numpy() > 0.45

    min_excl_frames = max(8, int(0.25 * fps))
    pad_frames = int(0.12 * fps)
    exclusion_regions = []
    for s, e in _bool_runs(smooth_bad):
        if (e - s + 1) < min_excl_frames:
            continue
        s2 = max(0, s - pad_frames)
        e2 = min(len(frame_nums) - 1, e + pad_frames)
        exclusion_regions.append((int(frame_nums[s2]), int(frame_nums[e2])))

    if len(exclusion_regions) > 1:
        exclusion_regions.sort()
        merged = [exclusion_regions[0]]
        for s, e in exclusion_regions[1:]:
            ps, pe = merged[-1]
            if s <= pe + 1:
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))
        exclusion_regions = merged

    good_mask = np.ones(len(frame_nums), dtype=bool)
    for s, e in exclusion_regions:
        good_mask &= ~((frame_nums >= s) & (frame_nums <= e))

    seg_min = max(20, int(0.55 * fps))
    good_segments = [(s, e) for s, e in _bool_runs(good_mask) if (e - s + 1) >= seg_min]

    candidates = []
    recovery_frames = int(0.50 * fps)
    min_dist = max(10, int(0.30 * fps))

    for s, e in good_segments:
        seg_l = left_vals[s:e+1]
        seg_r = right_vals[s:e+1]
        p_l = max(1.0, 0.20 * float(np.std(seg_l)))
        p_r = max(1.0, 0.20 * float(np.std(seg_r)))

        peaks_l, _ = find_peaks(seg_l, distance=min_dist, prominence=p_l)
        peaks_r, _ = find_peaks(seg_r, distance=min_dist, prominence=p_r)

        rs = s
        re = min(e, s + recovery_frames)
        if re > rs + 4:
            relax_l, _ = find_peaks(left_vals[rs:re+1], distance=max(8, int(min_dist * 0.8)), prominence=max(0.8, p_l * 0.7))
            relax_r, _ = find_peaks(right_vals[rs:re+1], distance=max(8, int(min_dist * 0.8)), prominence=max(0.8, p_r * 0.7))
            if len(relax_l):
                peaks_l = np.unique(np.concatenate([peaks_l, relax_l + (rs - s)]))
            if len(relax_r):
                peaks_r = np.unique(np.concatenate([peaks_r, relax_r + (rs - s)]))

        pl_scores = _peak_prominence_scores(seg_l, peaks_l)
        pr_scores = _peak_prominence_scores(seg_r, peaks_r)

        for p in peaks_l:
            gidx = int(s + p)
            candidates.append((gidx, 'left', pl_scores.get(int(p), 0.0)))
        for p in peaks_r:
            gidx = int(s + p)
            candidates.append((gidx, 'right', pr_scores.get(int(p), 0.0)))

    if not candidates:
        return [], exclusion_regions, []

    candidates.sort(key=lambda x: x[0])

    deduped = [candidates[0]]
    close_frames = int(0.20 * fps)
    for c in candidates[1:]:
        prev = deduped[-1]
        if c[0] - prev[0] <= close_frames:
            if c[2] > prev[2]:
                deduped[-1] = c
        else:
            deduped.append(c)

    accepted = []
    min_step_frames = int(0.28 * fps)
    max_step_frames = int(1.20 * fps)
    for c in deduped:
        if not accepted:
            accepted.append(c)
            continue
        prev = accepted[-1]
        dt = c[0] - prev[0]
        if dt < min_step_frames:
            if c[2] > prev[2]:
                accepted[-1] = c
            continue
        if dt > max_step_frames:
            accepted.append(c)
            continue
        if c[1] == prev[1]:
            if c[2] > prev[2]:
                accepted[-1] = c
        else:
            accepted.append(c)

    if len(accepted) < 2:
        return [], exclusion_regions, []

    intervals = np.diff([a[0] for a in accepted])
    med_int = float(np.median(intervals)) if len(intervals) else float(max(min_step_frames, 1))
    if med_int <= 0:
        med_int = float(max(min_step_frames, 1))

    step_frames = []
    step_meta = []
    for i, (idx, side, prom) in enumerate(accepted):
        fnum = int(frame_nums[idx])
        q_prom = float(min(1.0, prom / (2.5 + 1e-6)))
        if i == 0:
            q_cad = 0.55
        else:
            dt = accepted[i][0] - accepted[i-1][0]
            q_cad = float(max(0.0, 1.0 - abs(dt - med_int) / (0.7 * med_int + 1e-6)))
        q_amp = float(min(1.0, max(0.0, amp[idx] / (amp_thr * 2.2 + 1e-6))))
        q_total = float(0.45 * q_prom + 0.35 * q_cad + 0.20 * q_amp)
        if q_total < 0.24:
            continue
        step_frames.append((fnum, side))
        step_meta.append({'frame': fnum, 'side': side, 'quality': q_total})

    return step_frames, exclusion_regions, step_meta


def _get_cropped_dimensions(video_path, needs_rotation, crop_rect=None):
    if crop_rect is None:
        crop_rect = detect_crop_region(video_path, needs_rotation)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    if needs_rotation:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if crop_rect:
        cx, cy, cw_r, ch_r = crop_rect
        return (cw_r, ch_r)
    else:
        h, w = frame.shape[:2]
        return (w, h)


def select_video_paths():
    root = tk.Tk()
    root.withdraw()
    paths = filedialog.askopenfilenames(
        title="Select two videos for comparison",
        filetypes=[("Video files", "*.mov *.mp4 *.avi *.m4v"), ("All files", "*.*")])
    root.destroy()
    return list(paths)


def _apply_dynamic_window_size(window, preferred_width=None, preferred_height=None):
    window.update_idletasks()
    req_w = max(1, int(window.winfo_reqwidth()))
    req_h = max(1, int(window.winfo_reqheight()))
    width = req_w if preferred_width is None else max(req_w, int(preferred_width))
    height = req_h if preferred_height is None else max(req_h, int(preferred_height))
    screen_w = window.winfo_screenwidth()
    screen_h = window.winfo_screenheight()
    width = min(width, screen_w)
    height = min(height, screen_h)
    window.geometry(f"{width}x{height}")
    window.minsize(req_w, req_h)


JOINT_DEFS = {
    'left_hip':    (None,                        PoseLandmark.LEFT_HIP,    PoseLandmark.LEFT_KNEE),
    'right_hip':   (None,                        PoseLandmark.RIGHT_HIP,   PoseLandmark.RIGHT_KNEE),
    'left_knee':   (PoseLandmark.LEFT_HIP,       PoseLandmark.LEFT_KNEE,   PoseLandmark.LEFT_ANKLE),
    'right_knee':  (PoseLandmark.RIGHT_HIP,      PoseLandmark.RIGHT_KNEE,  PoseLandmark.RIGHT_ANKLE),
    'left_ankle':  (PoseLandmark.LEFT_KNEE,      PoseLandmark.LEFT_ANKLE,
                    PoseLandmark.LEFT_FOOT_INDEX, PoseLandmark.LEFT_HEEL),
    'right_ankle': (PoseLandmark.RIGHT_KNEE,     PoseLandmark.RIGHT_ANKLE,
                    PoseLandmark.RIGHT_FOOT_INDEX, PoseLandmark.RIGHT_HEEL),
}


def _detect_subject_orientation(video_path):
    mp, mp_python, PoseLandmarker, PoseLandmarkerOptions, RunningMode = get_mediapipe_bindings()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    sample_count = min(10, total)
    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = PoseLandmarkerOptions(
        base_options=base_opts, running_mode=RunningMode.IMAGE,
        num_poses=1, min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5)
    landmarker = PoseLandmarker.create_from_options(opts)
    vertical_votes = 0
    detections = 0
    for i in range(sample_count):
        target_frame = int(total * (i + 1) / (sample_count + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            continue
        lm = result.pose_landmarks[0]
        ls = lm[PoseLandmark.LEFT_SHOULDER]
        rs = lm[PoseLandmark.RIGHT_SHOULDER]
        lh = lm[PoseLandmark.LEFT_HIP]
        rh = lm[PoseLandmark.RIGHT_HIP]
        sx = (ls.x + rs.x) / 2
        sy = (ls.y + rs.y) / 2
        hx = (lh.x + rh.x) / 2
        hy = (lh.y + rh.y) / 2
        dx = abs(hx - sx)
        dy = abs(hy - sy)
        detections += 1
        if dy > dx:
            vertical_votes += 1
    cap.release()
    landmarker.close()
    if detections == 0:
        return False
    print(f"  [{os.path.basename(video_path)}] vertical_votes={vertical_votes}/{detections} → needs_rotation={vertical_votes <= detections / 2}")
    return vertical_votes > detections / 2


def _subject_bounds_from_landmarks(pixel_landmarks, frame_w, frame_h):
    visible = [(lm.x, lm.y) for lm in pixel_landmarks if lm.visibility > 0.5]
    if not visible:
        return None
    xs = np.asarray([p[0] * frame_w for p in visible], dtype=float)
    ys = np.asarray([p[1] * frame_h for p in visible], dtype=float)
    return {
        'min_x': float(np.min(xs)), 'max_x': float(np.max(xs)),
        'min_y': float(np.min(ys)), 'max_y': float(np.max(ys)),
        'width': float(np.max(xs) - np.min(xs)),
        'height': float(np.max(ys) - np.min(ys)),
        'visible_count': int(len(visible)),
    }


def process_video(video_path, ann_dir, progress_cb, status_cb,
                  target_output_size=None, needs_rotation=None,
                  crop_rect=None, cache_key=None, cache_meta=None):
    mp, mp_python, PoseLandmarker, PoseLandmarkerOptions, RunningMode = get_mediapipe_bindings()

    cached_markup = _load_cached_markup(cache_key) if cache_key else None

    if cache_key:
        cached = _load_cached_video_result(cache_key)
        if cached is not None:
            if cached.get('confidence_data') is None or cached['confidence_data'].empty:
                status_cb("Cached data missing confidence scores - reprocessing...")
            else:
                cached['_cache_key'] = cache_key
                cached['_cache_meta'] = cache_meta or {}
                cached['_cached_markup'] = cached_markup
                status_cb("Loaded cached analysis")
                return cached

    if needs_rotation is None:
        status_cb("Detecting subject orientation…")
        needs_rotation = _detect_subject_orientation(video_path)
    if needs_rotation:
        status_cb("Subject is upright — rotating 90° CCW for analysis")
    else:
        status_cb("Subject is sideways — no rotation needed")

    if crop_rect is None:
        crop_rect = detect_crop_region(video_path, needs_rotation)

    cropped_size = None
    if crop_rect:
        cx, cy, cw_r, ch_r = crop_rect
        cropped_size = (cw_r, ch_r)
        status_cb(f"Cropping black borders: {cx},{cy} {cw_r}x{ch_r}")
    else:
        status_cb("No significant black borders detected")

    if target_output_size is None and cropped_size:
        target_output_size = cropped_size

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        status_cb(f"ERROR opening {os.path.basename(video_path)}")
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or SLOWMO_FPS

    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = PoseLandmarkerOptions(
        base_options=base_opts, running_mode=RunningMode.VIDEO,
        num_poses=1, min_pose_detection_confidence=0.7,
        min_pose_presence_confidence=0.7, min_tracking_confidence=0.7,
        output_segmentation_masks=False)
    landmarker = PoseLandmarker.create_from_options(opts)

    world_rows, pixel_rows, landmarks, landmark_depths, confidence_rows = [], [], [], [], []
    landmarks_for_crop = []
    world_landmarks_list = []
    skeleton_width_rows = []
    frame_count = 0

    frame_output_dir = ann_dir
    if cache_key:
        frame_output_dir = os.path.join(_cache_dir(cache_key), 'frames')
        os.makedirs(frame_output_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        progress_cb(frame_count / max(1, total))

        if needs_rotation:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if crop_rect:
            cx, cy, cw_r, ch_r = crop_rect
            frame = frame[cy:cy+ch_r, cx:cx+cw_r]

        view_frame = frame
        _last_view_w = frame.shape[1]   
        _last_view_h = frame.shape[0]   

        pad_left, pad_top = 0, 0
        pad_fw, pad_fh = frame.shape[1], frame.shape[0]
        if target_output_size:
            h, w = frame.shape[:2]
            target_w, target_h = target_output_size
            if w != target_w or h != target_h:
                pad_top  = (target_h - h) // 2
                pad_left = (target_w - w) // 2
                canvas = np.zeros((target_h, target_w, frame.shape[2]), dtype=frame.dtype)
                canvas[pad_top:pad_top+h, pad_left:pad_left+w] = frame
                frame = canvas
                pad_fw, pad_fh = target_w, target_h

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms  = int((frame_count / fps) * 1000)
        result = landmarker.detect_for_video(mp_img, ts_ms)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            pixel_lm = result.pose_landmarks[0]
            world_lm = (result.pose_world_landmarks[0]
                        if result.pose_world_landmarks else pixel_lm)
            world_landmarks_list.append(world_lm)

            raw = view_frame.copy()
            h, w = raw.shape[:2]
            if SAVE_HEIGHT and h > SAVE_HEIGHT:
                nw = int(w * SAVE_HEIGHT / h)
                raw = cv2.resize(raw, (nw, SAVE_HEIGHT), interpolation=cv2.INTER_AREA)
            raw_path = os.path.join(frame_output_dir, f"raw_{frame_count:06d}.jpg")
            cv2.imwrite(raw_path, raw, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

            view_w, view_h = view_frame.shape[1], view_frame.shape[0]
            def _remap_lm(lm, _pl=pad_left, _pt=pad_top, _pfw=pad_fw, _pfh=pad_fh,
                          _vw=view_w, _vh=view_h):
                nx = (lm.x * _pfw - _pl) / _vw if _vw > 0 else lm.x
                ny = (lm.y * _pfh - _pt) / _vh if _vh > 0 else lm.y
                return SimpleLandmark(nx, ny, lm.visibility)
            remapped_pixel_lm = [_remap_lm(lm) for lm in pixel_lm]
            landmarks.append((raw_path, remapped_pixel_lm))
            
            landmarks_for_crop.append((None, remapped_pixel_lm))

            if DEBUG_SKELETON_WIDTH:
                bounds = _subject_bounds_from_landmarks(remapped_pixel_lm, view_w, view_h)
                if bounds is not None:
                    span = max(bounds['width'], bounds['height'])
                    subject_scale = max(0.45, min(2.75, span / 260.0))
                    skeleton_width_rows.append({
                        'frame_num': frame_count,
                        'frame_width': int(view_w),
                        'frame_height': int(view_h),
                        'visible_landmarks': bounds['visible_count'],
                        'subject_width_px': bounds['width'],
                        'subject_height_px': bounds['height'],
                        'subject_span_px': span,
                        'subject_width_ratio': bounds['width'] / max(1, view_w),
                        'subject_height_ratio': bounds['height'] / max(1, view_h),
                        'subject_scale': subject_scale,
                        'estimated_line_thickness_8px': DEBUG_SKELETON_BASE_THICKNESS * subject_scale,
                    })

            direction = determine_walking_direction(pixel_lm)
            w_row = {'frame_num': frame_count, '_direction': direction}
            p_row = {'frame_num': frame_count, '_direction': direction}

            for name, defs in JOINT_DEFS.items():
                jt = name.split('_')[-1]
                A, B, C = defs[0], defs[1], defs[2]
                D = defs[3] if len(defs) == 4 else None

                if A is None:
                    aw = midpoint_shoulder(world_lm)
                    ap = midpoint_shoulder(pixel_lm)
                else:
                    lm = world_lm[A]
                    aw = (lm.x, lm.y, lm.z)
                    lm = pixel_lm[A]
                    ap = (lm.x, lm.y, 0)

                lm = world_lm[B]
                bw = (lm.x, lm.y, lm.z)
                lm = pixel_lm[B]
                bp = (lm.x, lm.y, 0)
                lm = world_lm[C]
                cw = (lm.x, lm.y, lm.z)
                lm = pixel_lm[C]
                cp = (lm.x, lm.y, 0)

                if D is not None:
                    lm = world_lm[D]
                    dw = (lm.x, lm.y, lm.z)
                    lm = pixel_lm[D]
                    dp = (lm.x, lm.y, 0)
                elif jt == "ankle":
                    hi = PoseLandmark.LEFT_HEEL if name.startswith("left") else PoseLandmark.RIGHT_HEEL
                    lm = world_lm[hi]
                    dw = (lm.x, lm.y, lm.z)
                    lm = pixel_lm[hi]
                    dp = (lm.x, lm.y, 0)
                else:
                    dw = dp = None

                w_row[name] = calculate_angle(aw, bw, cw, dw, direction, jt)
                p_row[name] = calculate_angle(ap, bp, cp, dp, direction, jt)

            for i, landmark in enumerate(world_lm):
                w_row[f'landmark_{i}_x'] = float(landmark.x)
                w_row[f'landmark_{i}_y'] = float(landmark.y)
                w_row[f'landmark_{i}_z'] = float(landmark.z)

            for i, landmark in enumerate(remapped_pixel_lm):
                p_row[f'landmark_{i}_x'] = float(landmark.x)
                p_row[f'landmark_{i}_y'] = float(landmark.y)

            depth_row = {'frame_num': frame_count}
            for i, landmark in enumerate(world_lm):
                depth_row[f'joint_{i}'] = landmark.z
            landmark_depths.append(depth_row)

            conf_row = {'frame_num': frame_count}
            if pixel_lm:
                visibilities = [lm.visibility for lm in pixel_lm]
                conf_row['avg_confidence'] = float(np.mean(visibilities))
            else:
                conf_row['avg_confidence'] = 0.0
            confidence_rows.append(conf_row)

            world_rows.append(w_row)
            pixel_rows.append(p_row)
        else:
            landmarks.append(None)
            landmarks_for_crop.append(None)
            world_landmarks_list.append(None)
            landmark_depths.append(None)
            confidence_rows.append({'frame_num': frame_count, 'avg_confidence': 0.0})


    _fw_for_crop = _last_view_w if '_last_view_w' in locals() else int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    _fh_for_crop = _last_view_h if '_last_view_h' in locals() else int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    landmarker.close()

    df_w = pd.DataFrame(world_rows)
    df_p = pd.DataFrame(pixel_rows)
    df_depths = pd.DataFrame([d for d in landmark_depths if d is not None])
    df_confidence = pd.DataFrame(confidence_rows)

    for df in (df_w, df_p):
        for col in df.columns:
            if col not in ('frame_num', '_direction'):
                df[col] = df[col].astype(np.float32)

    jittery_frames = _detect_jittery_frames(landmarks, threshold=0.04)

    landmark_cols_w = [c for c in df_w.columns if c.startswith('landmark_') and c.endswith(('_x', '_y', '_z'))]
    landmark_cols_p = [c for c in df_p.columns if c.startswith('landmark_') and c.endswith(('_x', '_y'))]

    for frame_idx in sorted(jittery_frames):
        if 0 < frame_idx < len(df_w) - 1:
            for col in landmark_cols_w:
                before = df_w.iloc[frame_idx - 1][col]
                after = df_w.iloc[frame_idx + 1][col]
                if not pd.isna(before) and not pd.isna(after):
                    df_w.at[frame_idx, col] = (before + after) / 2.0
        if 0 < frame_idx < len(df_p) - 1:
            for col in landmark_cols_p:
                before = df_p.iloc[frame_idx - 1][col]
                after = df_p.iloc[frame_idx + 1][col]
                if not pd.isna(before) and not pd.isna(after):
                    df_p.at[frame_idx, col] = (before + after) / 2.0

    df_w = _fix_jitter_outliers(df_w, max_frame_displacement=0.20)
    df_p = _fix_jitter_outliers(df_p, max_frame_displacement=0.15)

    joint_cols_w = set(c for c in df_w.columns if c not in ('frame_num', '_direction'))
    for col in joint_cols_w:
        if col in df_w.columns and len(df_w) > FILTER_ORDER:
            try:
                df_w[col] = _butterworth_lowpass_filter(df_w[col].values, FILTER_CUTOFF, SLOWMO_FPS, FILTER_ORDER)
            except Exception:
                pass

    df_p_filtered = df_p.copy()
    pixel_angle_cols = set(c for c in df_p_filtered.columns
                           if c not in ('frame_num', '_direction') and not c.startswith('landmark_'))
    for col in pixel_angle_cols:
        if col in df_p_filtered.columns and len(df_p_filtered) > FILTER_ORDER:
            try:
                df_p_filtered[col] = _butterworth_lowpass_filter(df_p_filtered[col].values, FILTER_CUTOFF, SLOWMO_FPS, FILTER_ORDER)
            except Exception:
                pass

    df_w = _fix_limb_swaps(df_w)
    df_p = _fix_limb_swaps(df_p)

    if jittery_frames:
        while True:
            frames_added = 0
            sorted_jittery = sorted(jittery_frames)
            for i in range(len(sorted_jittery) - 1):
                current_frame = sorted_jittery[i]
                next_frame = sorted_jittery[i + 1]
                gap = next_frame - current_frame - 1
                if gap > 0 and gap <= 5:
                    for frame_to_fill in range(current_frame + 1, next_frame):
                        if frame_to_fill not in jittery_frames:
                            jittery_frames.add(frame_to_fill)
                            frames_added += 1
            if frames_added == 0:
                break

    if 'avg_confidence' in df_confidence.columns:
        df_confidence['avg_confidence'] = df_confidence['avg_confidence'].astype(np.float32)

    for df in (df_w, df_p, df_p_filtered):
        if '_direction' in df.columns:
            df.drop('_direction', axis=1, inplace=True)

    ad = df_w if USE_WORLD_LANDMARKS else df_p_filtered
    del world_rows, pixel_rows
    gc.collect()

    result = {
        'df_world':           df_w,
        'df_pixel':           df_p,
        'df_pixel_filtered':  df_p_filtered,
        'angle_data':         ad,
        'confidence_data':    df_confidence,
        'step_frames':        [],
        'excluded_regions':   [],
        'all_landmarks':      landmarks,
        'landmark_depths':    df_depths,
        'jittery_frames':     jittery_frames,
        'needs_rotation':     needs_rotation,
        '_cache_key':         cache_key,
        '_cache_meta':        cache_meta or {},
        '_cached_markup':     cached_markup,
        '_fw_for_crop':       _fw_for_crop,
        '_fh_for_crop':       _fh_for_crop,
    }

    _crop_sk = _crop_skeleton_stats(landmarks_for_crop)

    if _crop_sk and needs_rotation:
        _crop_sk_upright = dict(_crop_sk)
        _crop_sk_upright['tight_x_min'] = 1.0 - _crop_sk['tight_y_max']
        _crop_sk_upright['tight_x_max'] = 1.0 - _crop_sk['tight_y_min']
        _crop_sk_upright['tight_y_min'] = _crop_sk['tight_x_min']
        _crop_sk_upright['tight_y_max'] = _crop_sk['tight_x_max']
        _crop_sk_upright['cx'] = 1.0 - _crop_sk['cy']
        _crop_sk_upright['cy'] = _crop_sk['cx']
        _fw_upright = _fh_for_crop
        _fh_upright = _fw_for_crop
        print(f"  UPRIGHT STATS: tight_x=[{_crop_sk_upright['tight_x_min']:.4f}-{_crop_sk_upright['tight_x_max']:.4f}] tight_y=[{_crop_sk_upright['tight_y_min']:.4f}-{_crop_sk_upright['tight_y_max']:.4f}]")
    else:
        _crop_sk_upright = _crop_sk
        _fw_upright = _fw_for_crop
        _fh_upright = _fh_for_crop

    if _crop_sk:
        _debug_crop_stats(landmarks_for_crop, _crop_sk, video_path)

    result['crop_stats']  = _crop_sk_upright
    result['_fw_upright'] = _fw_upright
    result['_fh_upright'] = _fh_upright
    result['crop_rect']   = None
    return result


# gait metrics
def _step_times(step_frames, fps=SLOWMO_FPS):
    if len(step_frames) < 2: return [], []
    lt, rt = [], []
    pf, ps = step_frames[0]
    for f, s in step_frames[1:]:
        t = (f - pf) / fps
        (lt if ps == 'left' else rt).append(t)
        pf, ps = f, s
    return lt, rt


def _cadence(lt, rt):
    all_t = lt + rt
    return 60 / np.mean(all_t) if all_t else 0


def _variability(lt, rt):
    all_t = lt + rt
    return np.std(all_t) / np.mean(all_t) * 100 if len(all_t) >= 2 else 0


def _joint_stats(ad, name):
    if name not in ad: return 0, 0
    a = ad[name].dropna()
    return (float(np.mean(a)), float(np.max(a))) if len(a) else (0, 0)


def _asymmetry(ad, base):
    la = ad.get(f'left_{base}',  pd.Series()).dropna()
    ra = ad.get(f'right_{base}', pd.Series()).dropna()
    return abs(float(np.mean(la)) - float(np.mean(ra))) if len(la) and len(ra) else 0

METRIC_LABELS = {
    'cadence':  ("Cadence",          "steps / min"),
    'step_var': ("Step Variability", "L-R timing CV"),
    'knee_mean':("Knee ROM (mean)",  "avg angle"),
    'knee_peak':("Knee ROM (peak)",  "peak angle"),
    'hip_mean': ("Hip ROM (mean)",   "avg angle"),
    'hip_peak': ("Hip ROM (peak)",   "peak angle"),
    'knee_sym': ("Knee Symmetry",    "L-R difference"),
    'hip_sym':  ("Hip Symmetry",     "L-R difference"),
}
METRIC_ORDER = [
    'cadence', 'step_var',
    'knee_mean', 'knee_peak',
    'hip_mean',  'hip_peak',
    'knee_sym',  'hip_sym',
]
METRIC_HIB = {
    'cadence':   True,  'step_var':  False,
    'knee_mean': None,  'knee_peak': None,
    'hip_mean':  None,  'hip_peak':  None,
    'knee_sym':  False, 'hip_sym':   False,
}

def compute_metrics(ds1, ds2):
    l1, r1 = _step_times(ds1.get('step_frames', []))
    l2, r2 = _step_times(ds2.get('step_frames', []))
    a1, a2 = ds1['angle_data'], ds2['angle_data']
    return {
        'cadence':   pct_change(_cadence(l2, r2),    _cadence(l1, r1)),
        'step_var':  pct_change(_variability(l2, r2), _variability(l1, r1)),
        'knee_mean': pct_change(_joint_stats(a2,'right_knee')[0], _joint_stats(a1,'right_knee')[0]),
        'knee_peak': pct_change(_joint_stats(a2,'right_knee')[1], _joint_stats(a1,'right_knee')[1]),
        'hip_mean':  pct_change(_joint_stats(a2,'right_hip')[0],  _joint_stats(a1,'right_hip')[0]),
        'hip_peak':  pct_change(_joint_stats(a2,'right_hip')[1],  _joint_stats(a1,'right_hip')[1]),
        'knee_sym':  pct_change(_asymmetry(a2,'knee'), _asymmetry(a1,'knee')),
        'hip_sym':   pct_change(_asymmetry(a2,'hip'),  _asymmetry(a1,'hip')),
    }


class FrameCache:
    def __init__(self, limit=CACHE_FRAMES):
        self._cache = OrderedDict()
        self._limit = limit
        self._lock  = threading.RLock()

    def get(self, ds_idx, frame_idx, store):
        key = (ds_idx, frame_idx)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        if not (0 <= frame_idx < len(store)):
            return None
        p = store[frame_idx]
        if p is None: return None
        path = p[0] if isinstance(p, tuple) else p
        img = cv2.imread(path) if isinstance(path, str) else path
        if img is not None:
            with self._lock:
                self._cache[key] = img
                if len(self._cache) > self._limit:
                    self._cache.popitem(last=False)
        return img

    def clear(self):
        with self._lock:
            self._cache.clear()


HELP_TEXT = [
    ("Navigation",    None),
    ("1 / 2",         "Previous / Next frame"),
    ("Mouse drag",    "Scrub timeline on graph"),
    ("9",             "Play / Pause"),
    ("q",             "Quit"),
    ("",              None),
    ("Display",       None),
    ("w",             "Toggle World / Pixel landmarks"),
    ("v",             "Graph: V1 → V2 → Both"),
    ("t",             "Toggle active video"),
    ("c",             "Toggle overlaid cycles / continuous"),
    ("x",             "Toggle cycle angle mode (native / aligned)"),
    ("b",             "Toggle stacked cycle lanes"),
    ("s",             "Toggle resample (cycle view)"),
    ("m",             "Mean curve: off → +data → only"),
    ("F3",            "Toggle ankle norm offset"),
    ("",              None),
    ("Step Editing",  None),
    ("Space",         "Add manual step at current frame (auto-detects foot)"),
    ("Backspace/Del", "Remove nearest manual step"),
    ("g",             "Toggle visibility of suggested steps"),
    ("d",             "Clear manual steps (all videos)"),
    ("",              None),
    ("Exclusions",    None),
    ("Right-click",   "Drag on graph to exclude region"),
    ("Clr Excl btn",  "Clear all excluded regions"),
    ("h / H",         "This help screen"),
]


class CacheManagerDialog(tk.Toplevel):
    def __init__(self, parent, cache_root):
        super().__init__(parent)
        self.title("Cache Manager")
        self.cache_root = cache_root
        self.checkboxes = {}
        self.delete_whole_vars = {}
        self._build_ui()
        self._scan_caches()
        _apply_dynamic_window_size(self, preferred_width=570, preferred_height=450)

    def _build_ui(self):
        canvas_frame = tk.Frame(self, bg=BG)
        canvas_frame.pack(fill='both', expand=True, padx=10, pady=(10, 0))
        canvas = tk.Canvas(canvas_frame, bg=BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient='vertical', command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=BG)
        scrollable.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        scroll_window = canvas.create_window((0, 0), window=scrollable, anchor='nw')
        canvas.bind('<Configure>', lambda e: canvas.itemconfigure(scroll_window, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        self.scrollable_frame = scrollable
        self.canvas = canvas
        self.bind('<Enter>', self._bind_mousewheel_global)
        self.bind('<Leave>', self._unbind_mousewheel_global)
        self.bind('<Destroy>', self._on_destroy)
        bottom_frame = tk.Frame(self, bg=BG)
        bottom_frame.pack(fill='x', padx=10, pady=10)
        self.delete_btn = tk.Button(bottom_frame, text="Delete Selected",
                                   font=("Helvetica", 10), bg='#e74c3c', fg='white',
                                   command=self._delete_selected, state='disabled')
        self.delete_btn.pack(side='left')

    def _on_mousewheel(self, event):
        widget_under_mouse = self.winfo_containing(self.winfo_pointerx(), self.winfo_pointery())
        if not widget_under_mouse or widget_under_mouse.winfo_toplevel() != self:
            return
        if getattr(event, 'delta', 0):
            steps = int(-1 * (event.delta / 120))
            if steps == 0:
                steps = -1 if event.delta > 0 else 1
            self.canvas.yview_scroll(steps, 'units')
        elif event.num == 4:
            self.canvas.yview_scroll(-1, 'units')
        elif event.num == 5:
            self.canvas.yview_scroll(1, 'units')
        return 'break'

    def _bind_mousewheel_global(self, event=None):
        self.bind_all('<MouseWheel>', self._on_mousewheel)
        self.bind_all('<Button-4>', self._on_mousewheel)
        self.bind_all('<Button-5>', self._on_mousewheel)

    def _unbind_mousewheel_global(self, event=None):
        self.unbind_all('<MouseWheel>')
        self.unbind_all('<Button-4>')
        self.unbind_all('<Button-5>')

    def _on_destroy(self, event=None):
        if event is None or event.widget == self:
            self._unbind_mousewheel_global()

    def _scan_caches(self):
        if not os.path.exists(self.cache_root):
            tk.Label(self.scrollable_frame, text="No cache directory found",
                    font=("Helvetica", 10), bg=BG, fg=SUBTEXT).pack(pady=20)
            return
        try:
            cache_dirs = [d for d in os.listdir(self.cache_root)
                         if os.path.isdir(os.path.join(self.cache_root, d))]
        except Exception as e:
            tk.Label(self.scrollable_frame, text=f"Error reading cache: {e}",
                    font=("Helvetica", 10), bg=BG, fg='#e74c3c').pack(pady=20)
            return
        if not cache_dirs:
            tk.Label(self.scrollable_frame, text="No cached videos found",
                    font=("Helvetica", 10), bg=BG, fg=SUBTEXT).pack(pady=20)
            return
        for cache_key in sorted(cache_dirs):
            self._add_cache_entry(cache_key)

    def _add_cache_entry(self, cache_key):
        cache_path = os.path.join(self.cache_root, cache_key)
        meta = {}
        manifest_path = os.path.join(cache_path, 'manifest.json')
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, 'r') as f:
                    meta = json.load(f)
            except:
                pass
        video_name = meta.get('video_name', f"Unknown - {cache_key[:16]}...")
        result_pkl = os.path.join(cache_path, 'result.pkl')
        markup_json = os.path.join(cache_path, 'markup.json')
        frames_dir = os.path.join(cache_path, 'frames')
        has_result = os.path.exists(result_pkl)
        has_markup = os.path.exists(markup_json)
        has_frames = os.path.isdir(frames_dir)
        total_size = 0
        try:
            for root, dirs, files in os.walk(cache_path):
                for f in files:
                    total_size += os.path.getsize(os.path.join(root, f))
        except:
            pass
        size_str = f"{total_size / (1024*1024):.1f} MB" if total_size > 0 else "0 KB"
        entry_frame = tk.Frame(self.scrollable_frame, bg=BG2, relief='solid', borderwidth=1)
        entry_frame.pack(fill='x', pady=6)
        header_frame = tk.Frame(entry_frame, bg=BG2)
        header_frame.pack(fill='x', padx=8, pady=6)
        delete_whole_var = tk.BooleanVar(value=False)
        self.delete_whole_vars[cache_key] = (delete_whole_var, cache_path)
        delete_whole_cb = tk.Checkbutton(header_frame, text="", variable=delete_whole_var,
                                        font=("Helvetica", 9), bg=BG2, fg=TEXT,
                                        command=self._update_delete_button)
        delete_whole_cb.pack(side='right', padx=4)
        tk.Label(header_frame, text=f"📹 {video_name}",
                font=("Helvetica", 10, "bold"), bg=BG2, fg=TEXT).pack(side='left', fill='x', expand=True)
        tk.Label(header_frame, text=size_str,
                font=("Helvetica", 8), bg=BG2, fg=SUBTEXT).pack(side='left', padx=4)
        data_items = []
        if has_result:
            data_items.append(('Coordinates & Angles', result_pkl))
        if has_markup:
            data_items.append(('Step Marks', markup_json))
        if has_frames:
            data_items.append(('Frame Cache', frames_dir))
        if data_items:
            items_frame = tk.Frame(entry_frame, bg=BG)
            items_frame.pack(fill='x', padx=8, pady=(0, 6))
            for item_name, item_path in data_items:
                checkbox_var = tk.BooleanVar(value=False)
                self.checkboxes[(cache_key, item_name)] = (checkbox_var, item_path)
                item_frame = tk.Frame(items_frame, bg=BG)
                item_frame.pack(fill='x', pady=2)
                checkbox = tk.Checkbutton(item_frame, text=item_name, variable=checkbox_var,
                                         font=("Helvetica", 9), bg=BG, fg=TEXT,
                                         command=self._update_delete_button)
                checkbox.pack(anchor='w')
        else:
            tk.Label(entry_frame, text="(Empty cache)", font=("Helvetica", 9),
                    bg=BG, fg=SUBTEXT).pack(anchor='w', padx=8, pady=6)

    def _update_delete_button(self):
        has_items_selected = any(var.get() for var, _ in self.checkboxes.values())
        has_whole_selected = any(var.get() for var, _ in self.delete_whole_vars.values())
        self.delete_btn.config(state='normal' if (has_items_selected or has_whole_selected) else 'disabled')

    def _delete_selected(self):
        selected_items = [(key, path) for key, (var, path) in self.checkboxes.items() if var.get()]
        selected_whole = [(key, path) for key, (var, path) in self.delete_whole_vars.items() if var.get()]
        if not selected_items and not selected_whole:
            messagebox.showwarning("No Selection", "Please select items to delete", parent=self)
            return
        total_count = len(selected_items) + len(selected_whole)
        if not messagebox.askyesno("Confirm Delete",
                                   f"Delete {total_count} selected item(s)?\n\nThis cannot be undone.",
                                   parent=self):
            return
        failed = []
        for (cache_key, item_name), item_path in selected_items:
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            except Exception as e:
                failed.append(f"{item_name}: {e}")
        for cache_key, cache_path in selected_whole:
            try:
                if os.path.exists(cache_path):
                    shutil.rmtree(cache_path)
            except Exception as e:
                failed.append(f"Cache {cache_key[:8]}...: {e}")
        if failed:
            messagebox.showerror("Partial Failure", f"Failed to delete:\n" + "\n".join(failed), parent=self)
        else:
            messagebox.showinfo("Success", f"{total_count} item(s) deleted", parent=self)
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.checkboxes = {}
        self.delete_whole_vars = {}
        self._scan_caches()


class SettingsDialog(tk.Toplevel):
    def __init__(self, parent, dashboard):
        super().__init__(parent)
        self.title("Settings")
        self.dashboard = dashboard
        self.configure(bg=BG)
        self._build_ui()
        _apply_dynamic_window_size(self, preferred_width=550, preferred_height=700)

    def _build_ui(self):
        title_frame = tk.Frame(self, bg=BG2, height=50)
        title_frame.pack(fill='x', padx=0, pady=0)
        tk.Label(title_frame, text="Settings",
                font=("Helvetica", 12, "bold"), bg=BG2, fg=TEXT).pack(side='left', padx=10, pady=8)
        content = tk.Frame(self, bg=BG)
        content.pack(fill='both', expand=True, padx=20, pady=20)

        conf_frame = tk.Frame(content, bg=BG)
        conf_frame.pack(fill='x', pady=10)
        tk.Label(conf_frame, text="Confidence Scores", font=("Helvetica", 10), bg=BG, fg=TEXT).pack(side='left')
        tk.Label(conf_frame, text="Toggle with Alt+C", font=("Helvetica", 8), bg=BG, fg=SUBTEXT).pack(side='left', padx=(10, 0))
        self.conf_var = tk.BooleanVar(value=self.dashboard.show_confidence)
        tk.Checkbutton(conf_frame, text="Show confidence scores on graph",
                       variable=self.conf_var, bg=BG, fg=TEXT,
                       activebackground=BG, activeforeground=TEXT,
                       command=self._toggle_confidence).pack(side='right')

        tk.Frame(content, bg=BG2, height=1).pack(fill='x', pady=10)

        rmse_frame = tk.Frame(content, bg=BG)
        rmse_frame.pack(fill='x', pady=10)
        tk.Label(rmse_frame, text="RMSE Threshold (%)", font=("Helvetica", 10), bg=BG, fg=TEXT).pack(side='left')
        tk.Label(rmse_frame, text="Lower = stricter filtering", font=("Helvetica", 8), bg=BG, fg=SUBTEXT).pack(side='left', padx=(10, 0))
        self.rmse_var = tk.DoubleVar(value=self.dashboard.rmse_threshold)
        self.rmse_slider = tk.Scale(rmse_frame, from_=0.0, to=100.0, resolution=0.5,
                                    variable=self.rmse_var, orient='horizontal',
                                    bg=BG, fg=TEXT, highlightthickness=0, troughcolor=BG3,
                                    command=self._on_rmse_change)
        self.rmse_slider.pack(side='right', fill='x', expand=True, padx=(10, 0))

        tk.Frame(content, bg=BG2, height=1).pack(fill='x', pady=10)

        cache_frame = tk.Frame(content, bg=BG)
        cache_frame.pack(fill='x', pady=10)
        tk.Label(cache_frame, text="Cache Management", font=("Helvetica", 10), bg=BG, fg=TEXT).pack(side='left')
        tk.Button(cache_frame, text="Manage Cache", font=("Helvetica", 9),
                 bg=BG3, fg=TEXT, relief='flat', padx=8,
                 command=self._open_cache_manager).pack(side='right')

        tk.Frame(content, bg=BG2, height=1).pack(fill='x', pady=10)

        export_frame = tk.Frame(content, bg=BG)
        export_frame.pack(fill='x', pady=10)
        tk.Label(export_frame, text="Export Report", font=("Helvetica", 10), bg=BG, fg=TEXT).pack(side='left')
        tk.Button(export_frame, text="Print to PDF", font=("Helvetica", 9),
                 bg='#27ae60', fg=TEXT, relief='flat', padx=8,
                 command=self._open_pdf_export).pack(side='right')

        tk.Frame(content, bg=BG2, height=1).pack(fill='x', pady=10)

        jitter_frame = tk.Frame(content, bg=BG)
        jitter_frame.pack(fill='x', pady=10)
        tk.Label(jitter_frame, text="Jitter Frames", font=("Helvetica", 10, "bold"), bg=BG, fg=TEXT).pack(side='left')
        self.remove_jitter_var = tk.BooleanVar(value=self.dashboard.remove_jitter_frames)
        tk.Checkbutton(content, text="Remove jitter frames",
                       variable=self.remove_jitter_var, bg=BG, fg=TEXT,
                       activebackground=BG, activeforeground=TEXT,
                       command=self._toggle_remove_jitter_frames).pack(anchor='w', pady=2)
        self.show_jitter_var = tk.BooleanVar(value=self.dashboard.show_jitter_frames)
        tk.Checkbutton(content, text="Show jitter frames in red",
                       variable=self.show_jitter_var, bg=BG, fg=TEXT,
                       activebackground=BG, activeforeground=TEXT,
                       command=self._toggle_show_jitter_frames).pack(anchor='w', pady=(2, 10))

        tk.Frame(content, bg=BG2, height=1).pack(fill='x', pady=10)

        graph_frame = tk.Frame(content, bg=BG)
        graph_frame.pack(fill='x', pady=10)
        tk.Label(graph_frame, text="Graph Display", font=("Helvetica", 10), bg=BG, fg=TEXT).pack(side='left')
        self._graph_display_btn = tk.Button(graph_frame, font=("Helvetica", 9),
                 bg=BG3, fg=TEXT, relief='flat', padx=8,
                 command=self._toggle_graph_display_mode)
        self._graph_display_btn.pack(side='right')
        self._update_graph_display_btn()

        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.pack(fill='x', padx=20, pady=15)
        tk.Button(btn_frame, text="Close", font=("Helvetica", 10),
                 bg=BG3, fg=TEXT, relief='flat', padx=12, pady=6,
                 command=self.destroy).pack(side='right')

    def _toggle_confidence(self):
        self.dashboard.show_confidence = self.conf_var.get()
        self.dashboard.redraw_graphs()

    def _toggle_remove_jitter_frames(self):
        self.dashboard.remove_jitter_frames = self.remove_jitter_var.get()
        if self.dashboard._marking_phase:
            self.dashboard._markup_show_frames()
        else:
            self.dashboard._show_video_frames()

    def _toggle_show_jitter_frames(self):
        self.dashboard.show_jitter_frames = self.show_jitter_var.get()
        if self.dashboard._marking_phase:
            self.dashboard._markup_show_frames()
        else:
            self.dashboard._show_video_frames()

    def _on_rmse_change(self, value):
        try:
            self.dashboard.rmse_threshold = max(1.0, min(100.0, float(value)))
            _save_ui_settings({'rmse_threshold': self.dashboard.rmse_threshold})
            self.dashboard.redraw_graphs()
        except Exception:
            pass

    def _update_graph_display_btn(self):
        mode_text = "SE Shading" if self.dashboard.graph_display_mode == 'se_shading' else "Lines"
        self._graph_display_btn.config(text=mode_text)

    def _toggle_graph_display_mode(self):
        if self.dashboard.graph_display_mode == 'se_shading':
            self.dashboard.graph_display_mode = 'lines_only'
            self.dashboard.show_data = True
        else:
            self.dashboard.graph_display_mode = 'se_shading'
            self.dashboard.show_data = False
        self._update_graph_display_btn()
        self.dashboard.redraw_graphs()

    def _open_cache_manager(self):
        if self.dashboard._cache_manager_dialog is not None and self.dashboard._cache_manager_dialog.winfo_exists():
            self.dashboard._cache_manager_dialog.lift()
            self.dashboard._cache_manager_dialog.focus()
        else:
            dialog = CacheManagerDialog(self, _cache_root_dir())
            self.dashboard._cache_manager_dialog = dialog
            def on_close():
                self.dashboard._cache_manager_dialog = None
                dialog.destroy()
            dialog.protocol("WM_DELETE_WINDOW", on_close)

    def _open_pdf_export(self):
        if self.dashboard._pdf_export_dialog is not None and self.dashboard._pdf_export_dialog.winfo_exists():
            self.dashboard._pdf_export_dialog.lift()
            self.dashboard._pdf_export_dialog.focus()
        else:
            dialog = PDFExportDialog(self, self.dashboard)
            self.dashboard._pdf_export_dialog = dialog
            def on_close():
                self.dashboard._pdf_export_dialog = None
                dialog.destroy()
            dialog.protocol("WM_DELETE_WINDOW", on_close)


class PDFExportDialog(tk.Toplevel):
    OUTCOME_MEASURES = {
        'cadence': 'Cadence (% change)',
        'step_var': 'Step Variability (% change)',
        'knee_mean': 'Knee Angle Mean (% change)',
        'knee_peak': 'Knee Angle Peak (% change)',
        'hip_mean': 'Hip Angle Mean (% change)',
        'hip_peak': 'Hip Angle Peak (% change)',
        'knee_sym': 'Knee Symmetry (% change)',
        'hip_sym': 'Hip Symmetry (% change)',
    }

    def __init__(self, parent, dashboard):
        super().__init__(parent)
        self.title("Export to PDF")
        self.dashboard = dashboard
        self.configure(bg=BG)
        self._build_ui()
        _apply_dynamic_window_size(self, preferred_width=550, preferred_height=900)

    def _build_ui(self):
        title_frame = tk.Frame(self, bg=BG2, height=50)
        title_frame.pack(fill='x', padx=0, pady=0)
        tk.Label(title_frame, text="Export Gait Analysis Report",
                font=("Helvetica", 12, "bold"), bg=BG2, fg=TEXT).pack(side='left', padx=10, pady=8)

        canvas_frame = tk.Frame(self, bg=BG)
        canvas_frame.pack(fill='both', expand=True, padx=10, pady=10)
        canvas = tk.Canvas(canvas_frame, bg=BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient='vertical', command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=BG)
        scrollable.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=scrollable, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        graphs_frame = tk.LabelFrame(scrollable, text="Select Graphs & Options", bg=BG, fg=TEXT,
                                    font=("Helvetica", 10, "bold"))
        graphs_frame.pack(fill='x', pady=(10, 5))
        self.graph_vars = {}
        self.graph_vars['continuous'] = tk.BooleanVar(value=True)
        self.graph_vars['cycles'] = tk.BooleanVar(value=True)
        self.graph_options = {}
        tk.Checkbutton(graphs_frame, text="Continuous View (Raw angle data)",
                      variable=self.graph_vars['continuous'], bg=BG, fg=TEXT,
                      activebackground=BG, activeforeground=TEXT).pack(anchor='w', padx=10, pady=4)
        self.graph_options['continuous'] = self._create_graph_options_frame(graphs_frame, 'continuous')
        tk.Checkbutton(graphs_frame, text="Overlaid Cycles (Normalized gait cycles)",
                      variable=self.graph_vars['cycles'], bg=BG, fg=TEXT,
                      activebackground=BG, activeforeground=TEXT).pack(anchor='w', padx=10, pady=4)
        self.graph_options['cycles'] = self._create_graph_options_frame(graphs_frame, 'cycles')

        limbs_frame = tk.LabelFrame(scrollable, text="Select Limbs", bg=BG, fg=TEXT,
                                   font=("Helvetica", 10, "bold"))
        limbs_frame.pack(fill='x', pady=5)
        self.limb_vars = {}
        limbs = [('left_hip', 'Left Hip'), ('left_knee', 'Left Knee'), ('left_ankle', 'Left Ankle'),
                 ('right_hip', 'Right Hip'), ('right_knee', 'Right Knee'), ('right_ankle', 'Right Ankle')]
        for key, label in limbs:
            self.limb_vars[key] = tk.BooleanVar(value=True)
            tk.Checkbutton(limbs_frame, text=label, variable=self.limb_vars[key], bg=BG, fg=TEXT,
                          activebackground=BG, activeforeground=TEXT).pack(anchor='w', padx=10, pady=2)

        measures_frame = tk.LabelFrame(scrollable, text="Outcome Measures", bg=BG, fg=TEXT,
                                      font=("Helvetica", 10, "bold"))
        measures_frame.pack(fill='x', pady=5)
        self.measure_vars = {}
        for key, label in self.OUTCOME_MEASURES.items():
            self.measure_vars[key] = tk.BooleanVar(value=True)
            tk.Checkbutton(measures_frame, text=label, variable=self.measure_vars[key], bg=BG, fg=TEXT,
                          activebackground=BG, activeforeground=TEXT).pack(anchor='w', padx=10, pady=2)

        tk.Label(scrollable, text="Note: PDF will be generated for currently loaded video pair.",
                font=("Helvetica", 9), bg=BG, fg=SUBTEXT, wraplength=400, justify='left').pack(anchor='w', padx=10, pady=10)

        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.pack(fill='x', padx=10, pady=15)
        tk.Button(btn_frame, text="Print to PDF", font=("Helvetica", 11),
                 bg='#27ae60', fg=TEXT, relief='flat', padx=15, pady=8,
                 command=self._export_pdf).pack(side='right', padx=4)
        tk.Button(btn_frame, text="Cancel", font=("Helvetica", 10),
                 bg=BG3, fg=TEXT, relief='flat', padx=12, pady=6,
                 command=self.destroy).pack(side='right', padx=4)

    def _create_graph_options_frame(self, parent, graph_type):
        options_frame = tk.Frame(parent, bg=BG)
        options_frame.pack(fill='x', padx=30, pady=2)
        version_var = tk.StringVar(value='both')
        self.graph_options[f'{graph_type}_version'] = version_var
        tk.Label(options_frame, text="Show:", bg=BG, fg=TEXT, font=("Helvetica", 9)).pack(side='left', padx=(0, 5))
        for v, t in [('v1', 'V1'), ('v2', 'V2'), ('both', 'Both')]:
            tk.Radiobutton(options_frame, text=t, variable=version_var, value=v,
                          bg=BG, fg=TEXT, activebackground=BG, activeforeground=TEXT).pack(side='left', padx=3)
        excluded_var = tk.BooleanVar(value=True)
        self.graph_options[f'{graph_type}_excluded'] = excluded_var
        tk.Checkbutton(options_frame, text="Show excluded areas", variable=excluded_var,
                      bg=BG, fg=TEXT, activebackground=BG, activeforeground=TEXT,
                      font=("Helvetica", 9)).pack(side='left', padx=10)
        return options_frame

    def _export_pdf(self):
        if not HAS_REPORTLAB:
            messagebox.showerror("Missing Dependency",
                               "reportlab is required for PDF export.\n\nInstall with: pip install reportlab")
            return
        if len(self.dashboard.datasets) < 2:
            messagebox.showwarning("Not Enough Videos", "Comparison reports require 2 videos.")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")],
            initialfile=f"gait_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        if not file_path:
            return
        try:
            graphs = {}
            for graph_type in ['continuous', 'cycles']:
                if self.graph_vars[graph_type].get():
                    graphs[graph_type] = {
                        'show_versions': self.graph_options[f'{graph_type}_version'].get(),
                        'include_excluded': self.graph_options[f'{graph_type}_excluded'].get(),
                    }
            limbs = {k: v.get() for k, v in self.limb_vars.items()}
            self.dashboard._generate_pdf(file_path, graphs=graphs, limbs=limbs,
                                         measures={k: v.get() for k, v in self.measure_vars.items()})
            messagebox.showinfo("Success", f"PDF exported to:\n{file_path}")
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate PDF:\n{str(e)}")


# main dashboard

class GaitAnalysisDashboard(tk.Tk):

    def __init__(self):
        super().__init__()
        self.configure(bg=BG)
        self.title("Gait Analysis")

        self.datasets           = []
        self.video_names        = ["Video 1", "Video 2"]
        self.current_frame_idx  = 0
        self.total_frames       = 0
        self.progress           = 0.0
        self._status_msg        = tk.StringVar(value="Loading…")

        self.df_world           = pd.DataFrame()
        self.df_pixel           = pd.DataFrame()
        self.df_pixel_filtered  = pd.DataFrame()
        self.angle_data         = pd.DataFrame()

        self.joint_visibility = {k: True for k in
            ('left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle')}

        self.graph_show_mode      = 'both'
        self.show_overlaid_cycles = False
        self.resample_cycles      = False
        self.resample_length      = 100
        self.show_mean            = True
        self.mean_only            = False
        self.show_normative       = True
        self.show_data            = False
        self.align_cycle_angles   = False
        self.stack_cycle_lanes    = False
        self.graph_display_mode   = 'se_shading'
        self.show_confidence      = False
        self.show_outliers_only   = False
        self.show_jitter_frames   = False
        self.remove_jitter_frames = True
        self.active_dataset_idx   = 0
        ui_settings = _load_ui_settings()
        self.skeleton_thickness = float(ui_settings.get('skeleton_thickness', DEFAULT_SKELETON_THICKNESS))
        self.skeleton_thickness = max(0.0, min(float(DRAW_THICKNESS), self.skeleton_thickness))
        self.rmse_threshold = float(ui_settings.get('rmse_threshold', 35.0))
        self.rmse_threshold = max(0.0, min(100.0, self.rmse_threshold))
        self.manual_step_mode     = True
        self.manual_side          = 'right'
        self.show_suggestions     = False
        self.playing              = False
        self._marking_phase       = None
        self._marking_video_idx   = 0
        self._markup_frame        = None
        self._play_after_id       = None
        self._graph_resize_after_id = None
        self._graph_dragging      = [False, False, False]   # per-graph
        self._exclusion_selecting = [False, False, False]
        self._exclusion_start     = [None, None, None]
        self._graph_limb_btns     = {}

        # per-graph zoom state (ankle=0, hip=1, knee=2)
        self._ax_xlim_full        = [None, None, None]
        self._ax_xlim_per_mode    = [{}, {}, {}]

        self._cache     = FrameCache()
        self._stop_pf   = False
        self._pf_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._pf_thread.start()

        self._settings_dialog = None
        self._cache_manager_dialog = None
        self._pdf_export_dialog = None

        self._build_ui()
        _apply_dynamic_window_size(self, preferred_width=1400, preferred_height=860)
        self._center_on_screen()
        self._bind_keys()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after_idle(self.redraw_graphs)

    def _center_on_screen(self):
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        if width <= 1 or height <= 1:
            geom = self.geometry().split('+')[0]
            if 'x' in geom:
                width, height = map(int, geom.split('x'))
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        x = max((screen_w - width) // 2, 0)
        y = max((screen_h - height) // 2, 0)
        self.geometry(f"{width}x{height}+{x}+{y}")

    # ui build

    def _build_ui(self):
        # header
        hdr = tk.Frame(self, bg=BG2, height=44)
        hdr.pack(fill='x', side='top')
        hdr.pack_propagate(False)

        try:
            _img = Image.open(resource_path("novita.png"))
            _img.thumbnail((38, 36), Image.LANCZOS)
            self._logo_img = ImageTk.PhotoImage(_img)
            tk.Label(hdr, image=self._logo_img, bg=BG2).pack(side='left', padx=4)
        except FileNotFoundError:
            pass

        title_label = tk.Label(hdr, text="NOVITA GAIT ANALYSIS",
                 font=("Coiny Cyrillic", 17), bg=BG2, fg=ACCENT, cursor="hand2")
        title_label.pack(side='left', pady=(6, 0))
        title_label.bind('<Button-1>', lambda e: self._open_settings())

        tk.Button(hdr, text="Re-mark", font=("Helvetica", 9),
              bg=BG3, fg=TEXT, relief='flat', padx=8,
              command=self._restart_marking_wizard
              ).pack(side='right', padx=2, pady=8)

        tk.Button(hdr, text="Select", font=("Helvetica", 9),
                  bg=BG3, fg=TEXT, relief='flat', padx=8,
                  command=self.find_videos
                  ).pack(side='right', padx=10, pady=8)

        # main content area
        main = tk.Frame(self, bg=BG)
        main.pack(fill='both', expand=True, padx=8, pady=(8, 8))
        self._main_content = main

        # layout: videos left, graphs right with 2x2 grid
        # content panels: ankle, hip, knee, videos
        main.grid_rowconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=0)
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=0, minsize=250)

        content = tk.Frame(main, bg=BG)
        content.grid(row=0, column=0, sticky='nsew')

        sidebar = tk.Frame(main, bg=BG, width=250)
        sidebar.grid(row=0, column=1, sticky='nsew', padx=(4, 0))
        sidebar.grid_propagate(False)
        sidebar.grid_rowconfigure(0, weight=0)
        sidebar.grid_rowconfigure(1, weight=1)
        sidebar.grid_columnconfigure(0, weight=1)

        content.grid_rowconfigure(0, weight=1, uniform='row')
        content.grid_rowconfigure(1, weight=1, uniform='row')
        content.grid_rowconfigure(2, weight=1, uniform='row')
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=2)

        # hip graph (row 0, col 1)
        hip_frame = tk.Frame(content, bg=BG2, bd=1, relief='flat')
        hip_frame.grid(row=0, column=1, sticky='nsew', padx=(0, 2), pady=(2, 3))
        self._build_graph_limb_header(hip_frame, 'left_hip', 'right_hip')
        self._fig_hip, self._ax_hip = plt.subplots(figsize=(5.5, 2.6), dpi=100)
        self._style_ax(self._ax_hip, self._fig_hip)
        self._canvas_hip = FigureCanvasTkAgg(self._fig_hip, master=hip_frame)
        self._canvas_hip.get_tk_widget().pack(fill='both', expand=True)
        self._canvas_hip.mpl_connect('button_press_event',   lambda e: self._on_graph_click(e, 1))
        self._canvas_hip.mpl_connect('motion_notify_event',  lambda e: self._on_graph_drag(e, 1))
        self._canvas_hip.mpl_connect('button_release_event', lambda e: self._on_graph_release(e, 1))
        self._canvas_hip.get_tk_widget().bind('<MouseWheel>', lambda e: self._on_canvas_scroll(e, 1))
        self._canvas_hip.get_tk_widget().bind('<Button-4>',   lambda e: self._on_canvas_scroll(e, 1))
        self._canvas_hip.get_tk_widget().bind('<Button-5>',   lambda e: self._on_canvas_scroll(e, 1))

        # display controls
        btn_panel = tk.Frame(sidebar, bg=BG2)
        btn_panel.grid(row=0, column=0, sticky='new', pady=(0, 3))
        self._build_buttons_panel(btn_panel)

        # knee graph (row 1, col 1)
        knee_frame = tk.Frame(content, bg=BG2, bd=1, relief='flat')
        knee_frame.grid(row=1, column=1, sticky='nsew', padx=(0, 2), pady=(3, 3))
        self._build_graph_limb_header(knee_frame, 'left_knee', 'right_knee')
        self._fig_knee, self._ax_knee = plt.subplots(figsize=(5.5, 2.6), dpi=100)
        self._style_ax(self._ax_knee, self._fig_knee)
        self._canvas_knee = FigureCanvasTkAgg(self._fig_knee, master=knee_frame)
        self._canvas_knee.get_tk_widget().pack(fill='both', expand=True)
        self._canvas_knee.mpl_connect('button_press_event',   lambda e: self._on_graph_click(e, 2))
        self._canvas_knee.mpl_connect('motion_notify_event',  lambda e: self._on_graph_drag(e, 2))
        self._canvas_knee.mpl_connect('button_release_event', lambda e: self._on_graph_release(e, 2))
        self._canvas_knee.get_tk_widget().bind('<MouseWheel>', lambda e: self._on_canvas_scroll(e, 2))
        self._canvas_knee.get_tk_widget().bind('<Button-4>',   lambda e: self._on_canvas_scroll(e, 2))
        self._canvas_knee.get_tk_widget().bind('<Button-5>',   lambda e: self._on_canvas_scroll(e, 2))

        # ankle graph (row 2, col 1)
        ankle_frame = tk.Frame(content, bg=BG2, bd=1, relief='flat')
        ankle_frame.grid(row=2, column=1, sticky='nsew', padx=(0, 2), pady=(3, 2))
        self._build_graph_limb_header(ankle_frame, 'left_ankle', 'right_ankle')
        self._fig_ankle, self._ax_ankle = plt.subplots(figsize=(5.5, 2.6), dpi=100)
        self._style_ax(self._ax_ankle, self._fig_ankle)
        self._canvas_ankle = FigureCanvasTkAgg(self._fig_ankle, master=ankle_frame)
        self._canvas_ankle.get_tk_widget().pack(fill='both', expand=True)
        self._canvas_ankle.mpl_connect('button_press_event',   lambda e: self._on_graph_click(e, 0))
        self._canvas_ankle.mpl_connect('motion_notify_event',  lambda e: self._on_graph_drag(e, 0))
        self._canvas_ankle.mpl_connect('button_release_event', lambda e: self._on_graph_release(e, 0))
        self._canvas_ankle.get_tk_widget().bind('<MouseWheel>', lambda e: self._on_canvas_scroll(e, 0))
        self._canvas_ankle.get_tk_widget().bind('<Button-4>',   lambda e: self._on_canvas_scroll(e, 0))
        self._canvas_ankle.get_tk_widget().bind('<Button-5>',   lambda e: self._on_canvas_scroll(e, 0))

        # videos panel (row 0, col 0), spanning all graph rows
        videos_frame = tk.Frame(content, bg=BG2, bd=1, relief='flat')
        videos_frame.grid(row=0, column=0, rowspan=3, sticky='nsew', padx=(0, 4), pady=(2, 2))
        videos_frame.grid_columnconfigure(0, weight=1)
        videos_frame.grid_rowconfigure(0, weight=1, uniform='video')
        videos_frame.grid_rowconfigure(1, weight=1, uniform='video')

        vid1_outer = tk.Frame(videos_frame, bg=BG2)
        vid1_outer.grid(row=0, column=0, sticky='nsew', padx=0, pady=(0, 2))
        self._vid1_lbl = tk.Label(vid1_outer, text="VIDEO 1",
                      font=("Helvetica", 8, "bold"), bg=BG2, fg=C_V1, anchor='w')
        self._vid1_lbl.pack(fill='x', padx=4, pady=(2, 0))
        self._vid_canvas1 = tk.Canvas(vid1_outer, bg=BG_VID, highlightthickness=0)
        self._vid_canvas1.pack(fill='both', expand=True)

        vid2_outer = tk.Frame(videos_frame, bg=BG2)
        vid2_outer.grid(row=1, column=0, sticky='nsew', padx=0, pady=(2, 0))
        self._vid2_lbl = tk.Label(vid2_outer, text="VIDEO 2",
                      font=("Helvetica", 8, "bold"), bg=BG2, fg=C_V2, anchor='w')
        self._vid2_lbl.pack(fill='x', padx=4, pady=(2, 0))
        self._vid_canvas2 = tk.Canvas(vid2_outer, bg=BG_VID, highlightthickness=0)
        self._vid_canvas2.pack(fill='both', expand=True)

        self._vid_canvases = [self._vid_canvas1, self._vid_canvas2]
        self._graph_canvas_widgets = [
            self._canvas_ankle.get_tk_widget(),
            self._canvas_hip.get_tk_widget(),
            self._canvas_knee.get_tk_widget(),
        ]
        for widget in self._graph_canvas_widgets:
            widget.bind('<Configure>', self._schedule_graph_resize_sync, add='+')
        self._update_graph_limb_btn_visuals()
        self.after_idle(self._sync_graph_canvas_sizes)

        # metrics panel
        right = tk.Frame(sidebar, bg=BG2)
        right.grid(row=1, column=0, sticky='nsew', pady=(3, 0))
        right.pack_propagate(False)
        self._build_metrics_panel(right)

        # status bar
        bottom = tk.Frame(self, bg=BG2, height=36)
        bottom.pack(fill='x', side='bottom')
        bottom.pack_propagate(False)
        self._bottom_bar = bottom

        self._prog_canvas = tk.Canvas(bottom, height=4, bg=BG3, highlightthickness=0)
        self._prog_canvas.pack(fill='x', side='top')

        bar = tk.Frame(bottom, bg=BG2)
        bar.pack(fill='x', expand=True, padx=6)

        btn_cfg = dict(bg=BG3, fg=TEXT, relief='flat',
                       font=("Helvetica", 8), padx=5, pady=1, cursor='hand2',
                       activebackground=ACCENT, activeforeground='white')
        for txt, cmd in [("Prev", self._prev_frame), ("Next", self._next_frame), ("Play", self._toggle_play)]:
            tk.Button(bar, text=txt, command=cmd, **btn_cfg).pack(side='left', padx=2, pady=3)

        self._frame_lbl = tk.Label(bar, text="Frame: —",
                                   font=("Helvetica", 8), bg=BG2, fg=SUBTEXT)
        self._frame_lbl.pack(side='right', padx=8)

        status_btn_cfg = dict(bg=BG3, fg=TEXT, relief='flat',
                      font=("Helvetica", 8), padx=4, pady=0, cursor='hand2',
                              activebackground=ACCENT, activeforeground='white')

        tk.Label(bar, textvariable=self._status_msg,
                 font=("Helvetica", 8), bg=BG2, fg=TEXT, anchor='w'
                 ).pack(side='left', padx=8)

    def _style_ax(self, ax, fig):
        """Apply common graph styling."""
        fig.patch.set_facecolor(BG2)
        ax.set_facecolor(BG_PLOT)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_color(BG2)
        ax.tick_params(colors=SUBTEXT, labelsize=7, length=0)
        ax.xaxis.label.set_color(SUBTEXT)
        ax.yaxis.label.set_color(SUBTEXT)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.08)

    def _build_graph_limb_header(self, parent, left_joint, right_joint):
        pass

    def _create_graph_limb_toggle(self, parent, joint_name, label_text):
        fg_col = JOINT_COLORS_MPL.get(joint_name, TEXT)
        state_on = self.joint_visibility.get(joint_name, True)

        frame = tk.Frame(
            parent,
            bg=BG3 if state_on else BG2,
            bd=0,
            relief='flat',
            cursor='hand2',
            highlightthickness=1,
            highlightbackground=BG2,
            highlightcolor=BG2,
        )

        swatch = tk.Canvas(frame, width=38, height=16, bg=frame['bg'], highlightthickness=0)
        swatch.pack(side='left', padx=(5, 3), pady=2)
        swatch.create_line(1, 8, 13, 8, fill=fg_col, dash=(4, 3), width=2)
        swatch.create_line(18, 8, 35, 8, fill=fg_col, width=2)

        label = tk.Label(
            frame,
            text=label_text,
            font=("Helvetica", 9, "bold"),
            bg=frame['bg'],
            fg=TEXT,
            padx=1,
            pady=2,
        )
        label.pack(side='left', padx=(0, 5))

        frame.bind('<Button-1>', lambda e, j=joint_name: self._toggle_graph_joint_visibility(j))
        swatch.bind('<Button-1>', lambda e, j=joint_name: self._toggle_graph_joint_visibility(j))
        label.bind('<Button-1>', lambda e, j=joint_name: self._toggle_graph_joint_visibility(j))

        self._graph_limb_btns[joint_name] = {'frame': frame, 'swatch': swatch, 'label': label, 'fg': fg_col}
        return self._graph_limb_btns[joint_name]

    def _toggle_graph_joint_visibility(self, joint_name):
        self.joint_visibility[joint_name] = not self.joint_visibility.get(joint_name, True)
        self._update_graph_limb_btn_visuals()
        self.redraw_graphs()
        self._show_video_frames()

    def _update_graph_limb_btn_visuals(self):
        for joint_name, parts in self._graph_limb_btns.items():
            fg_col = parts.get('fg', JOINT_COLORS_MPL.get(joint_name, TEXT))
            enabled = self.joint_visibility.get(joint_name, True)
            bg = BG3 if enabled else BG2
            parts['frame'].config(bg=bg)
            parts['swatch'].config(bg=bg)
            parts['label'].config(bg=bg, fg=TEXT)
            parts['frame'].lift()

    def _schedule_graph_resize_sync(self, _event=None):
        if self._graph_resize_after_id is not None:
            try:
                self.after_cancel(self._graph_resize_after_id)
            except Exception:
                pass
        self._graph_resize_after_id = self.after(40, self._sync_graph_canvas_sizes)

    def _sync_graph_canvas_sizes(self):
        self._graph_resize_after_id = None
        dims = []
        for widget in getattr(self, '_graph_canvas_widgets', []):
            w, h = widget.winfo_width(), widget.winfo_height()
            if w > 20 and h > 20:
                dims.append((w, h))
        if len(dims) < 3:
            return

        target_w = min(w for w, _ in dims)
        target_h = min(h for _, h in dims)
        if target_w < 20 or target_h < 20:
            return

        figs = [self._fig_ankle, self._fig_hip, self._fig_knee]
        for fig in figs:
            dpi = fig.get_dpi() or 100
            fig.set_size_inches(target_w / dpi, target_h / dpi, forward=True)
        for canvas in [self._canvas_ankle, self._canvas_hip, self._canvas_knee]:
            canvas.draw_idle()

    def _build_buttons_panel(self, parent):
        """Build the top-right buttons panel."""
        tk.Label(parent, text="DISPLAY", font=("Helvetica", 8, "bold"),
                 bg=BG2, fg=ACCENT).pack(pady=(10, 4), padx=6)

        btn_cfg = dict(bg=BG3, fg=TEXT, relief='flat', font=("Helvetica", 8, "bold"),
                   padx=5, pady=1, cursor='hand2', width=10,
                       activebackground=ACCENT, activeforeground='white')

        # v1 / v2 toggles
        ind_frame = tk.Frame(parent, bg=BG2)
        ind_frame.pack(fill='x', padx=6, pady=(2, 4))

        v1_f = tk.Frame(ind_frame, bg=BG3, cursor='hand2')
        v1_f.pack(side='left', fill='x', expand=True, padx=(0, 2))
        c1 = tk.Canvas(v1_f, width=20, height=12, bg=BG3, highlightthickness=0)
        c1.pack(side='left', padx=(3, 1), pady=2)
        c1.create_line(1, 6, 19, 6, fill=C_V1, dash=(4, 3), width=2)
        self._v1_toggle_lbl = tk.Label(v1_f, text="V1", font=("Helvetica", 8, "bold"),
                                        bg=BG3, fg=C_V1)
        self._v1_toggle_lbl.pack(side='left', padx=(1, 3))
        self._v1_btn_frame = v1_f
        self._v1_swatch = c1
        for w in (v1_f, c1, self._v1_toggle_lbl):
            w.bind('<Button-1>', lambda e: self._toggle_video_view(0))

        v2_f = tk.Frame(ind_frame, bg=BG3, cursor='hand2')
        v2_f.pack(side='left', fill='x', expand=True, padx=(2, 0))
        c2 = tk.Canvas(v2_f, width=20, height=12, bg=BG3, highlightthickness=0)
        c2.pack(side='left', padx=(3, 1), pady=2)
        c2.create_line(1, 6, 19, 6, fill=C_V2, width=2)
        self._v2_toggle_lbl = tk.Label(v2_f, text="V2", font=("Helvetica", 8, "bold"),
                                        bg=BG3, fg=C_V2)
        self._v2_toggle_lbl.pack(side='left', padx=(1, 3))
        self._v2_btn_frame = v2_f
        self._v2_swatch = c2
        for w in (v2_f, c2, self._v2_toggle_lbl):
            w.bind('<Button-1>', lambda e: self._toggle_video_view(1))

        tk.Frame(parent, bg=SUBTEXT, height=1).pack(fill='x', padx=6, pady=(4, 4))

        badge_styles = {
            'Ankle': {'bg': '#dbeeff', 'fg': '#1f4f7a'},
            'Hip':   {'bg': '#ffe7cc', 'fg': '#8a4b08'},
            'Knee':  {'bg': '#dff3df', 'fg': '#1f6b2f'},
        }
        for group_joints, group_label in [
            (['left_hip',   'right_hip'],   'Hip'),
            (['left_knee',  'right_knee'],  'Knee'),
            (['left_ankle', 'right_ankle'], 'Ankle'),
        ]:
            row = tk.Frame(parent, bg=BG2)
            row.pack(fill='x', padx=6, pady=1)
            style = badge_styles.get(group_label, {'bg': BG2, 'fg': SUBTEXT})
            badge = tk.Label(row, text=group_label, font=("Helvetica", 8, "bold"),
                             bg=style['bg'], fg=style['fg'], padx=4, pady=2, width=5)
            badge.pack(side='left', padx=(0, 4))
            for joint in group_joints:
                btn = self._create_graph_limb_toggle(
                    row,
                    joint,
                    ('Left' if 'left' in joint else 'Right'),
                )
                if 'left' in joint:
                    btn['frame'].pack(side='left', padx=(0, 4), pady=1)
                else:
                    btn['frame'].pack(side='left', padx=(0, 0), pady=1)

        tk.Frame(parent, bg=SUBTEXT, height=1).pack(fill='x', padx=6, pady=(4, 4))

        # section 1: cycles and world
        self._display_btns = {}
        
        # cycles button (text changes based on state)
        self._cycles_btn = tk.Button(parent, text="Cycles", **btn_cfg,
                           command=lambda: self._panel_btn_dispatch('Cycles'))
        self._cycles_btn.pack(fill='x', padx=6, pady=1)
        self._display_btns['cycles'] = self._cycles_btn
        
        # world button (text changes based on state)
        self._world_btn = tk.Button(parent, text="World", **btn_cfg,
                          command=lambda: self._panel_btn_dispatch('World'))
        self._world_btn.pack(fill='x', padx=6, pady=1)
        self._display_btns['world_px'] = self._world_btn

        tk.Frame(parent, bg=SUBTEXT, height=1).pack(fill='x', padx=6, pady=(4, 4))

        # section 2: mean, data, normal, outliers
        for label, key in [("Mean", "mean"), ("Data", "data"),
                   ("Normal", "normal"), ("Outliers", "outliers")]:
            btn = tk.Button(parent, text=label, **btn_cfg,
                           command=lambda k=label: self._panel_btn_dispatch(k))
            btn.pack(fill='x', padx=6, pady=1)
            self._display_btns[key] = btn

        tk.Frame(parent, bg=SUBTEXT, height=1).pack(fill='x', padx=6, pady=(4, 4))

        # section 3: clear steps and exclusion zones
        tk.Button(parent, text="Clear Steps", **btn_cfg,
              command=self._clear_steps).pack(fill='x', padx=6, pady=1)
        tk.Button(parent, text="Clear Excl. Zone", **btn_cfg,
              command=self._clear_exclusions).pack(fill='x', padx=6, pady=1)

        self._update_display_btn_visuals()

    def _build_metrics_panel(self, parent):
        """Build the bottom-right metrics panel."""
        tk.Label(parent, text="METRICS",
                 font=("Helvetica", 9, "bold"), bg=BG2, fg=ACCENT
                 ).pack(pady=(8, 4), padx=6, anchor='w')

        self._metrics_canvas = tk.Canvas(parent, bg=BG2, highlightthickness=0)
        self._metrics_canvas.pack(side='left', fill='both', expand=True)

        self._metrics_scrollable = tk.Frame(self._metrics_canvas, bg=BG2)
        self._metrics_window = self._metrics_canvas.create_window((0, 0), window=self._metrics_scrollable, anchor='nw')

        self._metrics_scrollable.bind('<Configure>', lambda e: self._metrics_canvas.configure(scrollregion=self._metrics_canvas.bbox('all')))
        self._metrics_canvas.bind('<Configure>', lambda e: self._metrics_canvas.itemconfigure(self._metrics_window, width=e.width))

        self._metric_value_lbls = {}
        for key in METRIC_ORDER:
            label, sub = METRIC_LABELS[key]
            card = tk.Frame(self._metrics_scrollable, bg=BG3, padx=6, pady=3)
            card.pack(fill='x', padx=6, pady=2)
            title_lbl = tk.Label(card, text=label, font=("Helvetica", 8, "bold"), bg=BG3, fg=TEXT)
            title_lbl.pack(anchor='w')
            sub_lbl = tk.Label(card, text=sub, font=("Helvetica", 6), bg=BG3, fg=SUBTEXT)
            sub_lbl.pack(anchor='w')
            val_lbl = tk.Label(card, text="—", font=("Helvetica", 11, "bold"), bg=BG3, fg=SUBTEXT)
            val_lbl.pack(anchor='w', pady=(1, 0))
            self._metric_value_lbls[key] = val_lbl
            for widget in (card, title_lbl, sub_lbl, val_lbl):
                self._bind_metrics_mousewheel_widget(widget)

    def _bind_metrics_mousewheel(self, event=None):
        self.bind_all('<MouseWheel>', self._on_metrics_mousewheel)
        self.bind_all('<Button-4>', self._on_metrics_mousewheel)
        self.bind_all('<Button-5>', self._on_metrics_mousewheel)

    def _unbind_metrics_mousewheel(self, event=None):
        self.unbind_all('<MouseWheel>')
        self.unbind_all('<Button-4>')
        self.unbind_all('<Button-5>')

    def _bind_metrics_mousewheel_widget(self, widget):
        widget.bind('<MouseWheel>', self._on_metrics_mousewheel)
        widget.bind('<Button-4>', self._on_metrics_mousewheel)
        widget.bind('<Button-5>', self._on_metrics_mousewheel)

    def _on_metrics_mousewheel(self, event):
        if not hasattr(self, '_metrics_canvas'):
            return
        if event.num == 4 or getattr(event, 'delta', 0) > 0:
            self._metrics_canvas.yview_scroll(-1, 'units')
        elif event.num == 5 or getattr(event, 'delta', 0) < 0:
            self._metrics_canvas.yview_scroll(1, 'units')
        return 'break'

    def _panel_btn_dispatch(self, label):
        dispatch = {
            "Cycles":  self._toggle_cycles,
            "Mean":    self._toggle_mean,
            "Data":    lambda: self._toggle_display_option('data'),
            "Normal":  lambda: self._toggle_display_option('normal'),
            "Outliers":lambda: self._toggle_display_option('outliers'),
            "World":   self._toggle_world,
        }
        fn = dispatch.get(label)
        if fn:
            fn()

    # video selection

    def find_videos(self):
        video_paths = list(select_video_paths())
        if len(video_paths) == 0:
            return
        if len(video_paths) == 1:
            second = filedialog.askopenfilename(
                title="Select second video", parent=self,
                filetypes=[("Video files", "*.mov *.mp4 *.avi *.m4v"), ("All files", "*.*")])
            if not second:
                messagebox.showwarning("Two videos required",
                                       "Please select two videos to run the analysis.", parent=self)
                return
            video_paths.append(second)
        video_paths = video_paths[:2]
        self.video_names = [os.path.splitext(os.path.basename(p))[0] for p in video_paths]
        self._vid1_lbl.config(text=f"VIDEO 1  —  {self.video_names[0]}")
        self._vid2_lbl.config(text=f"VIDEO 2  —  {self.video_names[1]}")
        self._status_msg.set("Checking pose model…")
        self.update()
        ensure_model()
        self._status_msg.set("Preparing videos…")
        session = self._session
        spill1 = os.path.join(session.path, "vid1")
        spill2 = os.path.join(session.path, "vid2")
        os.makedirs(spill1, exist_ok=True)
        os.makedirs(spill2, exist_ok=True)
        results          = [None, None]
        results_progress = [0.0, 0.0]
        pre_scan = [None, None]

        def _scan_video(i, path):
            try:
                needs_rot = _detect_subject_orientation(path)
                crop_rect = detect_crop_region(path, needs_rot)
                dims = _get_cropped_dimensions(path, needs_rot, crop_rect=crop_rect)
                pre_scan[i] = {
                    'needs_rotation': needs_rot, 'crop_rect': crop_rect, 'dims': dims,
                    'video_meta': _video_metadata(path), 'video_sha256': _file_sha256(path),
                }
            except Exception as e:
                pre_scan[i] = {'error': str(e)}

        scan_threads = [
            threading.Thread(target=_scan_video, args=(0, video_paths[0]), daemon=True),
            threading.Thread(target=_scan_video, args=(1, video_paths[1]), daemon=True),
        ]

        def _start_processing():
            target_output_size = None
            sizes = [s.get('dims') for s in pre_scan if isinstance(s, dict) and s.get('dims')]
            if len(sizes) == 2:
                target_w = max(sizes[0][0], sizes[1][0])
                target_h = max(sizes[0][1], sizes[1][1])
                target_output_size = (target_w, target_h)
                self._status_msg.set(f"Target output: {target_w}x{target_h}")
                self.update()

            cache_keys = [None, None]
            cache_meta = [None, None]
            for i in range(2):
                info = pre_scan[i] if isinstance(pre_scan[i], dict) else None
                if not info or info.get('error'):
                    continue
                cache_keys[i] = _build_cache_key(info, target_output_size)
                video_name = os.path.basename(video_paths[i]) if video_paths[i] else f"Video {i+1}"
                cache_meta[i] = {
                    'schema': CACHE_SCHEMA_VERSION, 'video_path': video_paths[i],
                    'video_name': video_name, 'video_meta': info.get('video_meta'),
                    'video_sha256': info.get('video_sha256'),
                    'target_output_size': list(target_output_size) if target_output_size else None,
                }

            self._status_msg.set("Processing videos… 0%")

            def _process(i, path, ann_dir):
                info = pre_scan[i] if isinstance(pre_scan[i], dict) else {}
                def _prog(p): results_progress[i] = p
                def _stat(s): self.after(0, lambda: self._status_msg.set(s))
                results[i] = process_video(path, ann_dir, _prog, _stat,
                    target_output_size=target_output_size,
                    needs_rotation=info.get('needs_rotation'),
                    crop_rect=info.get('crop_rect'),
                    cache_key=cache_keys[i], cache_meta=cache_meta[i])

            t1 = threading.Thread(target=_process, args=(0, video_paths[0], spill1), daemon=True)
            t2 = threading.Thread(target=_process, args=(1, video_paths[1], spill2), daemon=True)
            t1.start(); t2.start()
            self.after(300, _poll_loading)

        def _poll_pre_scan():
            done = sum(0 if t.is_alive() else 1 for t in scan_threads)
            self._status_msg.set(f"Preparing videos… {done}/2")
            self._update_status()
            if done < 2:
                self.after(200, _poll_pre_scan)
                return
            _start_processing()

        def _poll_loading():
            p = (results_progress[0] + results_progress[1]) / 2
            self.progress = p
            self._status_msg.set(f"Processing videos… {int(p*100)}%")
            self._update_status()
            if results[0] is None or results[1] is None:
                self.after(250, _poll_loading)
            else:
                _finish()

        def _finish():
            if results[0] is None or results[1] is None:
                messagebox.showerror("Error", "Failed to process one or both videos.")
                return
            
            # save results to cache if cache_key is present
            for i in range(2):
                if results[i] is not None:
                    cache_key = results[i].get('_cache_key')
                    cache_meta = results[i].get('_cache_meta')
                    if cache_key and cache_meta:
                        _save_cached_video_result(cache_key, cache_meta, results[i])
            
            self.datasets           = results
            _ca, _cb = _compute_auto_crops_pair(
               results[0].get('crop_stats'), results[1].get('crop_stats'),
                results[0].get('_fw_upright', 1), results[0].get('_fh_upright', 1),
                results[1].get('_fw_upright', 1), results[1].get('_fh_upright', 1),
                pad=0.05)
            results[0]['crop_rect'] = _ca
            results[1]['crop_rect'] = _cb
            self.df_world           = results[0]['df_world']
            self.df_pixel           = results[0]['df_pixel']
            self.df_pixel_filtered  = results[0].get('df_pixel_filtered', results[0]['df_pixel'])
            key = 'df_world' if USE_WORLD_LANDMARKS else 'df_pixel_filtered'
            for ds in self.datasets:
                if key in ds:
                    ds['angle_data'] = ds[key]
            self.angle_data         = self.datasets[0].get(key, self.datasets[0]['angle_data'])
            self.total_frames = max(len(results[0]['all_landmarks']), len(results[1]['all_landmarks']))
            min_video_frames = min(len(results[0]['all_landmarks']), len(results[1]['all_landmarks']))
            self.current_frame_idx = 0
            if not self.angle_data.empty:
                data_min = self.angle_data['frame_num'].min()
                full = (data_min, data_min + min_video_frames)
                self._ax_xlim_full = [full, full, full]
                for gi in range(3):
                    self._ax_xlim_per_mode[gi]['both'] = full
                    for idx, ds in enumerate(results[:2]):
                        v_frames = len(ds['all_landmarks'])
                        self._ax_xlim_per_mode[gi][f'v{idx+1}'] = (data_min, data_min + v_frames)
            self.progress = 1.0
            self.show_overlaid_cycles = False
            self.resample_cycles = False
            cached_loaded = self._resolve_cached_markup()
            auto_excl_count = self._auto_exclude_bad_regions(overwrite=False)
            if self._start_required_markup_flow():
                return
            self.show_overlaid_cycles = True
            self.resample_cycles = True
            self._update_display_btn_visuals()
            self.refresh()
            # force a second pass once tk has finalized widget sizes
            self.after_idle(self._show_video_frames)
            self.after(60, self._show_video_frames)
            if cached_loaded:
                self._status_msg.set("Loaded cached steps")
            elif auto_excl_count:
                self._status_msg.set(f"Loaded analysis with {auto_excl_count} auto exclusion zones")
            else:
                self._status_msg.set("Loaded analysis")

        for t in scan_threads:
            t.start()
        self.after(200, _poll_pre_scan)

    # key bindings

    def _bind_keys(self):
        for canvas in [self._canvas_ankle.get_tk_widget(),
                       self._canvas_hip.get_tk_widget(),
                       self._canvas_knee.get_tk_widget()]:
            canvas.bind('<MouseWheel>', lambda e, gi=0: self._on_canvas_scroll(e, gi))

        shortcuts = {
            '<Key-1>': self._prev_frame,
            '<Key-2>': self._next_frame,
            '<Key-9>': self._toggle_play,
            '<q>': self._on_close,
            '<w>': self._toggle_world,
            '<c>': self._toggle_cycles,
            '<x>': self._toggle_cycle_alignment,
            '<b>': self._toggle_cycle_stacked,
            '<s>': self._toggle_resample,
            '<m>': self._toggle_mean,
            '<v>': self._cycle_graph_view,
            '<t>': self._toggle_active,
            '<g>': self._toggle_suggestions,
            '<d>': self._clear_steps,
            '<space>': self._add_manual_step,
            '<BackSpace>': self._delete_nearest_step,
            '<Delete>': self._delete_nearest_step,
            'z': self._reset_zoom,
            '<Alt-c>': self._toggle_confidence,
            '<F3>': self._toggle_ankle_norm_offset,
        }

        for seq, fn in shortcuts.items():
            self.bind_all(seq, lambda e, _fn=fn: _fn())

        # support caps-lock/shifted letters for single-key shortcuts
        for seq, fn in shortcuts.items():
            if len(seq) == 3 and seq.startswith('<') and seq.endswith('>') and seq[1].isalpha():
                upper_seq = f"<{seq[1].upper()}>"
                self.bind_all(upper_seq, lambda e, _fn=fn: _fn())

    # prefetch worker for frame caching

    def _prefetch_worker(self):
        last = -1
        while not self._stop_pf:
            idx = self.current_frame_idx
            if idx != last:
                for vi, ds in enumerate(self.datasets[:2]):
                    store = ds.get('all_landmarks', [])
                    for i in range(max(0, idx-16), min(len(store), idx+17)):
                        self._cache.get(vi, i, store)
                last = idx
            time.sleep(0.02)

    # dataset helpers

    def _active_ds(self):
        if not self.datasets: return None
        return self.datasets[min(self.active_dataset_idx, len(self.datasets)-1)]

    def _active_angle_data(self):
        ds = self._active_ds()
        return ds['angle_data'] if ds else self.angle_data

    def _active_max_index(self):
        ad = self._active_angle_data()
        return len(ad)-1 if (ad is not None and not ad.empty) else -1

    def _persist_dataset_markup(self, ds):
        if not ds: return
        cache_key = ds.get('_cache_key')
        if cache_key:
            _save_cached_markup(cache_key, ds.get('step_frames', []))

    def _persist_all_dataset_markup(self):
        for ds in self.datasets:
            self._persist_dataset_markup(ds)

    def _dataset_needs_markup(self, video_idx):
        if not hasattr(self, '_markup_required_videos'):
            return True
        if video_idx >= len(self._markup_required_videos):
            return True
        return self._markup_required_videos[video_idx]

    def _next_markup_phase(self, side, video_idx):
        order = [('left', 0), ('left', 1), ('right', 0), ('right', 1)]
        try:
            start_idx = order.index((side, video_idx)) + 1
        except ValueError:
            start_idx = 0
        for next_side, next_video_idx in order[start_idx:]:
            if self._dataset_needs_markup(next_video_idx):
                return next_side, next_video_idx
        return None

    def _resolve_cached_markup(self):
        self._markup_required_videos = [True] * len(self.datasets)
        cached_loaded = False
        for i, ds in enumerate(self.datasets):
            if not ds: continue
            cached_markup = ds.get('_cached_markup') or {}
            cached_steps = cached_markup.get('step_frames', [])
            if not cached_steps: continue
            vid_name = self.video_names[i] if i < len(self.video_names) else f"Video {i+1}"
            answer = messagebox.askyesno("Cached steps found",
                f"Cached manual steps were found for {vid_name}.\n\n"
                "Yes = load cached steps\nNo = overwrite and mark again", parent=self)
            if answer:
                ds['step_frames'] = list(cached_steps)
                self._markup_required_videos[i] = False
                cached_loaded = True
            else:
                ds['step_frames'] = []
                self._markup_required_videos[i] = True
                cache_key = ds.get('_cache_key')
                if cache_key:
                    _clear_cached_markup(cache_key)
        return cached_loaded

    def _start_required_markup_flow(self):
        order = [('left', 0), ('left', 1), ('right', 0), ('right', 1)]
        for side, video_idx in order:
            if self._dataset_needs_markup(video_idx):
                self._enter_marking_phase(side, video_idx)
                return True
        return False

    def _auto_exclude_bad_regions(self, overwrite=False):
        if not self.datasets: return 0
        total = 0
        for ds in self.datasets:
            if not ds: continue
            if not overwrite and ds.get('excluded_regions'):
                total += len(ds.get('excluded_regions', []))
                continue
            ad = ds.get('angle_data', pd.DataFrame())
            depths = ds.get('landmark_depths', pd.DataFrame())
            _, auto_excl, _ = detect_steps_robust(ad, depth_df=depths, fps=SLOWMO_FPS)
            ds['excluded_regions'] = self._merge_exclusion_regions(list(auto_excl))
            ds['suggested_step_frames'] = []
            ds['suggested_step_meta'] = []
            total += len(ds.get('excluded_regions', []))
        return total

    def _get_filtered_angle_data(self, angle_data, excluded_regions):
        if not excluded_regions or angle_data.empty:
            return angle_data
        mask = pd.Series([True] * len(angle_data), index=angle_data.index)
        for start_frame, end_frame in excluded_regions:
            mask &= ~((angle_data['frame_num'] >= start_frame) & (angle_data['frame_num'] < end_frame))
        return angle_data[mask].reset_index(drop=True)

    def _region_crosses_exclusion(self, start_frame, end_frame, excluded_regions):
        for ex_start, ex_end in excluded_regions:
            if not (end_frame <= ex_start or start_frame >= ex_end):
                return True
        return False

    @staticmethod
    def _merge_exclusion_regions(regions):
        if len(regions) <= 1: return regions
        regions.sort()
        merged = [regions[0]]
        for s, e in regions[1:]:
            ps, pe = merged[-1]
            if s <= pe:
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))
        return merged

    # graph drawing and display logic

    # map graph index to joint names
    _GRAPH_JOINTS = [
        ['left_ankle', 'right_ankle'],   # 0 = ankle
        ['left_hip',   'right_hip'],     # 1 = hip
        ['left_knee',  'right_knee'],    # 2 = knee
    ]
    _GRAPH_NORM_KEY = ['ankle', 'hip', 'knee']
    _GRAPH_AXES    = None   # set in redraw_graphs

    def _get_graph_axes(self):
        return [self._ax_ankle, self._ax_hip, self._ax_knee]

    def _get_graph_canvases(self):
        return [self._canvas_ankle, self._canvas_hip, self._canvas_knee]

    def redraw_graphs(self):
        """redraw all three joint graphs."""
        axes    = self._get_graph_axes()
        canvases = self._get_graph_canvases()
        for gi in range(3):
            self._redraw_single_graph(gi, axes[gi], canvases[gi])
        self._update_graph_limb_btn_visuals()

    def _redraw_single_graph(self, gi, ax, mpl_canvas):
        """draw one of the three joint graphs (gi=0 ankle, 1 hip, 2 knee)."""
        ax.cla()
        ax.set_facecolor(BG_PLOT)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_color(BG2)
        ax.tick_params(colors=SUBTEXT, labelsize=7, length=0)
        ax.tick_params(axis='y', labelleft=False)
        ax.xaxis.label.set_color(SUBTEXT)
        ax.set_ylabel('')

        joint_names = self._GRAPH_JOINTS[gi]
        norm_key    = self._GRAPH_NORM_KEY[gi]

        if self.angle_data is None or self.angle_data.empty:
            ax.set_xlabel('')
            ax.tick_params(axis='x', labelbottom=False)
            ax.tick_params(axis='y', labelleft=False)
            mpl_canvas.draw_idle()
            return

        linestyles = ['--', '-']
        datasets   = self.datasets or [{'angle_data': self.angle_data, 'step_frames': []}]
        if len(datasets) >= 2:
            if   self.graph_show_mode == 'v1': dfg = [datasets[0]]
            elif self.graph_show_mode == 'v2': dfg = [datasets[1]]
            else:                              dfg = datasets[:2]
        else:
            dfg = datasets[:1]

        def _src_idx(ds):
            if len(self.datasets) >= 2:
                if ds is self.datasets[0]: return 0
                if ds is self.datasets[1]: return 1
            return 0

        def _to_fnums(ad, sf):
            if not sf: return []
            fn     = ad['frame_num'].to_numpy(dtype=int)
            fn_set = set(fn.tolist())
            vals   = [int(v) for v, _ in sf]
            if sum(1 for v in vals if v in fn_set) >= max(1, len(vals)//2):
                mapped = [(int(v), s) for v, s in sf if int(v) in fn_set]
            else:
                mapped = [(int(fn[int(v)]), s) for v, s in sf if 0 <= int(v) < len(fn)]
            mapped.sort(key=lambda x: x[0])
            return mapped

        def _get_cycle_strikes_for_joint(norm_steps, joint_name):
            side = 'left' if joint_name.startswith('left_') else 'right'
            return [f for f, s in norm_steps if s == side]

        def _cycle_plot_transform(values):
            if not self.align_cycle_angles: return values
            y = np.asarray(values, dtype=float)
            if y.size == 0: return y
            finite = np.isfinite(y)
            if not np.any(finite): return y
            return y - y[np.argmax(finite)]

        max_cycle_length = self.resample_length
        if self.show_overlaid_cycles:
            for ds in dfg:
                ad = ds['angle_data']
                excluded = ds.get('excluded_regions', [])
                ad_f = self._get_filtered_angle_data(ad, excluded)
                sf = ds.get('step_frames', [])
                norm = _to_fnums(ad_f, sf)
                for jn in joint_names:
                    strikes = _get_cycle_strikes_for_joint(norm, jn)
                    if len(strikes) < 2: continue
                    for i in range(len(strikes)-1):
                        if self._region_crosses_exclusion(strikes[i], strikes[i+1], excluded): continue
                        seg = ad_f[(ad_f['frame_num'] >= strikes[i]) & (ad_f['frame_num'] <= strikes[i+1])]
                        if not seg.empty:
                            max_cycle_length = max(max_cycle_length, len(seg))

        if not self.show_overlaid_cycles:
            # continuous mode
            ax.set_xlabel('Frame', fontsize=7)
            ax.tick_params(axis='x', labelbottom=True)
            ax.set_ylabel('')
            continuous_y_vals = []

            for ds in dfg:
                ad  = ds['angle_data']
                excluded = ds.get('excluded_regions', [])
                ad_f = self._get_filtered_angle_data(ad, excluded)
                sf  = ds.get('step_frames', [])
                si  = _src_idx(ds)
                ls  = linestyles[si % 2]
                frames = ad['frame_num'].values
                excl_mask = np.zeros(len(frames), dtype=bool)
                for ex_s, ex_e in excluded:
                    excl_mask |= (frames >= ex_s) & (frames < ex_e)

                show_excluded = getattr(self, '_show_excluded_in_pdf', True)
                if show_excluded:
                    for s, e in excluded:
                        ax.axvspan(s, e, alpha=0.10, color='darkgray', zorder=1)

                for joint in joint_names:
                    col = JOINT_COLORS_MPL.get(joint, '#888888')
                    if not self.joint_visibility.get(joint, True):
                        continue
                    if joint not in ad.columns: continue
                    values = ad[joint].values.copy().astype(float)
                    finite_vals = values[np.isfinite(values)]
                    if finite_vals.size:
                        continuous_y_vals.append(finite_vals)
                    if excluded and show_excluded:
                        gray_vals = values.copy()
                        gray_vals[~excl_mask] = np.nan
                        ax.plot(frames, gray_vals, color='#999999', lw=1.0, alpha=0.5, linestyle=ls, zorder=2)
                    clean_vals = values.copy()
                    if excluded and show_excluded:
                        clean_vals[excl_mask] = np.nan
                    ax.plot(frames, clean_vals, color=col, lw=1.3, alpha=0.85, linestyle=ls, zorder=3)

                nsteps  = _to_fnums(ad_f, sf)
                fn_min  = int(ad['frame_num'].min())
                fn_max  = int(ad['frame_num'].max())
                for f, side in nsteps:
                    if fn_min <= f <= fn_max:
                        ax.axvline(f, color=C_RIGHT if side=='right' else C_LEFT,
                                   lw=0.7, alpha=0.6, linestyle=ls)

            ref_ds = self._active_ds()
            ref_ad = ref_ds['angle_data'] if ref_ds else self.angle_data
            if ref_ad is not None and not ref_ad.empty and self.current_frame_idx < len(ref_ad):
                cf = ref_ad['frame_num'].iloc[self.current_frame_idx]
                ax.axvline(cf, color=C_CURSOR, lw=1.5, linestyle='--', zorder=10)

            if continuous_y_vals:
                merged_y = np.concatenate(continuous_y_vals)
                y_min, y_max = float(np.nanmin(merged_y)), float(np.nanmax(merged_y))
                if np.isfinite(y_min) and np.isfinite(y_max):
                    y_pad = 0.08 * (y_max - y_min) if y_max > y_min else max(2.0, abs(y_max) * 0.08)
                    ax.set_ylim(y_min - y_pad, y_max + y_pad)

            if self._ax_xlim_full[gi] is not None:
                ax.set_xlim(self._ax_xlim_full[gi])

        else:
            # overlaid cycles mode
            ax.set_xlabel('Frames Since Strike', fontsize=7)
            ax.tick_params(axis='x', labelbottom=True)
            ax.set_ylabel('')

            for ds in dfg:
                ad   = ds['angle_data']
                excluded = ds.get('excluded_regions', [])
                ad_f = self._get_filtered_angle_data(ad, excluded)
                sf   = ds.get('step_frames', [])
                si   = _src_idx(ds)
                ls   = linestyles[si % 2]
                norm = _to_fnums(ad_f, sf)

                for joint in joint_names:
                    col = JOINT_COLORS_MPL.get(joint, '#888888')
                    if not self.joint_visibility.get(joint, True):
                        continue
                    strikes = _get_cycle_strikes_for_joint(norm, joint)
                    if len(strikes) < 2: continue

                    cycles, lengths = [], []
                    for i in range(len(strikes)-1):
                        if self._region_crosses_exclusion(strikes[i], strikes[i+1], excluded): continue
                        seg = ad_f[(ad_f['frame_num'] >= strikes[i]) & (ad_f['frame_num'] <= strikes[i+1])]
                        if seg.empty: continue
                        x = seg['frame_num'].values - strikes[i]
                        y = seg[joint].values
                        cycles.append((x, y))
                        lengths.append(len(x))

                    if not cycles: continue
                    med = np.median(lengths)
                    length_ok = [0.8*med <= l <= 1.2*med for l in lengths]

                    if self.resample_cycles:
                        inliers_pre = []
                        for (x, y), good in zip(cycles, length_ok):
                            if not good: continue
                            t = np.linspace(0, 1, len(y))
                            inliers_pre.append(interp1d(t, y)(np.linspace(0, 1, max_cycle_length)))
                        mean_c = np.nanmean(np.vstack(inliers_pre), axis=0) if inliers_pre else None
                    else:
                        mean_c = None

                    ok = []
                    for idx2, ((x, y), length_good) in enumerate(zip(cycles, length_ok)):
                        if not length_good:
                            ok.append(False); continue
                        if mean_c is not None and self.resample_cycles:
                            rmse = _compute_cycle_rmse(y, mean_c, max_cycle_length)
                            ok.append(rmse <= self.rmse_threshold)
                        else:
                            ok.append(True)

                    if self.show_data:
                        if self.resample_cycles:
                            for (x, y), length_good, rmse_good in zip(cycles, length_ok, ok):
                                if not length_good: continue
                                t = np.linspace(0, 1, len(y))
                                y_plot = interp1d(t, y)(np.linspace(0, 1, max_cycle_length))
                                y_plot = _cycle_plot_transform(y_plot)
                                plot_col = C_OUTLIER if not rmse_good else col
                                ax.plot(np.arange(max_cycle_length), y_plot,
                                        color=plot_col, alpha=0.2 if not rmse_good else 0.25,
                                        lw=0.6, linestyle=ls)
                        else:
                            for (x, y), length_good in zip(cycles, length_ok):
                                plot_col = C_OUTLIER if not length_good else col
                                ax.plot(x, _cycle_plot_transform(y),
                                        color=plot_col, alpha=0.2 if not length_good else 0.25, lw=0.8, linestyle=ls)

                    if self.resample_cycles and self.show_mean:
                        inliers = []
                        for (x, y), good in zip(cycles, ok):
                            if not good: continue
                            t = np.linspace(0, 1, len(y))
                            inliers.append(interp1d(t, y)(np.linspace(0, 1, max_cycle_length)))
                        if inliers:
                            arr = np.vstack(inliers)
                            mean_c2 = np.nanmean(arr, axis=0)
                            se_c = np.nanstd(arr, axis=0) / np.sqrt(len(inliers))
                            if self.align_cycle_angles and len(mean_c2):
                                start = mean_c2[0]
                                mean_c2 -= start; se_c  # se stays relative
                            x_plot = np.arange(len(mean_c2))
                            if self.graph_display_mode == 'se_shading':
                                ax.fill_between(x_plot, mean_c2-se_c, mean_c2+se_c, color=col, alpha=0.15, zorder=2)
                            ax.plot(x_plot, mean_c2, color=col, lw=2.0, linestyle=ls, zorder=3)

            if self.resample_cycles and self.show_normative:
                norm_x = np.linspace(0, max_cycle_length, 100)
                d = NORMATIVE_GAIT[norm_key]
                norm_mean = np.asarray(d['mean'])
                if self.align_cycle_angles and len(norm_mean):
                    norm_mean = norm_mean - norm_mean[0]
                ax.plot(norm_x, norm_mean, color=C_NORM, lw=1.4, linestyle='-', alpha=0.8, zorder=2)

            ax.set_xlim(0, max_cycle_length)

        fig = [self._fig_ankle, self._fig_hip, self._fig_knee][gi]
        # tighter margins so plots fill their containers
        bottom_pad = 0.11
        fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=bottom_pad)
        mpl_canvas.draw_idle()

    # convenience alias kept for markup screen
    def redraw_graph(self):
        self.redraw_graphs()

    # video frame display and overlay drawing

    def _show_video_frames(self):
            
        for vi, canvas in enumerate(self._vid_canvases):
            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
            if cw < 2 or ch < 2: continue

            frame = None
            pixel_lm = None
            if vi < len(self.datasets):
                store = self.datasets[vi].get('all_landmarks', [])
                entry = None
                if 0 <= self.current_frame_idx < len(store):
                    entry = store[self.current_frame_idx]
                if isinstance(entry, tuple):
                    frame = self._cache.get(vi, self.current_frame_idx, store)
                    pixel_lm = entry[1]
                else:
                    frame = self._cache.get(vi, self.current_frame_idx, store)

            canvas.delete('all')
            if frame is None:
                canvas.create_text(cw//2, ch//2, text="No frame", fill=SUBTEXT, font=("Helvetica", 10))
                continue

            # apply skeleton-based auto-crop
            crop = self.datasets[vi].get('crop_rect') if vi < len(self.datasets) else None
            if crop is not None and self.datasets[vi].get('needs_rotation'):
                # crop is in upright space, frame is sideways - convert back
                # inverse of 90° cw: x_side = y_up, y_side = 1 - x_up - w_up
                x0, y0, cw_n, ch_n = crop
                crop_sideways = (y0, 1.0 - x0 - cw_n, ch_n, cw_n)
                frame = _apply_crop_rect(frame.copy(), crop_sideways)
                if pixel_lm is not None:
                    sx0, sy0, scw, sch = crop_sideways
                    pixel_lm = [SimpleLandmark(
                                    (lm.x - sx0) / scw if scw > 0 else lm.x,
                                    (lm.y - sy0) / sch if sch > 0 else lm.y,
                                    lm.visibility)
                                for lm in pixel_lm]
            elif crop is not None:
                frame = _apply_crop_rect(frame.copy(), crop)
                if pixel_lm is not None:
                    x0, y0, cw_n, ch_n = crop
                    pixel_lm = [SimpleLandmark(
                                    (lm.x - x0) / cw_n if cw_n > 0 else lm.x,
                                    (lm.y - y0) / ch_n if ch_n > 0 else lm.y,
                                    lm.visibility)
                                for lm in pixel_lm]

            jittery_frames = self.datasets[vi].get('jittery_frames', set()) if self.remove_jitter_frames else set()
            is_jittery = self.current_frame_idx in jittery_frames
            if pixel_lm is not None:
                if not (is_jittery and not self.show_jitter_frames):
                    frame = frame.copy()
                    draw_pose_landmarks_on_frame(frame, pixel_lm, self.joint_visibility,
                                                 skeleton_thickness=self.skeleton_thickness,
                                                 draw_jitter_red=is_jittery and self.show_jitter_frames)

            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            show_v1 = self.graph_show_mode in ('v1', 'both')
            show_v2 = self.graph_show_mode in ('v2', 'both')
            is_inactive = (vi == 0 and not show_v1) or (vi == 1 and not show_v2)
            if is_inactive:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rgb  = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

            fh, fw = rgb.shape[:2]
            scale  = min(cw/fw, ch/fh)
            nw, nh = int(fw*scale), int(fh*scale)
            rgb    = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            img    = ImageTk.PhotoImage(Image.fromarray(rgb))
            canvas._img = img
            canvas.create_image((cw-nw)//2, (ch-nh)//2, anchor='nw', image=img)

    # refresh display and graph rendering

    def refresh(self):
        self.redraw_graphs()
        self._show_video_frames()
        self._update_metrics_panel()
        self._update_status()

    def _update_status(self):
        ad = self._active_angle_data()
        n  = len(ad) if (ad is not None and not ad.empty) else 0
        t  = self.current_frame_idx / SLOWMO_FPS
        self._frame_lbl.config(text=f"Frame {self.current_frame_idx+1}/{n or '?'}  ({t:.2f} s)")
        self._prog_canvas.delete('all')
        W = self._prog_canvas.winfo_width()
        if W > 1 and n > 0:
            pw = int(W * self.progress)
            self._prog_canvas.create_rectangle(0, 0, pw, 4, fill=ACCENT, outline='')

    def _update_metrics_panel(self):
        if len(self.datasets) < 2:
            for lbl in self._metric_value_lbls.values():
                lbl.config(text="—", fg=SUBTEXT)
            return
        metrics = compute_metrics(self.datasets[0], self.datasets[1])
        for key, val in metrics.items():
            lbl = self._metric_value_lbls[key]
            hib = METRIC_HIB[key]
            arrow = "▲" if val > 0.5 else ("▼" if val < -0.5 else "●")
            txt   = f"{arrow} {val:+.1f}%"
            if hib is None: fg = TEXT
            elif abs(val) < 1.0: fg = SUBTEXT
            elif hib: fg = GREEN if val > 0 else RED
            else:     fg = GREEN if val < 0 else RED
            lbl.config(text=txt, fg=fg)

    # graph mouse interaction (per-graph index gi)

    def _seek_from_event(self, event, gi):
        ad = self._active_angle_data()
        if ad is None or ad.empty or event.xdata is None: return
        fn  = ad['frame_num'].to_numpy()
        idx = int(np.argmin(np.abs(fn - event.xdata)))
        self.current_frame_idx = max(0, min(idx, len(ad)-1))
        self._show_video_frames()
        self._update_status()
        self.redraw_graphs()

    def _on_graph_click(self, event, gi):
        axes = self._get_graph_axes()
        if event.button == 1 and event.inaxes == axes[gi]:
            self._graph_dragging[gi] = True
            self._seek_from_event(event, gi)
        elif event.button == 3 and event.inaxes == axes[gi]:
            self._exclusion_selecting[gi] = True
            self._exclusion_start[gi] = event.xdata

    def _on_graph_drag(self, event, gi):
        axes = self._get_graph_axes()
        if self._graph_dragging[gi] and event.inaxes == axes[gi]:
            self._seek_from_event(event, gi)

    def _on_graph_release(self, event, gi):
        if event.button == 1:
            self._graph_dragging[gi] = False
        elif event.button == 3:
            if self._exclusion_selecting[gi] and self._exclusion_start[gi] is not None and event.xdata is not None:
                start_frame = round(self._exclusion_start[gi])
                end_frame   = round(event.xdata)
                if start_frame > end_frame:
                    start_frame, end_frame = end_frame, start_frame
                if end_frame > start_frame:
                    target_datasets = []
                    if self.graph_show_mode == 'v1' and len(self.datasets) >= 1:
                        target_datasets = [self.datasets[0]]
                    elif self.graph_show_mode == 'v2' and len(self.datasets) >= 2:
                        target_datasets = [self.datasets[1]]
                    elif self.graph_show_mode == 'both' and len(self.datasets) >= 2:
                        target_datasets = [self.datasets[0], self.datasets[1]]
                    else:
                        target_datasets = [self._active_ds()]
                    for target_ds in target_datasets:
                        if target_ds:
                            target_ds.setdefault('excluded_regions', []).append((start_frame, end_frame))
                            target_ds['excluded_regions'] = self._merge_exclusion_regions(target_ds['excluded_regions'])
                            self._persist_dataset_markup(target_ds)
                    self._status_msg.set(f"Excluded frames {start_frame}-{end_frame}")
                    self.redraw_graphs()
            self._exclusion_selecting[gi] = False
            self._exclusion_start[gi] = None

    def _on_canvas_scroll(self, event, gi):
        if self.angle_data is None or self.angle_data.empty: return
        if self.show_overlaid_cycles: return
        if hasattr(event, 'delta'):
            scroll_dir = 1 if event.delta > 0 else -1
        elif hasattr(event, 'num'):
            scroll_dir = 1 if event.num == 4 else -1
        else:
            return
        ctrl_held = bool(event.state & 0x0004)
        if ctrl_held:
            self._on_graph_zoom(scroll_dir, gi)
        else:
            self._on_graph_pan(scroll_dir, gi)

    def _get_ax_for(self, gi):
        return self._get_graph_axes()[gi]

    def _get_canvas_for(self, gi):
        return self._get_graph_canvases()[gi]

    def _on_graph_pan(self, direction, gi):
        ax = self._get_ax_for(gi)
        cur_xlim = ax.get_xlim()
        full = self._ax_xlim_full[gi]
        if not full: return
        full_min, full_max = full
        pan_amount = (full_max - full_min) * 0.1 * direction
        new_min = max(full_min, cur_xlim[0] - pan_amount)
        new_max = min(full_max, cur_xlim[1] - pan_amount)
        if new_max - new_min < (cur_xlim[1] - cur_xlim[0]) * 0.99:
            if direction < 0: new_max = new_min + (cur_xlim[1] - cur_xlim[0])
            else:              new_min = new_max - (cur_xlim[1] - cur_xlim[0])
        ax.set_xlim(new_min, new_max)
        self._get_canvas_for(gi).draw_idle()

    def _on_graph_zoom(self, direction, gi):
        ax = self._get_ax_for(gi)
        cur_xlim = ax.get_xlim()
        full = self._ax_xlim_full[gi]
        if not full: return
        full_min, full_max = full
        full_range = full_max - full_min
        zoom_factor = 0.8 if direction < 0 else 1.2
        new_width = (cur_xlim[1] - cur_xlim[0]) * zoom_factor
        zoom_center = (cur_xlim[0] + cur_xlim[1]) / 2
        if new_width >= full_range:
            ax.set_xlim(full_min, full_max)
        else:
            rel_pos = (zoom_center - cur_xlim[0]) / max(cur_xlim[1] - cur_xlim[0], 1)
            new_min = zoom_center - new_width * rel_pos
            new_max = new_min + new_width
            if new_min < full_min: new_min = full_min; new_max = full_min + new_width
            if new_max > full_max: new_max = full_max; new_min = full_max - new_width
            ax.set_xlim(new_min, new_max)
        self._get_canvas_for(gi).draw_idle()

    def _reset_zoom(self):
        axes = self._get_graph_axes()
        canvases = self._get_graph_canvases()
        for gi in range(3):
            if self._ax_xlim_full[gi]:
                axes[gi].set_xlim(self._ax_xlim_full[gi])
                canvases[gi].draw_idle()
        self._status_msg.set("Zoom reset")

    # playback control

    def _play_tick(self):
        if not self.playing: return
        if self.current_frame_idx < self._active_max_index():
            self.current_frame_idx += 1
            self._show_video_frames()
            self.redraw_graphs()
            self._update_status()
            self._play_after_id = self.after(16, self._play_tick)
        else:
            self.playing = False
            self._status_msg.set("Playback finished")

    # ui controls and event handlers

    def _prev_frame(self):
        if self.playing: return
        self.current_frame_idx = max(0, self.current_frame_idx - 1)
        if self._marking_phase:
            t = self.current_frame_idx / SLOWMO_FPS
            self._markup_frame_lbl.config(text=f"Frame {self.current_frame_idx + 1}  ({t:.2f} s)")
            self._markup_show_frames()
            self._redraw_markup_graph()
            return
        self._show_video_frames()
        self.redraw_graphs()
        self._update_status()

    def _next_frame(self):
        if self.playing: return
        if self._marking_phase:
            vi = self._marking_video_idx
            max_idx = (len(self.datasets[vi].get('all_landmarks', [])) - 1
                       if vi < len(self.datasets) else self._active_max_index())
            self.current_frame_idx = min(max_idx, self.current_frame_idx + 1)
            t = self.current_frame_idx / SLOWMO_FPS
            self._markup_frame_lbl.config(text=f"Frame {self.current_frame_idx + 1}  ({t:.2f} s)")
            self._markup_show_frames()
            self._redraw_markup_graph()
            return
        self.current_frame_idx = min(self._active_max_index(), self.current_frame_idx + 1)
        self._show_video_frames()
        self.redraw_graphs()
        self._update_status()

    def _toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self._status_msg.set("Playing")
            self._play_tick()
        else:
            self._status_msg.set("Paused")
            if self._play_after_id:
                self.after_cancel(self._play_after_id)

    def _toggle_cycles(self):
        self.show_overlaid_cycles = not self.show_overlaid_cycles
        if self.show_overlaid_cycles:
            self.resample_cycles = True
        self._status_msg.set("Overlaid cycles" if self.show_overlaid_cycles else "Continuous view")
        self._update_display_btn_visuals()
        self.redraw_graphs()

    def _toggle_resample(self):
        if not self.show_overlaid_cycles: return
        self.resample_cycles = not self.resample_cycles
        self.redraw_graphs()

    def _toggle_cycle_alignment(self):
        if not self.show_overlaid_cycles: return
        self.align_cycle_angles = not self.align_cycle_angles
        self._update_display_btn_visuals()
        self.redraw_graphs()

    def _toggle_cycle_stacked(self):
        if not self.show_overlaid_cycles: return
        self.stack_cycle_lanes = not self.stack_cycle_lanes
        self._update_display_btn_visuals()
        self.redraw_graphs()

    def _toggle_mean(self):
        if not self.show_overlaid_cycles: return
        self.show_mean = not self.show_mean
        if not self.show_mean: self.mean_only = False
        self._update_display_btn_visuals()
        self.redraw_graphs()

    def _toggle_confidence(self):
        self.show_confidence = not self.show_confidence
        self.redraw_graphs()

    def _toggle_world(self):
        global USE_WORLD_LANDMARKS
        USE_WORLD_LANDMARKS = not USE_WORLD_LANDMARKS
        key = 'df_world' if USE_WORLD_LANDMARKS else 'df_pixel_filtered'
        for ds in self.datasets:
            if key in ds: ds['angle_data'] = ds[key]
        self.angle_data = self.df_world if USE_WORLD_LANDMARKS else self.df_pixel_filtered
        self._status_msg.set("World landmarks" if USE_WORLD_LANDMARKS else "Pixel landmarks")
        self._update_display_btn_visuals()
        self.redraw_graphs()

    def _toggle_video_view(self, which):
        show_v1 = self.graph_show_mode in ('v1', 'both')
        show_v2 = self.graph_show_mode in ('v2', 'both')
        if which == 0: show_v1 = not show_v1
        else:          show_v2 = not show_v2
        if not show_v1 and not show_v2:
            show_v1 = show_v2 = True
        if show_v1 and show_v2: self.graph_show_mode = 'both'
        elif show_v1:           self.graph_show_mode = 'v1'
        else:                   self.graph_show_mode = 'v2'
        for gi in range(3):
            if self.graph_show_mode in self._ax_xlim_per_mode[gi]:
                self._ax_xlim_full[gi] = self._ax_xlim_per_mode[gi][self.graph_show_mode]
        self._update_video_btn_visuals()
        self._show_video_frames()
        self.redraw_graphs()

    def _update_video_btn_visuals(self):
        show_v1 = self.graph_show_mode in ('v1', 'both')
        show_v2 = self.graph_show_mode in ('v2', 'both')
        for show, frame, lbl, swatch, col, dash in [
            (show_v1, self._v1_btn_frame, self._v1_toggle_lbl, self._v1_swatch, C_V1, (4,3)),
            (show_v2, self._v2_btn_frame, self._v2_toggle_lbl, self._v2_swatch, C_V2, ()),
        ]:
            bg = BG3 if show else BG2
            fg = col if show else '#555555'
            frame.config(bg=bg); lbl.config(bg=bg, fg=fg); swatch.config(bg=bg)
            swatch.delete('all')
            if dash:
                swatch.create_line(1, 6, 19, 6, fill=fg, dash=dash, width=2)
            else:
                swatch.create_line(1, 6, 19, 6, fill=fg, width=2)

    def _cycle_graph_view(self):
        modes  = ['both', 'v1', 'v2']
        self.graph_show_mode = modes[(modes.index(self.graph_show_mode)+1) % 3]
        for gi in range(3):
            if self.graph_show_mode in self._ax_xlim_per_mode[gi]:
                self._ax_xlim_full[gi] = self._ax_xlim_per_mode[gi][self.graph_show_mode]
        self._update_video_btn_visuals()
        self._show_video_frames()
        self.redraw_graphs()

    def _toggle_active(self):
        if len(self.datasets) >= 2:
            self.active_dataset_idx = 1 - self.active_dataset_idx
            self._show_video_frames()
            self.redraw_graphs()

    def _toggle_display_option(self, key):
        mapping = {
            'data':     ('show_data', True),
            'normal':   ('show_normative', True),
            'outliers': ('show_outliers_only', True),
            'aligned':  ('align_cycle_angles', self.show_overlaid_cycles),
            'stacked':  ('stack_cycle_lanes', self.show_overlaid_cycles),
        }
        attr, requires_cycles = mapping.get(key, (None, False))
        if attr is None: return
        if requires_cycles and not self.show_overlaid_cycles: return
        setattr(self, attr, not getattr(self, attr))
        self._update_display_btn_visuals()
        self.redraw_graphs()

    def _toggle_ankle_norm_offset(self):
        current = NORMATIVE_GAIT['ankle'].get('offset', 120.0)
        _apply_ankle_normative_offset(0.0 if current > 0.0 else 120.0)
        self.redraw_graphs()

    def _update_display_btn_visuals(self):
        active_map = {
            'cycles':   self.show_overlaid_cycles,
            'mean':     self.show_mean,
            'data':     self.show_data,
            'normal':   self.show_normative,
            'outliers': self.show_outliers_only,
            'aligned':  self.align_cycle_angles,
            'stacked':  self.stack_cycle_lanes,
            'world_px': USE_WORLD_LANDMARKS,
        }
        disabled_in_continuous = {'mean', 'data', 'normal', 'outliers'}
        
        # update cycles button text and state
        cycles_text = "Cycles" if self.show_overlaid_cycles else "Continuous"
        self._cycles_btn.config(text=cycles_text)
        if active_map['cycles']:
            self._cycles_btn.config(bg=ACCENT, fg='white')
        else:
            self._cycles_btn.config(bg=BG3, fg=TEXT)
        
        # update world button text and state
        world_text = "World" if USE_WORLD_LANDMARKS else "Pixel"
        self._world_btn.config(text=world_text)
        if active_map['world_px']:
            self._world_btn.config(bg=ACCENT, fg='white')
        else:
            self._world_btn.config(bg=BG3, fg=TEXT)
        
        # handle other buttons (mean, data, normal, outliers)
        for key, btn in self._display_btns.items():
            if key in ('cycles', 'world_px'):
                continue  # already handled above
            is_disabled = (not self.show_overlaid_cycles) and (key in disabled_in_continuous)
            if is_disabled:
                btn.config(
                    state='disabled',
                    bg=BG2,
                    fg='#777777',
                    disabledforeground='#777777',
                    activebackground=BG2,
                )
                continue
            btn.config(state='normal')
            if active_map.get(key, False):
                btn.config(bg=ACCENT, fg='white')
            else:
                btn.config(bg=BG3, fg=TEXT)

    def _on_skeleton_thickness_change(self, value):
        try:
            self.skeleton_thickness = max(0.0, min(float(DRAW_THICKNESS), float(value)))
        except Exception:
            return
        _save_ui_settings({'skeleton_thickness': self.skeleton_thickness})
        if self._marking_phase:
            self._markup_show_frames()
        else:
            self._show_video_frames()

    def _add_manual_step(self):
        if self._marking_phase:
            self._markup_add_step()
            return
        if self.graph_show_mode == 'v1' and len(self.datasets) >= 1:
            ds = self.datasets[0]
        elif self.graph_show_mode == 'v2' and len(self.datasets) >= 2:
            ds = self.datasets[1]
        else:
            ds = self._active_ds()
        if not ds or ds['angle_data'].empty: return
        idx = min(self.current_frame_idx, len(ds['angle_data'])-1)
        fn  = int(ds['angle_data']['frame_num'].iloc[idx])
        suggested = ds.get('suggested_step_frames', [])
        detected_foot = 'right'
        if suggested:
            nearest_idx = min(range(len(suggested)), key=lambda i: abs(suggested[i][0] - fn))
            detected_foot = suggested[nearest_idx][1]
        ds.setdefault('step_frames', []).append((fn, detected_foot))
        ds['step_frames'].sort(key=lambda x: x[0])
        self._persist_dataset_markup(ds)
        self._status_msg.set(f"Added {detected_foot} step @ frame {fn}")
        self._update_metrics_panel()
        self.redraw_graphs()

    def _delete_nearest_step(self):
        if self._marking_phase:
            self._markup_remove_last()
            return
        if self.graph_show_mode == 'v1' and len(self.datasets) >= 1:
            ds = self.datasets[0]
        elif self.graph_show_mode == 'v2' and len(self.datasets) >= 2:
            ds = self.datasets[1]
        else:
            ds = self._active_ds()
        if not ds or not ds.get('step_frames'): return
        idx = min(self.current_frame_idx, len(ds['angle_data'])-1)
        fn  = int(ds['angle_data']['frame_num'].iloc[idx])
        i   = min(range(len(ds['step_frames'])),
                  key=lambda k: abs(ds['step_frames'][k][0] - fn))
        removed = ds['step_frames'].pop(i)
        self._persist_dataset_markup(ds)
        self._status_msg.set(f"Removed step @ frame {removed[0]}")
        self._update_metrics_panel()
        self.redraw_graphs()

    def _toggle_suggestions(self):
        self.show_suggestions = not self.show_suggestions
        self.redraw_graphs()

    def _clear_steps(self):
        if not self.datasets: return
        if not messagebox.askyesno("Clear Steps", "Clear all manual steps?", parent=self):
            return
        for ds in self.datasets:
            if ds:
                ds['step_frames'] = []
                self._persist_dataset_markup(ds)
        self._status_msg.set("Steps cleared")
        self._update_metrics_panel()
        self.redraw_graphs()

    def _clear_exclusions(self):
        if not self.datasets: return
        for ds in self.datasets:
            if ds:
                ds['excluded_regions'] = []
                self._persist_dataset_markup(ds)
        self._status_msg.set("All exclusions cleared")
        self.redraw_graphs()

    def _recompute_steps(self):
        if not self.datasets: return
        for ds in self.datasets:
            if ds and 'angle_data' in ds:
                ad = ds.get('angle_data', pd.DataFrame())
                depths = ds.get('landmark_depths', pd.DataFrame())
                suggested, auto_excl, step_meta = detect_steps_robust(ad, depth_df=depths, fps=SLOWMO_FPS)
                ds['suggested_step_frames'] = suggested
                ds['suggested_step_meta'] = step_meta
                ds['excluded_regions'] = self._merge_exclusion_regions(list(auto_excl))
                self._persist_dataset_markup(ds)
        self._status_msg.set("Auto steps recomputed")
        self._update_metrics_panel()
        self.redraw_graphs()

    def _open_settings(self):
        if self._settings_dialog is not None and self._settings_dialog.winfo_exists():
            self._settings_dialog.lift(); self._settings_dialog.focus()
        else:
            self._settings_dialog = SettingsDialog(self, self)
            dialog = self._settings_dialog
            def on_close():
                self._settings_dialog = None
                dialog.destroy()
            self._settings_dialog.protocol("WM_DELETE_WINDOW", on_close)

    def _restart_marking_wizard(self):
        if len(self.datasets) < 2:
            messagebox.showwarning("No videos loaded",
                                   "Load two videos before reopening the step marking wizard.", parent=self)
            return
        for ds in self.datasets:
            if ds: self._persist_dataset_markup(ds)
        self._markup_required_videos = [True] * len(self.datasets)
        self.playing = False
        if self._play_after_id:
            self.after_cancel(self._play_after_id)
            self._play_after_id = None
        self._enter_marking_phase('left', 0)

    # guided step marking screen

    def _build_markup_screen(self):
        self._markup_frame = tk.Frame(self, bg=BG)
        self._markup_banner_area = tk.Frame(self._markup_frame, bg=BG2)
        self._markup_banner_area.pack(fill='x')
        top_row = tk.Frame(self._markup_banner_area, bg=BG2)
        top_row.pack(fill='x', padx=20, pady=(10, 0))
        self._markup_step_lbl = tk.Label(top_row, text="STEP  1  OF  4",
            font=("Helvetica", 9, "bold"), bg=BG2, fg=SUBTEXT, anchor='w')
        self._markup_step_lbl.pack(side='left')
        self._markup_side_badge = tk.Label(top_row, text=" LEFT ",
            font=("Helvetica", 9, "bold"), bg=C_LEFT, fg='white', padx=6, pady=1)
        self._markup_side_badge.pack(side='left', padx=(10, 0))
        self._markup_count_lbl = tk.Label(top_row, text="0 steps marked",
            font=("Helvetica", 10, "bold"), bg=BG2, fg=C_LEFT, anchor='e')
        self._markup_count_lbl.pack(side='right')
        title_row = tk.Frame(self._markup_banner_area, bg=BG2)
        title_row.pack(fill='x', padx=20, pady=(2, 4))
        self._markup_banner_lbl = tk.Label(title_row,
            text="MARKING LEFT STEPS — VIDEO 1",
            font=("Coiny Cyrillic", 13,), bg=BG2, fg=ACCENT, anchor='w')
        self._markup_banner_lbl.pack(side='left')
        self._markup_sub_lbl = tk.Label(title_row,
            text="· press  SPACE  to mark each LEFT foot strike",
            font=("Helvetica", 9), bg=BG2, fg=SUBTEXT, anchor='w')
        self._markup_sub_lbl.pack(side='left', padx=(12, 0))
        tk.Frame(self._markup_banner_area, bg=BG3, height=1).pack(fill='x')
        vid_area = tk.Frame(self._markup_frame, bg=BG)
        vid_area.pack(fill='both', expand=True, padx=8, pady=(8, 0))
        vf = tk.Frame(vid_area, bg=BG2, bd=1, relief='flat')
        vf.pack(fill='both', expand=True)
        self._markup_vid_lbl = tk.Label(vf, text="VIDEO 1",
            font=("Helvetica", 9, "bold"), bg=BG2, fg=C_V1, anchor='w')
        self._markup_vid_lbl.pack(fill='x', padx=6, pady=(3, 0))
        self._markup_canvas = tk.Canvas(vf, bg=BG_VID, highlightthickness=0)
        self._markup_canvas.pack(fill='both', expand=True)
        self._markup_canvases = [self._markup_canvas]
        graph_area = tk.Frame(self._markup_frame, bg=BG2)
        graph_area.pack(fill='x', padx=8, pady=(6, 0))
        self._markup_fig, self._markup_ax = plt.subplots(figsize=(12, 1.6), dpi=100)
        self._markup_fig.patch.set_facecolor(BG2)
        self._markup_ax.set_facecolor(BG_PLOT)
        for spine in self._markup_ax.spines.values():
            spine.set_color(BG2)
        self._markup_fig.subplots_adjust(left=0.03, right=0.99, top=0.88, bottom=0.28)
        self._markup_mpl_canvas = FigureCanvasTkAgg(self._markup_fig, master=graph_area)
        self._markup_mpl_canvas.get_tk_widget().pack(fill='x')
        self._markup_mpl_canvas.mpl_connect('button_press_event',   self._on_markup_graph_click)
        self._markup_mpl_canvas.mpl_connect('motion_notify_event',  self._on_markup_graph_drag)
        self._markup_mpl_canvas.mpl_connect('button_release_event', self._on_markup_graph_release)
        self._markup_graph_dragging = False
        ctrl_row = tk.Frame(self._markup_frame, bg=BG2, height=52)
        ctrl_row.pack(fill='x', side='bottom')
        ctrl_row.pack_propagate(False)
        tk.Button(ctrl_row, text="⌫  Undo Last Step",
            font=("Helvetica", 9), bg=BG3, fg=TEXT, relief='flat', padx=12, cursor='hand2',
            command=self._markup_remove_last).pack(side='left', padx=16, pady=10)
        self._markup_frame_lbl = tk.Label(ctrl_row, text="Frame 3  (0.01 s)",
            font=("Helvetica", 9, "bold"), bg=BG2, fg=TEXT)
        self._markup_frame_lbl.pack(side='left', padx=(0, 16))
        self._markup_continue_btn = tk.Button(ctrl_row, text="Done  →  Next",
            font=("Helvetica", 11, "bold"),
            bg=ACCENT, fg='white', relief='flat', padx=20, cursor='hand2',
            activebackground='#5a186a', activeforeground='white')
        self._markup_continue_btn.pack(side='right', padx=20, pady=10)
        tk.Label(ctrl_row,
            text="Click graph to seek  ·  SPACE = mark step  ·  1/2 = frame by frame",
            font=("Helvetica", 8), bg=BG2, fg=SUBTEXT).pack(side='left', padx=8)

    def _enter_marking_phase(self, side, video_idx):
        self._marking_phase = side
        self._marking_video_idx = video_idx
        if self._markup_frame is None:
            self._build_markup_screen()
        self.current_frame_idx = 2
        t0 = 2 / SLOWMO_FPS
        self._markup_frame_lbl.config(text=f"Frame 3  ({t0:.2f} s)")
        phase_order = [(s, vi) for s, vi in [('left', 0), ('left', 1), ('right', 0), ('right', 1)]
                       if self._dataset_needs_markup(vi)]
        phase_num = phase_order.index((side, video_idx)) + 1 if (side, video_idx) in phase_order else 1
        phase_total = max(1, len(phase_order))
        noun = "LEFT" if side == 'left' else "RIGHT"
        vid_num = video_idx + 1
        vid_name = self.video_names[video_idx] if video_idx < len(self.video_names) else f"Video {vid_num}"
        fg_col = C_V1 if video_idx == 0 else C_V2
        marker_col = C_LEFT if side == 'left' else C_RIGHT
        next_phase = self._next_markup_phase(side, video_idx)
        if next_phase is not None:
            next_side, next_video_idx = next_phase
            next_side_label = "LEFT" if next_side == 'left' else "RIGHT"
            continue_txt = f"Done  →  Mark {next_side_label} steps on Video {next_video_idx + 1}"
            continue_cmd = lambda s=next_side, vi=next_video_idx: self._enter_marking_phase(s, vi)
        else:
            continue_txt = "Finish  →  View Gait Analysis"
            continue_cmd = self._exit_marking_phase
        self._markup_step_lbl.config(text=f"STEP  {phase_num}  OF  {phase_total}")
        self._markup_side_badge.config(text=f"  {noun}  ", bg=marker_col)
        self._markup_banner_lbl.config(text=f"MARKING {noun} STEPS — VIDEO {vid_num}")
        self._markup_sub_lbl.config(text=f"· press  SPACE  to mark each {noun} foot strike  ({vid_name})")
        self._markup_count_lbl.config(fg=marker_col)
        self._markup_vid_lbl.config(text=f"VIDEO {vid_num}  —  {vid_name}", fg=fg_col)
        self._markup_continue_btn.config(text=continue_txt, command=continue_cmd)
        self._markup_count_update()
        self._markup_show_frames()
        self._main_content.pack_forget()
        self._bottom_bar.pack_forget()
        self._markup_frame.pack(fill='both', expand=True)
        self._status_msg.set(f"Mark every {noun} foot strike with SPACE — Video {vid_num}")
        self.after(50, self._markup_show_frames)
        self.after(60, self._redraw_markup_graph)

    def _exit_marking_phase(self):
        self._marking_phase = None
        self._markup_frame.pack_forget()
        self._bottom_bar.pack(fill='x', side='bottom')
        self._main_content.pack(fill='both', expand=True, padx=6, pady=(4, 0))
        self._persist_all_dataset_markup()
        self.show_overlaid_cycles = True
        self.resample_cycles = True
        self._status_msg.set("Steps confirmed — showing overlaid gait cycles")
        self._update_display_btn_visuals()
        self.refresh()

    def _markup_show_frames(self):
        vi = self._marking_video_idx
        canvas = self._markup_canvas
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        canvas.delete('all')
        frame = None
        pixel_lm = None
        if vi < len(self.datasets):
            store = self.datasets[vi].get('all_landmarks', [])
            if 0 <= self.current_frame_idx < len(store):
                entry = store[self.current_frame_idx]
                if isinstance(entry, tuple):
                    frame = self._cache.get(vi, self.current_frame_idx, store)
                    pixel_lm = entry[1]
                else:
                    frame = self._cache.get(vi, self.current_frame_idx, store)
        if frame is None or cw < 2 or ch < 2:
            if cw >= 2 and ch >= 2:
                canvas.create_text(cw // 2, ch // 2, text="No frame", fill=SUBTEXT, font=("Helvetica", 10))
            return
        if pixel_lm is not None:
            jittery_frames = self.datasets[vi].get('jittery_frames', set()) if self.remove_jitter_frames else set()
            is_jittery = self.current_frame_idx in jittery_frames
            if not (is_jittery and not self.show_jitter_frames):
                frame = frame.copy()
                draw_pose_landmarks_on_frame(frame, pixel_lm, self.joint_visibility,
                                             focus_side=self._marking_phase,
                                             skeleton_thickness=self.skeleton_thickness,
                                             draw_jitter_red=is_jittery and self.show_jitter_frames)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fh, fw = rgb.shape[:2]
        scale = min(cw / fw, ch / fh)
        nw, nh = int(fw * scale), int(fh * scale)
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        canvas._img = img
        canvas.create_image((cw - nw) // 2, (ch - nh) // 2, anchor='nw', image=img)

    def _redraw_markup_graph(self):
        ax = self._markup_ax
        ax.cla()
        ax.set_facecolor(BG_PLOT)
        for spine in ax.spines.values():
            spine.set_color(BG2)
        ax.tick_params(colors=SUBTEXT, labelsize=7)
        vi = self._marking_video_idx
        if not self.datasets or vi >= len(self.datasets):
            self._markup_mpl_canvas.draw_idle()
            return
        ds  = self.datasets[vi]
        ad  = ds.get('angle_data')
        if ad is None or ad.empty:
            self._markup_mpl_canvas.draw_idle()
            return
        frames = ad['frame_num'].values
        excluded = ds.get('excluded_regions', [])
        excl_mask = np.zeros(len(frames), dtype=bool)
        for s, e in excluded:
            excl_mask |= (frames >= s) & (frames < e)
        for s, e in excluded:
            ax.axvspan(s, e, alpha=0.18, color='#6e6e6e', zorder=1)
        for joint, col in JOINT_COLORS_MPL.items():
            if joint not in ad.columns: continue
            vals = ad[joint].values.astype(float)
            if excluded:
                gray_vals = vals.copy(); gray_vals[~excl_mask] = np.nan
                ax.plot(frames, gray_vals, color='#7a7a7a', lw=1.2, alpha=0.9, zorder=2)
            clean_vals = vals.copy(); clean_vals[excl_mask] = np.nan
            ax.plot(frames, clean_vals, color=col, lw=1.0, alpha=0.85, zorder=3)
        side = self._marking_phase
        if side:
            fn_set = set(int(f) for f in frames)
            for f, s in ds.get('step_frames', []):
                if s == side and int(f) in fn_set:
                    ax.axvline(int(f), color=C_LEFT if s == 'left' else C_RIGHT, lw=1.5, alpha=0.9, zorder=5)
        if self.current_frame_idx < len(ad):
            cf = int(ad['frame_num'].iloc[self.current_frame_idx])
            ax.axvline(cf, color=C_CURSOR, lw=1.5, linestyle='--', zorder=10)
        ax.set_xlim(frames[0], frames[-1])
        ax.set_xlabel('Frame', fontsize=7, color=SUBTEXT)
        ax.set_ylabel('°', fontsize=7, color=SUBTEXT)
        self._markup_fig.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.28)
        self._markup_mpl_canvas.draw_idle()

    def _on_markup_graph_click(self, event):
        if event.button == 1 and event.inaxes == self._markup_ax:
            self._markup_graph_dragging = True
            self._markup_seek_from_event(event)

    def _on_markup_graph_drag(self, event):
        if self._markup_graph_dragging and event.inaxes == self._markup_ax:
            self._markup_seek_from_event(event)

    def _on_markup_graph_release(self, event):
        if event.button == 1:
            self._markup_graph_dragging = False

    def _markup_seek_from_event(self, event):
        if event.xdata is None: return
        vi = self._marking_video_idx
        if not self.datasets or vi >= len(self.datasets): return
        ad = self.datasets[vi].get('angle_data')
        if ad is None or ad.empty: return
        fn  = ad['frame_num'].to_numpy()
        idx = int(np.argmin(np.abs(fn - event.xdata)))
        self.current_frame_idx = max(0, min(idx, len(ad) - 1))
        t = self.current_frame_idx / SLOWMO_FPS
        self._markup_frame_lbl.config(text=f"Frame {self.current_frame_idx + 1}  ({t:.2f} s)")
        self._markup_show_frames()
        self._redraw_markup_graph()

    def _markup_add_step(self):
        side = self._marking_phase
        if not side or not self.datasets: return
        vi = self._marking_video_idx
        if vi >= len(self.datasets): return
        ds = self.datasets[vi]
        if ds and not ds['angle_data'].empty:
            idx = min(self.current_frame_idx, len(ds['angle_data']) - 1)
            fn = int(ds['angle_data']['frame_num'].iloc[idx])
            ds.setdefault('step_frames', []).append((fn, side))
            ds['step_frames'].sort(key=lambda x: x[0])
            self._persist_dataset_markup(ds)
        self._markup_count_update()
        self._redraw_markup_graph()

    def _markup_remove_last(self):
        side = self._marking_phase
        if not side or not self.datasets: return
        vi = self._marking_video_idx
        if vi >= len(self.datasets): return
        ds = self.datasets[vi]
        if ds and ds.get('step_frames'):
            for i in range(len(ds['step_frames']) - 1, -1, -1):
                if ds['step_frames'][i][1] == side:
                    ds['step_frames'].pop(i)
                    self._persist_dataset_markup(ds)
                    break
        self._markup_count_update()
        self._redraw_markup_graph()

    def _markup_count_update(self):
        side = self._marking_phase
        if not side or self._markup_frame is None: return
        vi = self._marking_video_idx
        if vi >= len(self.datasets): return
        ds = self.datasets[vi]
        total = sum(1 for _, s in ds.get('step_frames', []) if s == side) if ds else 0
        marker_col = C_LEFT if side == 'left' else C_RIGHT
        noun = "LEFT" if side == 'left' else "RIGHT"
        self._markup_count_lbl.config(
            text=f"{total} {noun} step{'s' if total != 1 else ''} marked", fg=marker_col)

    # pdf export dialog

    def _generate_pdf(self, output_path, graphs, limbs, measures):
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
        temp_images = []
        try:
            doc = SimpleDocTemplate(output_path, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
            story = []
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24,
                                         textColor=colors.HexColor('#2c3e50'), spaceAfter=6, alignment=1)
            story.append(Paragraph("Gait Analysis Report", title_style))
            story.append(Spacer(1, 0.1*inch))
            subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=11,
                                            textColor=colors.HexColor('#7f8c8d'), alignment=1)
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", subtitle_style))
            story.append(Spacer(1, 0.3*inch))
            graph_height = 3.5*inch
            graph_width = 6.5*inch
            if 'continuous' in graphs:
                story.append(Paragraph("Continuous View (Raw Angle Data)", styles['Heading2']))
                img_path = self._capture_graph_image('continuous', graphs['continuous'], limbs)
                if img_path:
                    try:
                        story.append(RLImage(img_path, width=graph_width, height=graph_height))
                        story.append(Spacer(1, 0.2*inch))
                        temp_images.append(img_path)
                    except Exception as e:
                        print(f"Error adding image: {e}")
            if 'cycles' in graphs:
                story.append(Paragraph("Overlaid Cycles (Normalized Gait Cycles)", styles['Heading2']))
                img_path = self._capture_graph_image('cycles', graphs['cycles'], limbs)
                if img_path:
                    try:
                        story.append(RLImage(img_path, width=graph_width, height=graph_height))
                        story.append(Spacer(1, 0.2*inch))
                        temp_images.append(img_path)
                    except Exception as e:
                        print(f"Error adding image: {e}")
            if any(measures.values()) and len(self.datasets) >= 2:
                story.append(PageBreak())
                story.append(Paragraph("Outcome Measures", styles['Heading2']))
                story.append(Spacer(1, 0.15*inch))
                metrics = compute_metrics(self.datasets[0], self.datasets[1])
                table_data = [['Measure', 'Change (%)']]
                for key, label in PDFExportDialog.OUTCOME_MEASURES.items():
                    if measures.get(key):
                        value = metrics.get(key, 0)
                        value_str = f"{value:+.1f}%" if isinstance(value, (int, float)) else "N/A"
                        table_data.append([label, value_str])
                if len(table_data) > 1:
                    table = Table(table_data, colWidths=[4.5*inch, 1.5*inch])
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 11),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
                        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 1), (-1, -1), 10),
                        ('TOPPADDING', (0, 1), (-1, -1), 8),
                        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
                    ]))
                    story.append(table)
            doc.build(story)
        finally:
            for img_path in temp_images:
                try:
                    if os.path.exists(img_path):
                        time.sleep(0.05)
                        os.unlink(img_path)
                except Exception as e:
                    print(f"Warning: Could not delete temp file {img_path}: {e}")

    def _capture_graph_image(self, graph_type, graph_options=None, limbs=None):
        if graph_options is None:
            graph_options = {'show_versions': 'both', 'include_excluded': True}
        if limbs is None:
            limbs = {k: True for k in ('left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle')}
        try:
            import tempfile as tf
            temp_dir = tf.gettempdir()
            temp_path = os.path.join(temp_dir, f'gait_graph_{int(time.time()*1000)}_{graph_type}.png')
            orig_mode = self.show_overlaid_cycles
            orig_graph_mode = self.graph_show_mode
            orig_joint_visibility = self.joint_visibility.copy()
            orig_show_excluded = getattr(self, '_show_excluded_in_pdf', True)
            try:
                self.show_overlaid_cycles = (graph_type == 'cycles')
                self.graph_show_mode = graph_options.get('show_versions', 'both')
                self.joint_visibility = limbs.copy()
                self._show_excluded_in_pdf = graph_options.get('include_excluded', True)
                self.redraw_graphs()
                self.update()
                # capture the first graph figure for pdf (ankle as representative)
                self._fig_ankle.savefig(temp_path, dpi=150, bbox_inches='tight', format='png')
                gc.collect()
                time.sleep(0.1)
                if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                    raise Exception(f"Image file not created: {temp_path}")
                return temp_path
            finally:
                self.show_overlaid_cycles = orig_mode
                self.graph_show_mode = orig_graph_mode
                self.joint_visibility = orig_joint_visibility
                self._show_excluded_in_pdf = orig_show_excluded
                self.redraw_graphs()
        except Exception as e:
            print(f"Error capturing graph: {e}")
            return None

    # close and cleanup

    def _on_close(self):
        self._stop_pf = True
        if self.playing and self._play_after_id:
            self.after_cancel(self._play_after_id)
        plt.close('all')
        self.destroy()


# session temp directory cleanup

class SessionTempDir:
    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="gait_")
        self.path = self._tmp.name
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT,  self._on_sig)
        signal.signal(signal.SIGTERM, self._on_sig)

    def _on_sig(self, sig, frame):
        self.cleanup()
        signal.signal(sig, signal.SIG_DFL)
        raise KeyboardInterrupt

    def cleanup(self):
        try: self._tmp.cleanup()
        except Exception: pass

    def __enter__(self): return self
    def __exit__(self, *_): self.cleanup()


# entry point

def main():
    app = GaitAnalysisDashboard()
    app._session = SessionTempDir()
    app._status_msg.set("Select two videos to begin")
    app.mainloop()
    app._session.cleanup()


if __name__ == "__main__":
    main()