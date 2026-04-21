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

pyglet.font.add_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Coiny-Cyrillic.ttf'))


def resource_path(filename):
    base = sys._MEIPASS if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, filename)

# analysis settings
SLOWMO_FPS    = 240
FILTER_CUTOFF = 6   # increased cutoff to reduce drag while maintaining smoothing
FILTER_ORDER  = 3      # reduced order for weaker filter

# frame storage settings
SAVE_HEIGHT  = 540
JPEG_QUALITY = 65
CACHE_FRAMES = 96
DEBUG_DIRECTION_DIAGNOSTICS = False
DEBUG_LOG_JITTER = False 
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
        # auto-detect coordinate columns (landmark_i_x, landmark_i_y, landmark_i_z)
        columns = [c for c in df.columns if 'landmark_' in c and c.endswith(('_x', '_y', '_z'))]
    
    for col in columns:
        if col not in df.columns:
            continue
        
        # calculate frame-to-frame displacement
        diff = df[col].diff().abs()
        bad_frames = diff[diff > max_frame_displacement].index.tolist()
        
        if len(bad_frames) > 0:
            # interpolate across bad frames from neighbors
            for frame_idx in bad_frames:
                if 0 < frame_idx < len(df) - 1:
                    # estimate from surrounding frames
                    before = df.iloc[frame_idx - 1][col]
                    after = df.iloc[frame_idx + 1][col]
                    df.at[frame_idx, col] = (before + after) / 2.0
    
    return df


def _fix_limb_swaps(df):
    # hip indices: left=23, right=24; similar pattern for other joints
    # left landmarks should have smaller x (left side of frame), right should have larger x
    # only fix obvious swaps (large overlap) to avoid over-correcting and creating discontinuities
    
    landmark_pairs = [
        ('landmark_23_x', 'landmark_24_x'),  # left/right hip
        ('landmark_25_x', 'landmark_26_x'),  # left/right knee
        ('landmark_27_x', 'landmark_28_x'),  # left/right ankle
    ]
    
    for frame_idx in range(len(df)):
        for left_col, right_col in landmark_pairs:
            if left_col not in df.columns or right_col not in df.columns:
                continue
            
            left_x = df.iloc[frame_idx][left_col]
            right_x = df.iloc[frame_idx][right_col]
            
            # only fix if swap is definite: left significantly overlaps right
            # this avoids correcting borderline cases that may just be natural variance
            if not pd.isna(left_x) and not pd.isna(right_x) and (right_x - left_x) < -0.05:
                if 0 < frame_idx < len(df) - 1:
                    before_lx = df.iloc[frame_idx - 1][left_col]
                    after_lx = df.iloc[frame_idx + 1][left_col]
                    before_rx = df.iloc[frame_idx - 1][right_col]
                    after_rx = df.iloc[frame_idx + 1][right_col]
                    
                    # use quadratic interpolation for smoother blending
                    if not pd.isna(before_lx) and not pd.isna(after_lx):
                        # weighted towards neighbors, lighter correction
                        df.at[frame_idx, left_col] = 0.25 * before_lx + 0.50 * left_x + 0.25 * after_lx
                    if not pd.isna(before_rx) and not pd.isna(after_rx):
                        df.at[frame_idx, right_col] = 0.25 * before_rx + 0.50 * right_x + 0.25 * after_rx
    
    return df


def _detect_jittery_frames(pixel_landmarks, threshold=0.15):
    # detect frames with excessive frame-to-frame landmark displacement (jitter)
    # returns a set of frame indices. pixel_landmarks is list of (raw_path, landmark_list) tuples
    # each landmark has x, y in 0-1 range. threshold is max allowed displacement between frames (default 0.15)
    jittery_frames = set()
    
    if not pixel_landmarks or len(pixel_landmarks) < 2:
        return jittery_frames
    
    # track previous valid frame's landmark positions
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
        
        # check displacement from previous frame
        if prev_landmarks is not None and len(prev_landmarks) == len(current_lm):
            max_displacement = 0.0
            for prev_lm, curr_lm in zip(prev_landmarks, current_lm):
                if prev_lm.visibility > 0.5 and curr_lm.visibility > 0.5:
                    # calculate euclidean distance in normalized coords (0-1)
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
        return {
            'size_bytes': os.path.getsize(video_path),
            'frame_count': 0,
            'fps': 0.0,
            'width': 0,
            'height': 0,
        }
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        'size_bytes': os.path.getsize(video_path),
        'frame_count': frame_count,
        'fps': fps,
        'width': width,
        'height': height,
    }


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
        return {
            'step_frames': step_frames,
        }
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


# pose landmark indices
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
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),          # face
    (9,10),                                                     # mouth
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),   # left arm
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),           # right arm
    (11,23),(12,24),(23,24),                                    # torso
    (23,25),(24,26),(25,27),(26,28),                            # upper leg
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),           # lower leg and foot
]

# skeleton display settings
DRAW_THICKNESS      = 8
USE_WORLD_LANDMARKS = True
SKELETON_LINE_COL   = (0, 0, 255)   # cyan line

JOINT_NAME_TO_LANDMARK = {
    'left_hip':    PoseLandmark.LEFT_HIP,
    'right_hip':   PoseLandmark.RIGHT_HIP,
    'left_knee':   PoseLandmark.LEFT_KNEE,
    'right_knee':  PoseLandmark.RIGHT_KNEE,
    'left_ankle':  PoseLandmark.LEFT_ANKLE,
    'right_ankle': PoseLandmark.RIGHT_ANKLE,
}

# reverse lookup from landmark index to joint name
LANDMARK_TO_JOINT_NAME = {v: k for k, v in JOINT_NAME_TO_LANDMARK.items()}

# joint colors
JOINT_COLORS_MPL = {
    'left_hip':    '#c0392b', 'left_knee':   '#d35400', 'left_ankle':  '#8e44ad',
    'right_hip':   '#2471a3', 'right_knee':  '#1a5276', 'right_ankle': '#148f77',
}

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)  # opencv uses bgr, not rgb

# convert mpl colors for opencv drawing
JOINT_COLORS_BGR = {k: hex_to_bgr(v) for k, v in JOINT_COLORS_MPL.items()}
C_RIGHT = '#4a90d9'   # right step marker
C_LEFT  = '#e8913a'   # left step marker

# ui colors
BG      = "#f0f0f0"   # window background
BG2     = "#d6d6d6"   # header and panels
BG3     = "#c8c8c8"   # cards and toolbar
BG_VID  = "#d8d8d8"   # video canvas
BG_PLOT = "#dbdbdb"   # graph axes
BG_INIT = "#c0c0c0"   # graph before data

# text colors
ACCENT  = "#3a083a"   # logo and header accent
TEXT    = "#1a1a1a"
SUBTEXT = "#4a4a4a"
GREEN   = '#27ae60'
RED     = '#c0392b'

# label colors
C_V1      = "#4a1a44"   # video 1 tint
C_V2      = "#2e6b40"   # video 2 tint
C_CURSOR  = '#ff4444'   # playhead line
C_OUTLIER = '#555555'   # outlier cycle
C_NORM    = '#888888'   # normative band

# normative gait reference data
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
NORMATIVE_GAIT['ankle']['mean'] = NORMATIVE_GAIT['ankle']['mean'][:100]
for jt in NORMATIVE_GAIT:
    m  = np.array(NORMATIVE_GAIT[jt]["mean"])
    # replace any NaN with nearest valid value
    m = np.nan_to_num(m, nan=np.nanmean(m))
    sd = np.std(m)
    se = sd / np.sqrt(len(m))  # standard error of the mean
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
        # resample cycle to match reference length
        t = np.linspace(0, 1, len(y_values))
        y_resampled = interp1d(t, y_values)(np.linspace(0, 1, max_cycle_length))
        
        # ensure reference has same length
        if len(reference_y) != max_cycle_length:
            t_ref = np.linspace(0, 1, len(reference_y))
            reference_resampled = interp1d(t_ref, reference_y)(np.linspace(0, 1, max_cycle_length))
        else:
            reference_resampled = reference_y
        
        # compute rmse
        rmse = float(np.sqrt(np.mean((y_resampled - reference_resampled) ** 2)))
        
        # normalize by the larger range (cycle or reference) for lenient filtering
        cycle_range = np.nanmax(y_resampled) - np.nanmin(y_resampled)
        ref_range = np.nanmax(reference_resampled) - np.nanmin(reference_resampled)
        max_range = max(cycle_range, ref_range)
        
        if max_range <= 0:
            return 0.0
        
        rmse_pct = (rmse / max_range) * 100.0
        return rmse_pct
    except Exception:
        return 0.0

# drawing and analysis helpers
def draw_pose_landmarks_on_frame(frame_bgr, pixel_landmarks, joint_visibility=None, focus_side=None,
                                 skeleton_thickness=None, draw_jitter_red=False):
    h, w = frame_bgr.shape[:2]
    if skeleton_thickness is None:
        skeleton_thickness = DRAW_THICKNESS
    if skeleton_thickness <= 0:
        return frame_bgr

    # keep skeleton thickness consistent across crop sizes
    line_thickness = max(1, int(h * skeleton_thickness / 540))
    circle_radius  = max(1, int(h * (skeleton_thickness * 0.5) / 540))
    default_line_col = GRAY_BGR if focus_side in ('left', 'right') else (200, 200, 200)

    def _side_matches(jname):
        if focus_side not in ('left', 'right'):
            return True
        return jname.startswith(focus_side + '_')
    
    def _resolve_color(landmark_idx, base_color):
        # if marking jitter frame, all colors become red
        if draw_jitter_red:
            return (0, 0, 255)  # red in BGR
        if landmark_idx not in LANDMARK_TO_JOINT_NAME:
            return GRAY_BGR if focus_side in ('left', 'right') else base_color
        jname = LANDMARK_TO_JOINT_NAME[landmark_idx]
        if not _side_matches(jname):
            return GRAY_BGR
        if joint_visibility is not None and not joint_visibility.get(jname, True):
            return GRAY_BGR
        return JOINT_COLORS_BGR[jname]

    # draw limb connections
    for s, e in POSE_CONNECTIONS:
        if s < len(pixel_landmarks) and e < len(pixel_landmarks):
            ls, le = pixel_landmarks[s], pixel_landmarks[e]
            if ls.visibility > 0.5 and le.visibility > 0.5:
                # prefer the tracked joint color when available
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
    
    # draw landmark circles
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
    # calculate the joint angle
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

    # collect the union of non black regions across samples
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
        # blur first to reduce border compression noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # threshold slightly above black to catch border noise
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

    # convert each border width to a frame fraction
    margin = 0.02
    left_frac   = x_min / full_w
    top_frac    = y_min / full_h
    right_frac  = (full_w - x_max) / full_w
    bottom_frac = (full_h - y_max) / full_h

    # skip cropping when borders are negligible
    if max(left_frac, top_frac, right_frac, bottom_frac) < margin:
        return None  # borders are too small to crop

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

        # reject peaks that arrive much too soon after the last accepted peak
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

        # allow lighter thresholds right after a bad segment ends
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

    # drop duplicate hits that are too close in time
    deduped = [candidates[0]]
    close_frames = int(0.20 * fps)
    for c in candidates[1:]:
        prev = deduped[-1]
        if c[0] - prev[0] <= close_frames:
            if c[2] > prev[2]:
                deduped[-1] = c
        else:
            deduped.append(c)

    # enforce left-right alternation and plausible step timing
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


# joint angle definitions
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

def _log_direction_diagnostics(df_w, df_p, video_path, detected_rotation):
    import csv
    from datetime import datetime
    
    output_dir = os.path.expanduser('~/Desktop/Gait_Analysis')
    os.makedirs(output_dir, exist_ok=True)
    vid_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_path = os.path.join(output_dir, f'direction_diagnostics_{vid_name}_{timestamp}.csv')
    
    rows = []
    
    for df_label, df in [('world', df_w), ('pixel', df_p)]:
        if '_direction' not in df.columns or df.empty:
            continue
        
        directions = df['_direction'].values
        # smooth the direction labels before summarizing them
        smoothed = directions.copy()
        window = 30
        for i in range(len(smoothed)):
            start = max(0, i - window)
            end = min(len(smoothed), i + window + 1)
            chunk = directions[start:end]
            left_count = np.sum(chunk == 'left')
            smoothed[i] = 'left' if left_count > len(chunk) // 2 else 'right'
        
        left_mask = smoothed == 'left'
        right_mask = smoothed == 'right'
        
        joint_cols = [c for c in df.columns if c not in ('frame_num', '_direction')]
        
        for col in joint_cols:
            vals = df[col].values.astype(float)
            
            left_vals = vals[left_mask]
            right_vals = vals[right_mask]
            
            row = {
                'video': vid_name,
                'landmark_type': df_label,
                'rotation_applied': detected_rotation,
                'joint': col,
                'left_dir_frames': int(left_mask.sum()),
                'right_dir_frames': int(right_mask.sum()),
                'left_mean': float(np.nanmean(left_vals)) if len(left_vals) > 0 else None,
                'left_std': float(np.nanstd(left_vals)) if len(left_vals) > 0 else None,
                'left_min': float(np.nanmin(left_vals)) if len(left_vals) > 0 else None,
                'left_max': float(np.nanmax(left_vals)) if len(left_vals) > 0 else None,
                'left_median': float(np.nanmedian(left_vals)) if len(left_vals) > 0 else None,
                'right_mean': float(np.nanmean(right_vals)) if len(right_vals) > 0 else None,
                'right_std': float(np.nanstd(right_vals)) if len(right_vals) > 0 else None,
                'right_min': float(np.nanmin(right_vals)) if len(right_vals) > 0 else None,
                'right_max': float(np.nanmax(right_vals)) if len(right_vals) > 0 else None,
                'right_median': float(np.nanmedian(right_vals)) if len(right_vals) > 0 else None,
                'offset_left_minus_right': float(np.nanmean(left_vals) - np.nanmean(right_vals)) if len(left_vals) > 0 and len(right_vals) > 0 else None,
            }
            
            # summarize peaks and troughs for each walking direction
            for dir_label, dir_vals in [('left', left_vals), ('right', right_vals)]:
                if len(dir_vals) > 20:
                    peaks, _ = find_peaks(dir_vals, distance=40, prominence=1.0)
                    troughs, _ = find_peaks(-dir_vals, distance=40, prominence=1.0)
                    row[f'{dir_label}_peak_mean'] = float(np.mean(dir_vals[peaks])) if len(peaks) > 0 else None
                    row[f'{dir_label}_peak_count'] = int(len(peaks))
                    row[f'{dir_label}_trough_mean'] = float(np.mean(dir_vals[troughs])) if len(troughs) > 0 else None
                    row[f'{dir_label}_trough_count'] = int(len(troughs))
                    row[f'{dir_label}_rom'] = float(np.mean(dir_vals[peaks]) - np.mean(dir_vals[troughs])) if len(peaks) > 0 and len(troughs) > 0 else None
                else:
                    row[f'{dir_label}_peak_mean'] = None
                    row[f'{dir_label}_peak_count'] = 0
                    row[f'{dir_label}_trough_mean'] = None
                    row[f'{dir_label}_trough_count'] = 0
                    row[f'{dir_label}_rom'] = None
            
            rows.append(row)
    
    # save raw per frame values to a second csv
    raw_path = os.path.join(output_dir, f'raw_angles_{vid_name}_{timestamp}.csv')
    df_raw = df_w.copy()
    df_raw['_smoothed_direction'] = smoothed if '_direction' in df_w.columns else 'unknown'
    df_raw.to_csv(raw_path, index=False)
    
    # write the summary csv
    if rows:
        fieldnames = rows[0].keys()
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    return output_path, raw_path


# video processing
def _detect_subject_orientation(video_path):
    mp, mp_python, PoseLandmarker, PoseLandmarkerOptions, RunningMode = get_mediapipe_bindings()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    sample_count = min(10, total)

    # use a temporary image mode landmarker for orientation checks
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

        # compare shoulder and hip midpoint offsets
        sx = (ls.x + rs.x) / 2
        sy = (ls.y + rs.y) / 2
        hx = (lh.x + rh.x) / 2
        hy = (lh.y + rh.y) / 2

        dx = abs(hx - sx)
        dy = abs(hy - sy)

        detections += 1
        if dy > dx:
            vertical_votes += 1  # person appears upright

    cap.release()
    landmarker.close()

    if detections == 0:
        return False  # leave the frame as is if orientation is unclear

    # use a simple majority vote across sampled frames
    return vertical_votes > detections / 2


def _save_jitter_log(world_landmarks_list, video_path):
    import csv
    from datetime import datetime
    
    output_dir = os.path.expanduser('~/Desktop/Gait_Analysis')
    os.makedirs(output_dir, exist_ok=True)
    vid_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_path = os.path.join(output_dir, f'jitter_log_{vid_name}_{timestamp}.csv')
    
    # leg joint indices in mediapipe
    left_hip_idx, right_hip_idx = 23, 24
    left_knee_idx, right_knee_idx = 25, 26
    left_ankle_idx, right_ankle_idx = 27, 28
    
    joint_indices = {
        'left_hip': left_hip_idx,
        'right_hip': right_hip_idx,
        'left_knee': left_knee_idx,
        'right_knee': right_knee_idx,
        'left_ankle': left_ankle_idx,
        'right_ankle': right_ankle_idx,
    }
    
    rows = []
    prev_positions = {}  # track previous frame positions for displacement calculation
    
    for frame_num, world_lm in enumerate(world_landmarks_list, start=1):
        if world_lm is None or len(world_lm) == 0:
            continue
        
        row = {'frame_num': frame_num}
        
        # extract and log each leg joint position
        for joint_name, idx in joint_indices.items():
            if idx < len(world_lm):
                lm = world_lm[idx]
                row[f'{joint_name}_x'] = float(lm.x)
                row[f'{joint_name}_y'] = float(lm.y)
                row[f'{joint_name}_z'] = float(lm.z)
                row[f'{joint_name}_visibility'] = float(lm.visibility)
                
                # calculate frame-to-frame displacement
                if joint_name in prev_positions:
                    prev_x, prev_y, prev_z = prev_positions[joint_name]
                    displacement = np.sqrt(
                        (lm.x - prev_x)**2 + 
                        (lm.y - prev_y)**2 + 
                        (lm.z - prev_z)**2
                    )
                    row[f'{joint_name}_displacement'] = float(displacement)
                else:
                    row[f'{joint_name}_displacement'] = 0.0
                
                prev_positions[joint_name] = (lm.x, lm.y, lm.z)
        
        rows.append(row)
    
    # write to CSV
    if rows:
        fieldnames = ['frame_num']
        for joint_name in joint_indices.keys():
            fieldnames.extend([
                f'{joint_name}_x', f'{joint_name}_y', f'{joint_name}_z',
                f'{joint_name}_visibility', f'{joint_name}_displacement'
            ])
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"jitter log saved: {output_path}")
        return output_path
    
    return None


def process_video(video_path, ann_dir, progress_cb, status_cb,
                  target_output_size=None, needs_rotation=None,
                  crop_rect=None, cache_key=None, cache_meta=None):
    mp, mp_python, PoseLandmarker, PoseLandmarkerOptions, RunningMode = get_mediapipe_bindings()

    cached_markup = _load_cached_markup(cache_key) if cache_key else None

    if cache_key:
        cached = _load_cached_video_result(cache_key)
        if cached is not None:
            # Check if cached data is missing confidence_data; if so, reprocess to add it
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

    # detect black borders for cropping
    if crop_rect is None:
        crop_rect = detect_crop_region(video_path, needs_rotation)
    
    # calculate the cropped frame size
    cropped_size = None
    if crop_rect:
        cx, cy, cw_r, ch_r = crop_rect
        cropped_size = (cw_r, ch_r)
        status_cb(f"Cropping black borders: {cx},{cy} {cw_r}x{ch_r}")
    else:
        status_cb("No significant black borders detected")
    
    # default the target size to the cropped size
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
    world_landmarks_list = []  # for jitter debugging
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

        # rotate upright captures into the expected analysis orientation
        if needs_rotation:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # remove detected black borders
        if crop_rect:
            cx, cy, cw_r, ch_r = crop_rect
            frame = frame[cy:cy+ch_r, cx:cx+cw_r]

        # keep an unpadded frame copy for the viewer
        view_frame = frame

        # pad to a shared output size so landmarks stay comparable
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
            
            # collect world landmarks for jitter debugging
            world_landmarks_list.append(world_lm)

            # save the clean frame so the viewer has no black bars
            raw = view_frame.copy()
            h, w = raw.shape[:2]
            if SAVE_HEIGHT and h > SAVE_HEIGHT:
                nw = int(w * SAVE_HEIGHT / h)
                raw = cv2.resize(raw, (nw, SAVE_HEIGHT), interpolation=cv2.INTER_AREA)
            raw_path = os.path.join(frame_output_dir, f"raw_{frame_count:06d}.jpg")
            cv2.imwrite(raw_path, raw, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

            # map padded landmark coordinates back into view frame space
            view_w, view_h = view_frame.shape[1], view_frame.shape[0]
            def _remap_lm(lm, _pl=pad_left, _pt=pad_top, _pfw=pad_fw, _pfh=pad_fh,
                          _vw=view_w, _vh=view_h):
                nx = (lm.x * _pfw - _pl) / _vw if _vw > 0 else lm.x
                ny = (lm.y * _pfh - _pt) / _vh if _vh > 0 else lm.y
                return SimpleLandmark(nx, ny, lm.visibility)
            remapped_pixel_lm = [_remap_lm(lm) for lm in pixel_lm]
            landmarks.append((raw_path, remapped_pixel_lm))

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

            # store skeleton landmark coordinates for filtering to reduce jitter/swapping
            for i, landmark in enumerate(world_lm):
                w_row[f'landmark_{i}_x'] = float(landmark.x)
                w_row[f'landmark_{i}_y'] = float(landmark.y)
                w_row[f'landmark_{i}_z'] = float(landmark.z)

            for i, landmark in enumerate(remapped_pixel_lm):
                p_row[f'landmark_{i}_x'] = float(landmark.x)
                p_row[f'landmark_{i}_y'] = float(landmark.y)

            # keep world landmark depth values for later analysis
            depth_row = {'frame_num': frame_count}
            for i, landmark in enumerate(world_lm):
                depth_row[f'joint_{i}'] = landmark.z
            landmark_depths.append(depth_row)

            # track confidence/visibility scores
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
            world_landmarks_list.append(None)
            landmark_depths.append(None)
            confidence_rows.append({'frame_num': frame_count, 'avg_confidence': 0.0})

    cap.release()
    landmarker.close()

    df_w = pd.DataFrame(world_rows)
    df_p = pd.DataFrame(pixel_rows)
    df_depths = pd.DataFrame([d for d in landmark_depths if d is not None])
    df_confidence = pd.DataFrame(confidence_rows)

    # diagnostics are noisy and slow, keep them opt-in only
    if DEBUG_DIRECTION_DIAGNOSTICS:
        try:
            diag_path, raw_path = _log_direction_diagnostics(df_w, df_p, video_path, 0)
            status_cb(f"Diagnostics saved: {os.path.basename(diag_path)}")
        except Exception as e:
            status_cb(f"Diagnostic logging failed: {e}")
    
    # log landmark positions for jitter analysis
    if DEBUG_LOG_JITTER:
        try:
            jitter_log_path = _save_jitter_log(world_landmarks_list, video_path)
            if jitter_log_path:
                status_cb(f"Jitter log saved: {os.path.basename(jitter_log_path)}")
        except Exception as e:
            status_cb(f"Jitter logging failed: {e}")

    for df in (df_w, df_p):
        for col in df.columns:
            if col not in ('frame_num', '_direction'):
                df[col] = df[col].astype(np.float32)
    
    # detect jittery frames FIRST from raw landmarks before any filtering (so we can detect real jitter)
    # threshold 0.04 in normalized coords (~4% of frame) roughly matches 15cm jitter in world coords
    jittery_frames = _detect_jittery_frames(landmarks, threshold=0.04)
    
    # interpolate across jittery frames in both world and pixel data
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
    
    # fix other jitter outliers (detect and interpolate spike frames)
    df_w = _fix_jitter_outliers(df_w, max_frame_displacement=0.20)
    df_p = _fix_jitter_outliers(df_p, max_frame_displacement=0.15)
    
    # apply butterworth low-pass filter to world coordinates only (for angle calculations in graphs)
    # skeleton coordinates (pixel_df) are NOT filtered to preserve exact visual position
    joint_cols_w = set(c for c in df_w.columns if c not in ('frame_num', '_direction'))
    
    for col in joint_cols_w:
        if col in df_w.columns and len(df_w) > FILTER_ORDER:
            try:
                df_w[col] = _butterworth_lowpass_filter(df_w[col].values, FILTER_CUTOFF, SLOWMO_FPS, FILTER_ORDER)
            except Exception:
                pass  # if filtering fails, keep original data
    
    # also create a filtered copy of pixel angles for graphs while keeping landmark coords unfiltered
    df_p_filtered = df_p.copy()
    pixel_angle_cols = set(c for c in df_p_filtered.columns 
                           if c not in ('frame_num', '_direction') and not c.startswith('landmark_'))
    
    for col in pixel_angle_cols:
        if col in df_p_filtered.columns and len(df_p_filtered) > FILTER_ORDER:
            try:
                df_p_filtered[col] = _butterworth_lowpass_filter(df_p_filtered[col].values, FILTER_CUTOFF, SLOWMO_FPS, FILTER_ORDER)
            except Exception:
                pass  # if filtering fails, keep original data
    
    # fix left/right limb swaps after filtering (happens when detection flips side assignment)
    # now works on cleaner, smoothed data
    df_w = _fix_limb_swaps(df_w)
    df_p = _fix_limb_swaps(df_p)
    
    # rebuild landmarks from filtered pixel coordinates for skeleton visualization
    # use filtered pixel coords (already 0-1 normalized) for correct display geometry
    filtered_landmarks = []
    for frame_idx in range(len(df_p)):
        if frame_idx < len(landmarks) and landmarks[frame_idx] is not None:
            raw_path, _ = landmarks[frame_idx]  # keep original frame path
            # rebuild landmark list from filtered df_p coordinate columns
            filtered_lm = []
            for i in range(33):  # mediapipe has 33 landmarks
                x_col = f'landmark_{i}_x'
                y_col = f'landmark_{i}_y'
                if x_col in df_p.columns and y_col in df_p.columns:
                    x = float(df_p.iloc[frame_idx][x_col]) if not pd.isna(df_p.iloc[frame_idx][x_col]) else 0.0
                    y = float(df_p.iloc[frame_idx][y_col]) if not pd.isna(df_p.iloc[frame_idx][y_col]) else 0.0
                    filtered_lm.append(SimpleLandmark(x, y, 1.0))
                else:
                    filtered_lm.append(SimpleLandmark(0.0, 0.0, 0.0))
            filtered_landmarks.append((raw_path, filtered_lm))
        elif frame_idx < len(landmarks):
            filtered_landmarks.append(landmarks[frame_idx])
    
    landmarks = filtered_landmarks
    
    # fill in gaps between jittery frames if gap is 5 or fewer frames (repeat until no more gaps)
    if jittery_frames:
        while True:
            frames_added = 0
            sorted_jittery = sorted(jittery_frames)
            for i in range(len(sorted_jittery) - 1):
                current_frame = sorted_jittery[i]
                next_frame = sorted_jittery[i + 1]
                gap = next_frame - current_frame - 1
                if gap > 0 and gap <= 5:
                    # add all frames in the gap to jittery set
                    for frame_to_fill in range(current_frame + 1, next_frame):
                        if frame_to_fill not in jittery_frames:
                            jittery_frames.add(frame_to_fill)
                            frames_added += 1
            # stop looping if no new frames were added
            if frames_added == 0:
                break
    
    #apply same filtering to confidence data
    if 'avg_confidence' in df_confidence.columns:
        df_confidence['avg_confidence'] = df_confidence['avg_confidence'].astype(np.float32)

    # drop the helper direction column before returning data
    for df in (df_w, df_p, df_p_filtered):
        if '_direction' in df.columns:
            df.drop('_direction', axis=1, inplace=True)

    ad    = df_w if USE_WORLD_LANDMARKS else df_p_filtered
    # suggested_steps = detect_steps(ad, fps=SLOWMO_FPS)
    del world_rows, pixel_rows
    gc.collect()

    result = {
        'df_world':           df_w,
        'df_pixel':           df_p,
        'df_pixel_filtered':  df_p_filtered,
        'angle_data':         ad,
        'confidence_data':    df_confidence,
        'step_frames':        [],
        # 'suggested_step_frames': suggested_steps,
        'excluded_regions':   [],
        'all_landmarks':      landmarks,   # each item stores a raw path and pixel landmarks
        'landmark_depths':    df_depths,
        'jittery_frames':     jittery_frames,  # set of frame indices with excessive displacement
        'needs_rotation':     needs_rotation,
        '_cache_key':         cache_key,
        '_cache_meta':        cache_meta or {},
        '_cached_markup':     cached_markup,
    }

    if cache_key:
        _save_cached_video_result(cache_key, cache_meta or {}, result)

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
# metric improvement direction
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


# frame cache
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
        # support raw path plus pixel landmark tuples
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


# help text
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
    ("s",             "Toggle resample (cycle view)"),
    ("m",             "Mean curve: off → +data → only"),
    ("3–8",           "Toggle joint visibility (hip / knee / ankle L & R)"),
    ("",              None),
    ("Step Editing",  None),
    ("Space",         "Add manual step at current frame (auto-detects foot)"),
    ("Backspace/Del", "Remove nearest manual step"),
    #("r",             "Generate step suggestions (all videos)"),
    ("g",             "Toggle visibility of suggested steps"),
    ("d",             "Clear manual steps (all videos)"),
    ("",              None),
    ("Exclusions",    None),
    ("Right-click",   "Drag on graph to exclude region"),
    ("Clr Excl btn",  "Clear all excluded regions"),
    ("h / H",         "This help screen"),
]

# cache manager dialog
class CacheManagerDialog(tk.Toplevel):
    
    def __init__(self, parent, cache_root):
        super().__init__(parent)
        self.title("Cache Manager")
        self.geometry("380x450")
        self.cache_root = cache_root
        self.checkboxes = {}  # map (cache_key, item_name) to (var, path)
        self.delete_whole_vars = {}  # map cache_key to delete-whole checkbox var
        
        self._build_ui()
        self._scan_caches()
    
    def _build_ui(self):
        # Canvas with scrollbar
        canvas_frame = tk.Frame(self, bg=BG)
        canvas_frame.pack(fill='both', expand=True, padx=10, pady=(10, 0))
        
        canvas = tk.Canvas(canvas_frame, bg=BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient='vertical', command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=BG)
        scrollable.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=scrollable, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        self.scrollable_frame = scrollable
        self.canvas = canvas
        
        # Bottom button frame
        bottom_frame = tk.Frame(self, bg=BG)
        bottom_frame.pack(fill='x', padx=10, pady=10)
        
        self.delete_btn = tk.Button(bottom_frame, text="Delete Selected", 
                                   font=("Helvetica", 10), bg='#e74c3c', fg='white',
                                   command=self._delete_selected, state='disabled')
        self.delete_btn.pack(side='left')
    
    def _scan_caches(self):
        """Scan cache directory and populate the list."""
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
        """Add a cache entry to the list."""
        cache_path = os.path.join(self.cache_root, cache_key)
        
        # Get metadata
        meta = {}
        manifest_path = os.path.join(cache_path, 'manifest.json')
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, 'r') as f:
                    meta = json.load(f)
            except:
                pass
        
        # Get video name from metadata, fall back to cache key
        video_name = meta.get('video_name', f"Unknown - {cache_key[:16]}...")
        
        # Check what data exists
        result_pkl = os.path.join(cache_path, 'result.pkl')
        markup_json = os.path.join(cache_path, 'markup.json')
        frames_dir = os.path.join(cache_path, 'frames')
        
        has_result = os.path.exists(result_pkl)
        has_markup = os.path.exists(markup_json)
        has_frames = os.path.isdir(frames_dir)
        
        # Get cache size
        total_size = 0
        try:
            for root, dirs, files in os.walk(cache_path):
                for f in files:
                    total_size += os.path.getsize(os.path.join(root, f))
        except:
            pass
        
        size_str = f"{total_size / (1024*1024):.1f} MB" if total_size > 0 else "0 KB"
        
        # Container for this video entry
        entry_frame = tk.Frame(self.scrollable_frame, bg=BG2, relief='solid', borderwidth=1)
        entry_frame.pack(fill='x', pady=6)
        
        # Header frame with delete whole cache checkbox
        header_frame = tk.Frame(entry_frame, bg=BG2)
        header_frame.pack(fill='x', padx=8, pady=6)
        
        # Delete whole cache checkbox on the right
        delete_whole_var = tk.BooleanVar(value=False)
        self.delete_whole_vars[cache_key] = (delete_whole_var, cache_path)
        
        delete_whole_cb = tk.Checkbutton(header_frame, text="", variable=delete_whole_var,
                                        font=("Helvetica", 9), bg=BG2, fg=TEXT,
                                        command=self._update_delete_button)
        delete_whole_cb.pack(side='right', padx=4)
        
        # Video name
        tk.Label(header_frame, text=f"📹 {video_name}", 
                font=("Helvetica", 10, "bold"), bg=BG2, fg=TEXT).pack(side='left', fill='x', expand=True)
        
        # Size
        tk.Label(header_frame, text=size_str,
                font=("Helvetica", 8), bg=BG2, fg=SUBTEXT).pack(side='left', padx=4)
        
        # Items frame (always visible)
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
        """Enable/disable delete button based on checkbox state."""
        has_items_selected = any(var.get() for var, _ in self.checkboxes.values())
        has_whole_selected = any(var.get() for var, _ in self.delete_whole_vars.values())
        self.delete_btn.config(state='normal' if (has_items_selected or has_whole_selected) else 'disabled')
    
    def _delete_selected(self):
        """Delete all selected cache items and whole caches."""
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
        
        # Delete individual items
        for (cache_key, item_name), item_path in selected_items:
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            except Exception as e:
                failed.append(f"{item_name}: {e}")
        
        # Delete whole caches
        for cache_key, cache_path in selected_whole:
            try:
                if os.path.exists(cache_path):
                    shutil.rmtree(cache_path)
            except Exception as e:
                failed.append(f"Cache {cache_key[:8]}...: {e}")
        
        if failed:
            messagebox.showerror("Partial Failure", 
                               f"Failed to delete:\n" + "\n".join(failed), parent=self)
        else:
            messagebox.showinfo("Success", f"{total_count} item(s) deleted", parent=self)
        
        # Refresh the UI by clearing and rebuilding
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        self.checkboxes = {}
        self.delete_whole_vars = {}
        self._scan_caches()


# settings dialog
class SettingsDialog(tk.Toplevel):
    """Settings dialog for confidence display and cache management."""
    
    def __init__(self, parent, dashboard):
        super().__init__(parent)
        self.title("Settings")
        self.geometry("550x700")
        self.dashboard = dashboard
        self.configure(bg=BG)
        
        self._build_ui()
    
    def _build_ui(self):
        # Title
        title_frame = tk.Frame(self, bg=BG2, height=50)
        title_frame.pack(fill='x', padx=0, pady=0)
        tk.Label(title_frame, text="Settings",
                font=("Helvetica", 12, "bold"), bg=BG2, fg=TEXT).pack(side='left', padx=10, pady=8)
        
        # Content frame
        content = tk.Frame(self, bg=BG)
        content.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Toggle confidence option
        conf_frame = tk.Frame(content, bg=BG)
        conf_frame.pack(fill='x', pady=10)
        
        conf_label = tk.Label(conf_frame, text="Confidence Scores", 
                             font=("Helvetica", 10), bg=BG, fg=TEXT)
        conf_label.pack(side='left')
        
        conf_info = tk.Label(conf_frame, text="Toggle with Alt+C", 
                            font=("Helvetica", 8), bg=BG, fg=SUBTEXT)
        conf_info.pack(side='left', padx=(10, 0))
        
        self.conf_var = tk.BooleanVar(value=self.dashboard.show_confidence)
        conf_check = tk.Checkbutton(conf_frame, text="Show confidence scores on graph",
                                   variable=self.conf_var, bg=BG, fg=TEXT,
                                   activebackground=BG, activeforeground=TEXT,
                                   command=self._toggle_confidence)
        conf_check.pack(side='right')
        
        # Separator
        sep = tk.Frame(content, bg=BG2, height=1)
        sep.pack(fill='x', pady=10)
        
        # RMSE threshold option
        rmse_frame = tk.Frame(content, bg=BG)
        rmse_frame.pack(fill='x', pady=10)
        
        rmse_label = tk.Label(rmse_frame, text="RMSE Threshold (%)", 
                             font=("Helvetica", 10), bg=BG, fg=TEXT)
        rmse_label.pack(side='left')
        
        rmse_info = tk.Label(rmse_frame, text="Lower = stricter filtering", 
                            font=("Helvetica", 8), bg=BG, fg=SUBTEXT)
        rmse_info.pack(side='left', padx=(10, 0))
        
        self.rmse_var = tk.DoubleVar(value=self.dashboard.rmse_threshold)
        self.rmse_slider = tk.Scale(
            rmse_frame,
            from_=0.0,
            to=100.0,
            resolution=0.5,
            variable=self.rmse_var,
            orient='horizontal',
            bg=BG, fg=TEXT, highlightthickness=0,
            troughcolor=BG3,
            command=self._on_rmse_change
        )
        self.rmse_slider.pack(side='right', fill='x', expand=True, padx=(10, 0))
        
        # Separator
        sep = tk.Frame(content, bg=BG2, height=1)
        sep.pack(fill='x', pady=10)
        
        # Cache management option
        cache_frame = tk.Frame(content, bg=BG)
        cache_frame.pack(fill='x', pady=10)
        
        cache_label = tk.Label(cache_frame, text="Cache Management", 
                              font=("Helvetica", 10), bg=BG, fg=TEXT)
        cache_label.pack(side='left')
        
        tk.Button(cache_frame, text="Manage Cache", font=("Helvetica", 9),
                 bg=BG3, fg=TEXT, relief='flat', padx=8,
                 command=self._open_cache_manager).pack(side='right')
        
        # Separator
        sep2 = tk.Frame(content, bg=BG2, height=1)
        sep2.pack(fill='x', pady=10)
        
        # PDF export option
        export_frame = tk.Frame(content, bg=BG)
        export_frame.pack(fill='x', pady=10)
        
        export_label = tk.Label(export_frame, text="Export Report", 
                               font=("Helvetica", 10), bg=BG, fg=TEXT)
        export_label.pack(side='left')
        
        tk.Button(export_frame, text="Print to PDF", font=("Helvetica", 9),
                 bg='#27ae60', fg=TEXT, relief='flat', padx=8,
                 command=self._open_pdf_export).pack(side='right')
        
        # Separator
        sep3 = tk.Frame(content, bg=BG2, height=1)
        sep3.pack(fill='x', pady=10)
        
        # Jitter Frames options
        jitter_frame = tk.Frame(content, bg=BG)
        jitter_frame.pack(fill='x', pady=10)
        
        jitter_label = tk.Label(jitter_frame, text="Jitter Frames", 
                               font=("Helvetica", 10, "bold"), bg=BG, fg=TEXT)
        jitter_label.pack(side='left')
        
        self.remove_jitter_var = tk.BooleanVar(value=self.dashboard.remove_jitter_frames)
        remove_jitter_check = tk.Checkbutton(
            content, text="Remove jitter frames",
            variable=self.remove_jitter_var, bg=BG, fg=TEXT,
            activebackground=BG, activeforeground=TEXT,
            command=self._toggle_remove_jitter_frames)
        remove_jitter_check.pack(anchor='w', pady=2)
        
        self.show_jitter_var = tk.BooleanVar(value=self.dashboard.show_jitter_frames)
        show_jitter_check = tk.Checkbutton(
            content, text="Show jitter frames in red",
            variable=self.show_jitter_var, bg=BG, fg=TEXT,
            activebackground=BG, activeforeground=TEXT,
            command=self._toggle_show_jitter_frames)
        show_jitter_check.pack(anchor='w', pady=(2, 10))
        
        # Separator
        sep4 = tk.Frame(content, bg=BG2, height=1)
        sep4.pack(fill='x', pady=10)
        
        # Graph viewer option
        graph_frame = tk.Frame(content, bg=BG)
        graph_frame.pack(fill='x', pady=10)
        
        graph_label = tk.Label(graph_frame, text="Graph Display", 
                              font=("Helvetica", 10), bg=BG, fg=TEXT)
        graph_label.pack(side='left')
        
        self._graph_display_btn = tk.Button(graph_frame, font=("Helvetica", 9),
                 bg=BG3, fg=TEXT, relief='flat', padx=8,
                 command=self._toggle_graph_display_mode)
        self._graph_display_btn.pack(side='right')
        self._update_graph_display_btn()
        
        # Bottom buttons
        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.pack(fill='x', padx=20, pady=15)
        
        tk.Button(btn_frame, text="Close", font=("Helvetica", 10),
                 bg=BG3, fg=TEXT, relief='flat', padx=12, pady=6,
                 command=self.destroy).pack(side='right')
    
    def _toggle_confidence(self):
        self.dashboard.show_confidence = self.conf_var.get()
        self.dashboard.redraw_graph()
    
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
            self.dashboard.redraw_graph()
        except Exception:
            pass
    
    def _update_graph_display_btn(self):
        # update button text to show current mode
        mode_text = "SE Shading" if self.dashboard.graph_display_mode == 'se_shading' else "Lines"
        self._graph_display_btn.config(text=mode_text)
    
    def _toggle_graph_display_mode(self):
        # toggle between se_shading and lines_only modes
        if self.dashboard.graph_display_mode == 'se_shading':
            self.dashboard.graph_display_mode = 'lines_only'
            self.dashboard.show_data = True
        else:
            self.dashboard.graph_display_mode = 'se_shading'
            self.dashboard.show_data = False
        self._update_graph_display_btn()
        self.dashboard.redraw_graph()
    
    def _open_cache_manager(self):
        if self.dashboard._cache_manager_dialog is not None and self.dashboard._cache_manager_dialog.winfo_exists():
            self.dashboard._cache_manager_dialog.lift()
            self.dashboard._cache_manager_dialog.focus()
        else:
            dialog = CacheManagerDialog(self, _cache_root_dir())
            self.dashboard._cache_manager_dialog = dialog
            # when the dialog is destroyed, clear the reference
            def on_close():
                self.dashboard._cache_manager_dialog = None
                dialog.destroy()
            dialog.protocol("WM_DELETE_WINDOW", on_close)
    
    def _open_pdf_export(self):
        """Open the PDF export dialog (singleton pattern)."""
        if self.dashboard._pdf_export_dialog is not None and self.dashboard._pdf_export_dialog.winfo_exists():
            self.dashboard._pdf_export_dialog.lift()
            self.dashboard._pdf_export_dialog.focus()
        else:
            dialog = PDFExportDialog(self, self.dashboard)
            self.dashboard._pdf_export_dialog = dialog
            # when the dialog is destroyed, clear the reference
            def on_close():
                self.dashboard._pdf_export_dialog = None
                dialog.destroy()
            dialog.protocol("WM_DELETE_WINDOW", on_close)

# PDF export dialog
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
        self.geometry("550x900")
        self.dashboard = dashboard
        self.configure(bg=BG)
        
        self._build_ui()
    
    def _build_ui(self):
        # Title
        title_frame = tk.Frame(self, bg=BG2, height=50)
        title_frame.pack(fill='x', padx=0, pady=0)
        tk.Label(title_frame, text="Export Gait Analysis Report",
                font=("Helvetica", 12, "bold"), bg=BG2, fg=TEXT).pack(side='left', padx=10, pady=8)
        
        # Content frame with scrollbar
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
        
        # Graphs section
        graphs_frame = tk.LabelFrame(scrollable, text="Select Graphs & Options", bg=BG, fg=TEXT,
                                    font=("Helvetica", 10, "bold"))
        graphs_frame.pack(fill='x', pady=(10, 5))
        
        self.graph_vars = {}
        self.graph_vars['continuous'] = tk.BooleanVar(value=True)
        self.graph_vars['cycles'] = tk.BooleanVar(value=True)
        self.graph_options = {}
        
        # Continuous graph options
        continuous_check = tk.Checkbutton(graphs_frame, text="Continuous View (Raw angle data)",
                      variable=self.graph_vars['continuous'], bg=BG, fg=TEXT,
                      activebackground=BG, activeforeground=TEXT)
        continuous_check.pack(anchor='w', padx=10, pady=4)
        
        self.graph_options['continuous'] = self._create_graph_options_frame(graphs_frame, 'continuous')
        
        # Cycles graph options
        cycles_check = tk.Checkbutton(graphs_frame, text="Overlaid Cycles (Normalized gait cycles)",
                      variable=self.graph_vars['cycles'], bg=BG, fg=TEXT,
                      activebackground=BG, activeforeground=TEXT)
        cycles_check.pack(anchor='w', padx=10, pady=4)
        
        self.graph_options['cycles'] = self._create_graph_options_frame(graphs_frame, 'cycles')
        
        # Limbs section
        limbs_frame = tk.LabelFrame(scrollable, text="Select Limbs", bg=BG, fg=TEXT,
                                   font=("Helvetica", 10, "bold"))
        limbs_frame.pack(fill='x', pady=5)
        
        self.limb_vars = {}
        limbs = [
            ('left_hip', 'Left Hip'),
            ('left_knee', 'Left Knee'),
            ('left_ankle', 'Left Ankle'),
            ('right_hip', 'Right Hip'),
            ('right_knee', 'Right Knee'),
            ('right_ankle', 'Right Ankle'),
        ]
        
        for key, label in limbs:
            self.limb_vars[key] = tk.BooleanVar(value=True)
            tk.Checkbutton(limbs_frame, text=label,
                          variable=self.limb_vars[key], bg=BG, fg=TEXT,
                          activebackground=BG, activeforeground=TEXT).pack(anchor='w', padx=10, pady=2)
        
        # Outcome measures section
        measures_frame = tk.LabelFrame(scrollable, text="Outcome Measures", bg=BG, fg=TEXT,
                                      font=("Helvetica", 10, "bold"))
        measures_frame.pack(fill='x', pady=5)
        
        self.measure_vars = {}
        for key, label in self.OUTCOME_MEASURES.items():
            self.measure_vars[key] = tk.BooleanVar(value=True)
            tk.Checkbutton(measures_frame, text=label,
                          variable=self.measure_vars[key], bg=BG, fg=TEXT,
                          activebackground=BG, activeforeground=TEXT).pack(anchor='w', padx=10, pady=2)
        
        # Info text
        info_frame = tk.Frame(scrollable, bg=BG)
        info_frame.pack(fill='x', pady=10)
        
        tk.Label(info_frame, text="Note: PDF will be generated for currently loaded video pair.",
                font=("Helvetica", 9), bg=BG, fg=SUBTEXT, wraplength=400, justify='left').pack(anchor='w', padx=10)
        
        # Bottom buttons
        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.pack(fill='x', padx=10, pady=15)
        
        tk.Button(btn_frame, text="Print to PDF", font=("Helvetica", 11),
                 bg='#27ae60', fg=TEXT, relief='flat', padx=15, pady=8,
                 command=self._export_pdf).pack(side='right', padx=4)
        
        tk.Button(btn_frame, text="Cancel", font=("Helvetica", 10),
                 bg=BG3, fg=TEXT, relief='flat', padx=12, pady=6,
                 command=self.destroy).pack(side='right', padx=4)
    
    def _create_graph_options_frame(self, parent, graph_type):
        """Create graph options frame for V1/V2/Both and excluded areas toggle."""
        options_frame = tk.Frame(parent, bg=BG)
        options_frame.pack(fill='x', padx=30, pady=2)
        
        # Version selection
        version_var = tk.StringVar(value='both')
        self.graph_options[f'{graph_type}_version'] = version_var
        
        tk.Label(options_frame, text="Show:", bg=BG, fg=TEXT, font=("Helvetica", 9)).pack(side='left', padx=(0, 5))
        tk.Radiobutton(options_frame, text="V1", variable=version_var, value='v1',
                      bg=BG, fg=TEXT, activebackground=BG, activeforeground=TEXT).pack(side='left', padx=3)
        tk.Radiobutton(options_frame, text="V2", variable=version_var, value='v2',
                      bg=BG, fg=TEXT, activebackground=BG, activeforeground=TEXT).pack(side='left', padx=3)
        tk.Radiobutton(options_frame, text="Both", variable=version_var, value='both',
                      bg=BG, fg=TEXT, activebackground=BG, activeforeground=TEXT).pack(side='left', padx=3)
        
        # Excluded areas toggle
        excluded_var = tk.BooleanVar(value=True)
        self.graph_options[f'{graph_type}_excluded'] = excluded_var
        tk.Checkbutton(options_frame, text="Show excluded areas", variable=excluded_var,
                      bg=BG, fg=TEXT, activebackground=BG, activeforeground=TEXT,
                      font=("Helvetica", 9)).pack(side='left', padx=10)
        
        return options_frame
    
    def _export_pdf(self):
        if not HAS_REPORTLAB:
            messagebox.showerror("Missing Dependency", 
                               "reportlab is required for PDF export.\n\n"
                               "Install with: pip install reportlab")
            return
        
        if len(self.dashboard.datasets) < 2:
            messagebox.showwarning("Not Enough Videos", 
                                 "Comparison reports require 2 videos.")
            return
        
        # Get file path
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")],
            initialfile=f"gait_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        
        if not file_path:
            return
        
        try:
            # Build graphs dict with full options
            graphs = {}
            for graph_type in ['continuous', 'cycles']:
                if self.graph_vars[graph_type].get():
                    graphs[graph_type] = {
                        'show_versions': self.graph_options[f'{graph_type}_version'].get(),
                        'include_excluded': self.graph_options[f'{graph_type}_excluded'].get(),
                    }
            
            # Build limbs dict
            limbs = {k: v.get() for k, v in self.limb_vars.items()}
            
            self.dashboard._generate_pdf(
                file_path,
                graphs=graphs,
                limbs=limbs,
                measures={k: v.get() for k, v in self.measure_vars.items()}
            )
            messagebox.showinfo("Success", f"PDF exported to:\n{file_path}")
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate PDF:\n{str(e)}")

# dashboard
class GaitAnalysisDashboard(tk.Tk):

    def __init__(self):
        super().__init__()
        self.configure(bg=BG)
        self.title("Gait Analysis")
        self.geometry("1400x860")

        # ui state
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
        self.graph_display_mode   = 'se_shading'  # 'se_shading' or 'lines_only'
        self.show_confidence      = False  # confidence scores hidden by default
        self.show_outliers_only   = False  # toggle to show only outlier cycles
        self.show_jitter_frames   = False  # toggle to show jitter frames in red instead of hiding them
        self.remove_jitter_frames = True   # toggle to enable/disable jitter frame removal entirely
        self.active_dataset_idx   = 0
        ui_settings = _load_ui_settings()
        self.skeleton_thickness = float(ui_settings.get('skeleton_thickness', DRAW_THICKNESS))
        self.skeleton_thickness = max(0.0, min(float(DRAW_THICKNESS), self.skeleton_thickness))
        self.rmse_threshold = float(ui_settings.get('rmse_threshold', 35.0))
        self.rmse_threshold = max(0.0, min(100.0, self.rmse_threshold))
        self.manual_step_mode     = True  # always active
        self.manual_side          = 'right'  # kept for older paths
        self.show_suggestions     = False
        self.playing              = False
        self._marking_phase       = None   # none, left, or right
        self._marking_video_idx   = 0      # active video during markup
        self._markup_frame        = None   # created on first use
        self._play_after_id       = None
        self._graph_dragging      = False
        self._exclusion_selecting = False
        self._exclusion_start     = None

        self._cache     = FrameCache()
        self._stop_pf   = False
        self._pf_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._pf_thread.start()

        # singleton window references
        self._settings_dialog = None
        self._cache_manager_dialog = None
        self._pdf_export_dialog = None

        self._build_ui()
        self._bind_keys()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ui
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
                 font=("Coiny Cyrillic", 17), bg=BG2, fg=ACCENT,
                 cursor="hand2")
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

        self._v2_lbl = tk.Label(hdr, text="Video 2: —",
                                font=("Helvetica", 10, "bold"), bg=BG2, fg=C_V2)
        self._v2_lbl.pack(side='right', padx=4)
        self._v1_lbl = tk.Label(hdr, text="Video 1: —",
                                font=("Helvetica", 10, "bold"), bg=BG2, fg=C_V1)
        self._v1_lbl.pack(side='right', padx=18)

        strike_key = tk.Frame(hdr, bg=BG2)
        strike_key.pack(side='right', padx=(4, 10))
        for colour, label in ((C_RIGHT, "Right strike"), (C_LEFT, "Left strike")):
            row = tk.Frame(strike_key, bg=BG2)
            row.pack(side='left', padx=4)
            tk.Canvas(row, width=10, height=10, bg=colour,
                      highlightthickness=0).pack(side='left', pady=1)
            tk.Label(row, text=f" {label}", font=("Helvetica", 8),
                     bg=BG2, fg=SUBTEXT).pack(side='left')


        # main layout with graph and videos on the left
        main = tk.Frame(self, bg=BG)
        main.pack(fill='both', expand=True, padx=8, pady=(4, 0))
        self._main_content = main

        left = tk.Frame(main, bg=BG)
        left.pack(side='left', fill='both', expand=True)

        # graph row with the legend docked on the right
        graph_row = tk.Frame(left, bg=BG2)
        graph_row.pack(fill='x')

        gf = tk.Frame(graph_row, bg=BG2)
        gf.pack(side='left', fill='both', expand=True)

        # add a horizontal scrollbar under the graph
        self._graph_hbar = tk.Scrollbar(gf, orient='horizontal', command=self._on_scrollbar_drag)
        self._graph_hbar.pack(fill='x', side='bottom')

        self._fig, self._ax = plt.subplots(figsize=(11, 4.2), dpi=100)
        self._ax_xlim_full = None # filled after data loads
        self._ax_xlim_per_mode = {}  # keep zoom limits per graph mode
        self._zoom_level = 1.0
        self._last_scroll_event = None  # keep the last wheel event for cursor zoom
        self._fig.patch.set_facecolor(BG)
        self._ax.set_facecolor(BG_INIT)
        # match the startup styling to redraw_graph
        for spine in self._ax.spines.values():
            spine.set_color(BG2)
        self._ax.tick_params(colors=SUBTEXT, labelsize=9)
        self._mpl_canvas = FigureCanvasTkAgg(self._fig, master=gf)
        self._mpl_canvas.get_tk_widget().pack(fill='x')
        self._mpl_canvas.mpl_connect('button_press_event',   self._on_graph_click)
        self._mpl_canvas.mpl_connect('motion_notify_event',  self._on_graph_drag)
        self._mpl_canvas.mpl_connect('button_release_event', self._on_graph_release)

        # interactive legend panel
        self._legend_frame = tk.Frame(graph_row, bg=BG2, width=140)
        self._legend_frame.pack(side='right', fill='y', padx=(2, 0))
        self._legend_frame.pack_propagate(False)

        self._legend_bottom = tk.Frame(self._legend_frame, bg=BG2)
        self._legend_bottom.pack(side='bottom', fill='x', padx=4, pady=(4, 8))

        # v1 and v2 toggle buttons
        ind_frame = tk.Frame(self._legend_frame, bg=BG2)
        ind_frame.pack(fill='x', padx=6, pady=(8, 2))

        # v1 button with dashed swatch
        v1_btn_frame = tk.Frame(ind_frame, bg=BG3, cursor='hand2', relief='flat', bd=0)
        v1_btn_frame.pack(side='left', fill='x', expand=True, padx=(0, 3))
        c1 = tk.Canvas(v1_btn_frame, width=24, height=14, bg=BG3, highlightthickness=0)
        c1.pack(side='left', padx=(4, 1), pady=2)
        c1.create_line(2, 7, 22, 7, fill=C_V1, dash=(4, 3), width=2)
        self._v1_toggle_lbl = tk.Label(v1_btn_frame, text="V1", font=("Helvetica", 8, "bold"),
                                        bg=BG3, fg=C_V1, anchor='w')
        self._v1_toggle_lbl.pack(side='left', padx=(1, 4))
        self._v1_btn_frame = v1_btn_frame
        self._v1_swatch = c1
        for w in (v1_btn_frame, c1, self._v1_toggle_lbl):
            w.bind('<Button-1>', lambda e: self._toggle_video_view(0))

        # v2 button with solid swatch
        v2_btn_frame = tk.Frame(ind_frame, bg=BG3, cursor='hand2', relief='flat', bd=0)
        v2_btn_frame.pack(side='left', fill='x', expand=True, padx=(3, 0))
        c2 = tk.Canvas(v2_btn_frame, width=24, height=14, bg=BG3, highlightthickness=0)
        c2.pack(side='left', padx=(4, 1), pady=2)
        c2.create_line(2, 7, 22, 7, fill=C_V2, width=2)
        self._v2_toggle_lbl = tk.Label(v2_btn_frame, text="V2", font=("Helvetica", 8, "bold"),
                                        bg=BG3, fg=C_V2, anchor='w')
        self._v2_toggle_lbl.pack(side='left', padx=(1, 4))
        self._v2_btn_frame = v2_btn_frame
        self._v2_swatch = c2
        for w in (v2_btn_frame, c2, self._v2_toggle_lbl):
            w.bind('<Button-1>', lambda e: self._toggle_video_view(1))

        ttk_sep = tk.Frame(self._legend_frame, bg=SUBTEXT, height=1)
        ttk_sep.pack(fill='x', padx=6, pady=(4, 4))

        # joint toggle entries
        self._legend_items = {}
        for joint, col in JOINT_COLORS_MPL.items():
            jf = tk.Frame(self._legend_frame, bg=BG3, cursor='hand2')
            jf.pack(fill='x', padx=4, pady=2)

            # line samples for both videos
            swatch = tk.Canvas(jf, width=36, height=18, bg=BG3, highlightthickness=0)
            swatch.pack(side='left', padx=(4, 2), pady=2)
            swatch.create_line(2, 9, 16, 9, fill=col, dash=(4, 3), width=2)  # v1 dashed
            swatch.create_line(20, 9, 34, 9, fill=col, width=2)              # v2 solid

            name = joint.replace('_', ' ').title()
            lbl = tk.Label(jf, text=name, font=("Helvetica", 8, "bold"),
                           bg=BG3, fg=TEXT, anchor='w')
            lbl.pack(side='left', fill='x', expand=True, padx=(0, 4))

            self._legend_items[joint] = {'frame': jf, 'swatch': swatch, 'label': lbl, 'color': col}

            for w in (jf, swatch, lbl):
                w.bind('<Button-1>', lambda e, j=joint: self._toggle_joint_legend(j))

        # toggle all button
        self._toggle_all_btn = tk.Button(self._legend_frame, text="Toggle All",
                    font=("Helvetica", 7, "bold"), bg=BG3, fg=TEXT,
                    relief='flat', cursor='hand2', command=self._toggle_all_joints)
        self._toggle_all_btn.pack(fill='x', padx=4, pady=(4, 2))

        # separator
        tk.Frame(self._legend_frame, bg=SUBTEXT, height=1).pack(fill='x', padx=6, pady=(4, 4))

        # display toggles for cycle view
        self._display_btns = {}
        for label, key in [("Mean", "mean"), ("Data", "data"), ("Normal", "normal"), ("Outliers", "outliers")]:
            btn = tk.Button(self._legend_frame, text=label,
                        font=("Helvetica", 7, "bold"), bg=BG3, fg=TEXT,
                        relief='flat', cursor='hand2',
                        command=lambda k=key: self._toggle_display_option(k))
            btn.pack(fill='x', padx=4, pady=1)
            self._display_btns[key] = btn

        tk.Frame(self._legend_bottom, bg=SUBTEXT, height=1).pack(fill='x', padx=2, pady=(0, 4))

        self._sidebar_toggle_btns = {}
        for label, key, cmd in [
            ("Cycles", "cycles", self._toggle_cycles),
            ("World", "world_px", self._toggle_world),
        ]:
            btn = tk.Button(self._legend_bottom, text=label,
                        font=("Helvetica", 7, "bold"), bg=BG3, fg=TEXT,
                        relief='flat', cursor='hand2', command=cmd)
            btn.pack(fill='x', pady=1)
            self._sidebar_toggle_btns[key] = btn

        tk.Frame(self._legend_bottom, bg=SUBTEXT, height=1).pack(fill='x', padx=2, pady=(4, 4))

        self._clear_btns = {}
        for label, key, cmd in [
            ("Clear Steps", "clear_steps", self._clear_steps),
            ("Clear Excluded Zone", "clear_excl", self._clear_exclusions),
        ]:
            btn = tk.Button(self._legend_bottom, text=label,
                        font=("Helvetica", 7, "bold"), bg=BG3, fg=TEXT,
                        relief='flat', cursor='hand2', command=cmd)
            btn.pack(fill='x', pady=1)
            self._clear_btns[key] = btn

        self._update_display_btn_visuals()

        # video panels
        vrow = tk.Frame(left, bg=BG)
        vrow.pack(fill='both', expand=True, pady=(4, 0))

        self._vid_canvases = []
        self._vid_labels   = []
        for i in range(2):
            cf = tk.Frame(vrow, bg=BG2, bd=1, relief='flat')
            cf.pack(side='left', fill='both', expand=True, padx=(0 if i == 0 else 4, 0))
            fg_col = C_V1 if i == 0 else C_V2
            lbl = tk.Label(cf, text=f"VIDEO {i+1}",
                           font=("Helvetica", 9, "bold"),
                           bg=BG2, fg=fg_col, anchor='w')
            lbl.pack(fill='x', padx=6, pady=(3, 0))
            self._vid_labels.append(lbl)
            c = tk.Canvas(cf, bg=BG_VID, highlightthickness=0)
            c.pack(fill='both', expand=True)
            self._vid_canvases.append(c)

        # metrics panel
        right = tk.Frame(main, bg=BG2, width=215)
        right.pack(side='right', fill='y', padx=(6, 0))
        right.pack_propagate(False)

        tk.Label(right, text="METRICS",
                 font=("Helvetica", 10, "bold"), bg=BG2, fg=ACCENT
                 ).pack(pady=(10, 4), padx=8)

        self._metric_value_lbls = {}
        for key in METRIC_ORDER:
            label, sub = METRIC_LABELS[key]
            card = tk.Frame(right, bg=BG3, padx=8, pady=5)
            card.pack(fill='x', padx=8, pady=3)
            tk.Label(card, text=label, font=("Helvetica", 9, "bold"),
                     bg=BG3, fg=TEXT).pack(anchor='w')
            tk.Label(card, text=sub, font=("Helvetica", 7),
                     bg=BG3, fg=SUBTEXT).pack(anchor='w')
            val_lbl = tk.Label(card, text="—",
                               font=("Helvetica", 13, "bold"),
                               bg=BG3, fg=SUBTEXT)
            val_lbl.pack(anchor='w', pady=(2, 0))
            self._metric_value_lbls[key] = val_lbl

        tk.Frame(right, bg=SUBTEXT, height=1).pack(fill='x', padx=8, pady=(8, 6))
        tk.Label(right, text="Skeleton",
                 font=("Helvetica", 8, "bold"), bg=BG2, fg=TEXT).pack(anchor='w', padx=10)
        self._skeleton_slider = tk.Scale(
            right,
            from_=0,
            to=DRAW_THICKNESS,
            orient='horizontal',
            resolution=0.5,
            length=175,
            bg=BG2,
            fg=TEXT,
            highlightthickness=0,
            troughcolor=BG3,
            command=self._on_skeleton_thickness_change,
        )
        self._skeleton_slider.set(self.skeleton_thickness)
        self._skeleton_slider.pack(fill='x', padx=10, pady=(0, 10))

        # toolbar and status bar
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

        buttons = [
            ("Prev",         self._prev_frame),
            ("Next",         self._next_frame),
            ("Play",         self._toggle_play),
            ("Auto steps",   self._recompute_steps),
        ]
        for txt, cmd in buttons:
            tk.Button(bar, text=txt, command=cmd, **btn_cfg).pack(side='left', padx=2, pady=3)

        self._frame_lbl = tk.Label(bar, text="Frame: —",
                                   font=("Helvetica", 8), bg=BG2, fg=SUBTEXT)
        self._frame_lbl.pack(side='right', padx=8)

        tk.Label(bar, textvariable=self._status_msg,
                 font=("Helvetica", 8), bg=BG2, fg=TEXT, anchor='w'
                 ).pack(side='left', padx=8)

    # video selection
    def find_videos(self):
        video_paths = list(select_video_paths())
        if len(video_paths) == 0:
            return
        if len(video_paths) == 1:
            second = filedialog.askopenfilename(
                title="Select second video",
                parent=self,
                filetypes=[("Video files", "*.mov *.mp4 *.avi *.m4v"), ("All files", "*.*")])
            if not second:
                messagebox.showwarning("Two videos required",
                                       "Please select two videos to run the analysis.",
                                       parent=self)
                return
            video_paths.append(second)
        video_paths = video_paths[:2]

        self.video_names = [os.path.splitext(os.path.basename(p))[0] for p in video_paths]
        self._v1_lbl.config(text=f"Video 1: {self.video_names[0]}")
        self._v2_lbl.config(text=f"Video 2: {self.video_names[1]}")

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
                    'needs_rotation': needs_rot,
                    'crop_rect': crop_rect,
                    'dims': dims,
                    'video_meta': _video_metadata(path),
                    'video_sha256': _file_sha256(path),
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
                    'schema': CACHE_SCHEMA_VERSION,
                    'video_path': video_paths[i],
                    'video_name': video_name,
                    'video_meta': info.get('video_meta'),
                    'video_sha256': info.get('video_sha256'),
                    'target_output_size': list(target_output_size) if target_output_size else None,
                }

            self._status_msg.set("Processing videos… 0%")

            def _process(i, path, ann_dir):
                info = pre_scan[i] if isinstance(pre_scan[i], dict) else {}
                needs_rot = info.get('needs_rotation')
                crop_rect = info.get('crop_rect')

                def _prog(p):
                    results_progress[i] = p

                def _stat(s):
                    self.after(0, lambda: self._status_msg.set(s))

                results[i] = process_video(
                    path, ann_dir, _prog, _stat,
                    target_output_size=target_output_size,
                    needs_rotation=needs_rot,
                    crop_rect=crop_rect,
                    cache_key=cache_keys[i],
                    cache_meta=cache_meta[i],
                )

            t1 = threading.Thread(target=_process, args=(0, video_paths[0], spill1), daemon=True)
            t2 = threading.Thread(target=_process, args=(1, video_paths[1], spill2), daemon=True)
            t1.start()
            t2.start()
            self.after(300, _poll_loading)

        def _poll_pre_scan():
            done = sum(0 if t.is_alive() else 1 for t in scan_threads)
            self._status_msg.set(f"Preparing videos… {done}/2")
            self._update_status()
            if done < 2:
                self.after(200, _poll_pre_scan)
                return

            for i, info in enumerate(pre_scan):
                if isinstance(info, dict) and info.get('error'):
                    print(f"Warning: pre-scan failed for video {i+1}: {info['error']}")
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
            for i, lbl in enumerate(self._vid_labels):
                lbl.config(text=f"VIDEO {i+1}  —  {self.video_names[i]}")
            self.datasets           = results
            self.df_world           = results[0]['df_world']
            self.df_pixel           = results[0]['df_pixel']
            self.df_pixel_filtered  = results[0].get('df_pixel_filtered', results[0]['df_pixel'])
            self.angle_data         = results[0]['angle_data']
            self.total_frames = max(len(results[0]['all_landmarks']),
                                    len(results[1]['all_landmarks']))
            # initialize zoom limits from the shorter video
            min_video_frames = min(len(results[0]['all_landmarks']),
                                   len(results[1]['all_landmarks']))
            if not self.angle_data.empty:
                data_min = self.angle_data['frame_num'].min()
                self._ax_xlim_full = (data_min, data_min + min_video_frames)
                # store limits for each graph view
                self._ax_xlim_per_mode['both'] = self._ax_xlim_full
                for i, ds in enumerate(results[:2]):
                    v_frames = len(ds['all_landmarks'])
                    self._ax_xlim_per_mode[f'v{i+1}'] = (data_min, data_min + v_frames)
            self.progress     = 1.0
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
        # mouse wheel pans by default and zooms with ctrl
        canvas = self._mpl_canvas.get_tk_widget()
        canvas.bind('<MouseWheel>', self._on_canvas_scroll)
        canvas.bind('<Button-4>', self._on_canvas_scroll)  # linux scroll up
        canvas.bind('<Button-5>', self._on_canvas_scroll)  # linux scroll down
        self.bind('<Key-1>',        lambda e: self._prev_frame())
        self.bind('<Key-2>',        lambda e: self._next_frame())
        self.bind('<Key-9>',        lambda e: self._toggle_play())
        self.bind('<q>',            lambda e: self._on_close())
        self.bind('<w>',            lambda e: self._toggle_world())
        self.bind('<c>',            lambda e: self._toggle_cycles())
        self.bind('<s>',            lambda e: self._toggle_resample())
        self.bind('<m>',            lambda e: self._toggle_mean())
        self.bind('<v>',            lambda e: self._cycle_graph_view())
        self.bind('<t>',            lambda e: self._toggle_active())
        self.bind('<r>',            lambda e: self._recompute_steps())
        self.bind('<g>',            lambda e: self._toggle_suggestions())
        self.bind('<d>',            lambda e: self._clear_steps())
        self.bind('<space>',        lambda e: self._add_manual_step())
        self.bind('<BackSpace>',    lambda e: self._delete_nearest_step())
        self.bind('<Delete>',       lambda e: self._delete_nearest_step())
        self.bind('z',              lambda e: self._reset_zoom())
        self.bind('<Alt-c>',        lambda e: self._toggle_confidence())
        for k, jt in [('3','left_hip'),('4','right_hip'),('5','left_knee'),
                       ('6','right_knee'),('7','left_ankle'),('8','right_ankle')]:
            self.bind(k, lambda e, j=jt: self._toggle_joint(j))

    # prefetch
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
        if not ds:
            return
        cache_key = ds.get('_cache_key')
        if not cache_key:
            return
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
            if not ds:
                continue
            cached_markup = ds.get('_cached_markup') or {}
            cached_steps = cached_markup.get('step_frames', [])
            has_cached_markup = bool(cached_steps)
            if not has_cached_markup:
                continue

            vid_name = self.video_names[i] if i < len(self.video_names) else f"Video {i+1}"
            answer = messagebox.askyesno(
                "Cached steps found",
                f"Cached manual steps were found for {vid_name}.\n\n"
                "Yes = load cached steps\n"
                "No = overwrite and mark again",
                parent=self,
            )

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
        if not self.datasets:
            return 0
        total = 0
        for ds in self.datasets:
            if not ds:
                continue
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
        
        # keep only rows outside excluded regions
        mask = pd.Series([True] * len(angle_data), index=angle_data.index)
        for start_frame, end_frame in excluded_regions:
            mask &= ~((angle_data['frame_num'] >= start_frame) & (angle_data['frame_num'] < end_frame))
        
        return angle_data[mask].reset_index(drop=True)

    def _region_crosses_exclusion(self, start_frame, end_frame, excluded_regions):
        for ex_start, ex_end in excluded_regions:
            # return true when this range overlaps an exclusion
            if not (end_frame <= ex_start or start_frame >= ex_end):
                return True
        return False

    @staticmethod
    def _merge_exclusion_regions(regions):
        if len(regions) <= 1:
            return regions
        regions.sort()
        merged = [regions[0]]
        for s, e in regions[1:]:
            prev_s, prev_e = merged[-1]
            if s <= prev_e:  # merge overlapping or adjacent ranges
                merged[-1] = (prev_s, max(prev_e, e))
            else:
                merged.append((s, e))
        return merged

    # graph
    def redraw_graph(self):
        ax = self._ax
        ax.cla()
        ax.set_facecolor(BG_PLOT)
        for spine in ax.spines.values():
            spine.set_color(BG2)
        ax.tick_params(colors=SUBTEXT, labelsize=9)
        ax.xaxis.label.set_color(SUBTEXT)
        ax.yaxis.label.set_color(SUBTEXT)

        if self.angle_data is None or self.angle_data.empty:
            self._mpl_canvas.draw_idle()
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

        def _get_cycle_strikes(norm_steps):
            # return separate left and right strikes instead of choosing one
            left = [f for f, s in norm_steps if s == 'left']
            right = [f for f, s in norm_steps if s == 'right']
            return left, right
        
        def _get_limb_side(joint_name):
            # determine if joint belongs to left or right limb
            return 'left' if joint_name.startswith('left_') else 'right'

        # find the longest usable cycle for overlaid mode
        max_cycle_length = self.resample_length
        if self.show_overlaid_cycles:
            for ds in dfg:
                ad = ds['angle_data']
                excluded = ds.get('excluded_regions', [])
                ad_filtered = self._get_filtered_angle_data(ad, excluded)
                
                sf = ds.get('step_frames', [])
                norm = _to_fnums(ad_filtered, sf)
                left_strikes, right_strikes = _get_cycle_strikes(norm)
                
                # check both left and right limbs for max cycle length
                for strikes in [left_strikes, right_strikes]:
                    if len(strikes) < 2:
                        continue
                    
                    for i in range(len(strikes)-1):
                        # skip step pairs that cross excluded regions
                        if self._region_crosses_exclusion(strikes[i], strikes[i+1], excluded):
                            continue
                        seg = ad_filtered[(ad_filtered['frame_num'] >= strikes[i]) & (ad_filtered['frame_num'] <= strikes[i+1])]
                        if not seg.empty:
                            max_cycle_length = max(max_cycle_length, len(seg))
        self._current_max_cycle_length = max_cycle_length

        if not self.show_overlaid_cycles:
            ax.set_xlabel('Frame', fontsize=10)
            ax.set_ylabel('Angle (°)', fontsize=10)
            plotted = False

            for ds in dfg:
                ad  = ds['angle_data']
                excluded = ds.get('excluded_regions', [])
                ad_filtered = self._get_filtered_angle_data(ad, excluded)
                
                sf  = ds.get('step_frames', [])
                si  = _src_idx(ds)
                ls  = linestyles[si % 2]

                # Show excluded regions unless explicitly hidden (for PDF export)
                show_excluded = getattr(self, '_show_excluded_in_pdf', True)
                if show_excluded:
                    for start_frame, end_frame in excluded:
                        ax.axvspan(start_frame, end_frame, alpha=0.10, color='darkgray', zorder=1)

                # build an exclusion mask for this dataset
                frames = ad['frame_num'].values
                excl_mask = np.zeros(len(frames), dtype=bool)
                for ex_s, ex_e in excluded:
                    excl_mask |= (frames >= ex_s) & (frames < ex_e)

                for joint, col in JOINT_COLORS_MPL.items():
                    if joint not in ad.columns:
                        continue
                    vis = self.joint_visibility.get(joint, True)
                    if not vis:
                        continue
                    values = ad[joint].values.copy().astype(float)

                    if excluded and show_excluded:
                        # draw excluded segments in gray
                        gray_vals = values.copy()
                        gray_vals[~excl_mask] = np.nan
                        ax.plot(frames, gray_vals, color='#999999', lw=1.2,
                                alpha=0.5, linestyle=ls, zorder=2)

                    # draw the remaining data in the joint color
                    clean_vals = values.copy()
                    if excluded and show_excluded:
                        clean_vals[excl_mask] = np.nan
                    ax.plot(frames, clean_vals, color=col, lw=1.4,
                            alpha=0.85, linestyle=ls, zorder=3,
                            label=f"{joint.replace('_',' ').title()} V{si+1}")
                    plotted = True

                nsteps  = _to_fnums(ad_filtered, sf)
                fn_min  = int(ad['frame_num'].min())
                fn_max  = int(ad['frame_num'].max())
                for f, side in nsteps:
                    if fn_min <= f <= fn_max:
                        ax.axvline(f, color=C_RIGHT if side=='right' else C_LEFT,
                                   lw=0.8, alpha=0.7, linestyle=ls)

                # draw suggested steps when enabled
                if self.show_suggestions:
                    ssf = ds.get('suggested_step_frames', [])
                    suggested = _to_fnums(ad_filtered, ssf)
                    
                    # place suggestion markers near the top of the graph
                    if suggested:
                        y_min, y_max = ax.get_ylim()
                        y_position = y_max - (y_max - y_min) * 0.05
                        
                        left_frames = [f for f, side in suggested if side == 'left' and fn_min <= f <= fn_max]
                        right_frames = [f for f, side in suggested if side == 'right' and fn_min <= f <= fn_max]
                        
                        if left_frames:
                            ax.scatter(left_frames, [y_position] * len(left_frames), 
                                     marker='v', s=60, color=C_LEFT, alpha=0.6, zorder=5)
                        if right_frames:
                            ax.scatter(right_frames, [y_position] * len(right_frames), 
                                     marker='v', s=60, color=C_RIGHT, alpha=0.6, zorder=5)

            ref_ds = self._active_ds()
            ref_ad = ref_ds['angle_data'] if ref_ds else self.angle_data
            if ref_ad is not None and not ref_ad.empty and self.current_frame_idx < len(ref_ad):
                cf = ref_ad['frame_num'].iloc[self.current_frame_idx]
                ax.axvline(cf, color=C_CURSOR, lw=1.5, linestyle='--', zorder=10)

            # plot confidence data on secondary y-axis (only if show_confidence is True)
            if self.show_confidence:
                for ds in dfg:
                    conf_data = ds.get('confidence_data')
                    if conf_data is not None and not conf_data.empty:
                        frames = conf_data['frame_num'].values
                        confidence = conf_data['avg_confidence'].values * 100  # scale to 0-100 for visibility
                        si = _src_idx(ds)
                        ax.plot(frames, confidence, color='#27ae60', alpha=0.35, lw=1.0,
                                linestyle=linestyles[si % 2], label=f"Confidence (×100%) V{si+1}", zorder=1)

            # add a time axis on top
            def f2s(x): return x / SLOWMO_FPS
            def s2f(x): return x * SLOWMO_FPS
            ax2 = ax.secondary_xaxis('top', functions=(f2s, s2f))
            ax2.set_xlabel('Time (s)', fontsize=9, color=SUBTEXT)
            ax2.tick_params(colors=SUBTEXT, labelsize=8)

        else:
            ax.set_xlabel('Frames Since Strike', fontsize=8)
            ax.set_ylabel('Angle (°)', fontsize=8)

            for ds in dfg:
                ad   = ds['angle_data']
                # filter out excluded regions
                excluded = ds.get('excluded_regions', [])
                ad_filtered = self._get_filtered_angle_data(ad, excluded)
                
                sf   = ds.get('step_frames', [])
                si   = _src_idx(ds)
                ls   = linestyles[si % 2]
                norm = _to_fnums(ad_filtered, sf)
                left_strikes, right_strikes = _get_cycle_strikes(norm)

                for joint, col in JOINT_COLORS_MPL.items():
                    vis = self.joint_visibility.get(joint, True)
                    if not vis:
                        continue

                    # use appropriate strikes based on limb side
                    limb_side = _get_limb_side(joint)
                    strikes = left_strikes if limb_side == 'left' else right_strikes
                    if len(strikes) < 2:
                        continue

                    cycles, lengths = [], []
                    
                    for i in range(len(strikes)-1):
                        # skip step pairs that cross excluded regions
                        if self._region_crosses_exclusion(strikes[i], strikes[i+1], excluded):
                            continue
                        
                        seg = ad_filtered[(ad_filtered['frame_num'] >= strikes[i]) &
                                 (ad_filtered['frame_num'] <= strikes[i+1])]
                        if seg.empty: continue
                        x = seg['frame_num'].values - strikes[i]
                        y = seg[joint].values
                        cycles.append((x, y))
                        lengths.append(len(x))

                    if not cycles: continue
                    med = np.median(lengths)
                    length_ok  = [0.8*med <= l <= 1.2*med for l in lengths]

                    # always compute mean for rmse checking in overlaid cycles view
                    if self.resample_cycles:
                        length_inliers = []
                        for (x, y), good in zip(cycles, length_ok):
                            if not good: continue
                            t = np.linspace(0, 1, len(y))
                            length_inliers.append(interp1d(t, y)(np.linspace(0, 1, max_cycle_length)))
                        
                        # compute mean from length-filtered cycles
                        if length_inliers:
                            mean_c = np.nanmean(np.vstack(length_inliers), axis=0)
                        else:
                            mean_c = None
                    else:
                        mean_c = None

                    # second pass: filter by both length and rmse
                    ok = []
                    rmse_ok_list = []
                    for idx, ((x, y), length_good) in enumerate(zip(cycles, length_ok)):
                        if not length_good:
                            ok.append(False)
                            rmse_ok_list.append(False)
                            continue
                        
                        # check rmse against mean if available
                        if mean_c is not None and self.resample_cycles:
                            rmse = _compute_cycle_rmse(y, mean_c, max_cycle_length)
                            rmse_ok = rmse <= self.rmse_threshold
                        else:
                            rmse_ok = True
                        
                        rmse_ok_list.append(rmse_ok)
                        ok.append(rmse_ok)

                    if self.show_data:
                        if self.show_outliers_only:
                            # show ONLY outlier cycles in bright color
                            if self.resample_cycles:
                                for (x, y), length_good, rmse_good in zip(cycles, length_ok, rmse_ok_list):
                                    is_outlier = (not length_good) or (not rmse_good)
                                    if not is_outlier:
                                        continue
                                    # show outlier in bright color, high alpha
                                    t = np.linspace(0, 1, len(y))
                                    y_plot = interp1d(t, y)(np.linspace(0, 1, max_cycle_length))
                                    x_plot = np.arange(max_cycle_length)
                                    ax.plot(x_plot, y_plot, color=col, alpha=0.7, lw=1.2, linestyle=ls)
                            else:
                                for (x, y), length_good in zip(cycles, length_ok):
                                    if length_good:
                                        continue
                                    # show length-bad cycle in bright color
                                    ax.plot(x, y, color=col, alpha=0.7, lw=1.2, linestyle=ls)
                        else:
                            # normal mode: show all with filtering
                            if self.resample_cycles:
                                # Resample mode: show length-good cycles (gray if rmse-bad), HIDE length-bad
                                for (x, y), length_good, rmse_good in zip(cycles, length_ok, rmse_ok_list):
                                    if not length_good:
                                        # Hide cycles that fail length check
                                        continue
                                    # This cycle passed length check - show in color or gray based on RMSE
                                    is_rmse_bad = not rmse_good
                                    plot_col = C_OUTLIER if is_rmse_bad else col
                                    plot_alpha = 0.12 if is_rmse_bad else 0.25
                                    
                                    t = np.linspace(0, 1, len(y))
                                    y_plot = interp1d(t, y)(np.linspace(0, 1, max_cycle_length))
                                    x_plot = np.arange(max_cycle_length)
                                    ax.plot(x_plot, y_plot, color=plot_col, alpha=plot_alpha, lw=0.6, linestyle=ls)
                            else:
                                # Non-resample mode: show ALL cycles, grayed out if they're length-bad
                                # First collect y-values from good cycles to set axis range
                                good_y_vals = []
                                for (x, y), good in zip(cycles, length_ok):
                                    if good:
                                        good_y_vals.extend(y)
                                
                                for (x, y), length_good in zip(cycles, length_ok):
                                    is_length_bad = not length_good
                                    plot_col = C_OUTLIER if is_length_bad else col
                                    plot_alpha = 0.12 if is_length_bad else 0.25
                                    ax.plot(x, y, color=plot_col, alpha=plot_alpha, lw=0.8, linestyle=ls)
                                
                                # Set y-limits based on good cycles only
                                if good_y_vals:
                                    y_min, y_max = np.nanmin(good_y_vals), np.nanmax(good_y_vals)
                                    y_margin = (y_max - y_min) * 0.1  # 10% margin
                                    ax.set_ylim(y_min - y_margin, y_max + y_margin)

                    if self.resample_cycles and self.show_mean and not self.show_outliers_only:
                        inliers = []
                        for (x, y), good in zip(cycles, ok):
                            if not good: continue
                            t = np.linspace(0, 1, len(y))
                            inliers.append(interp1d(t, y)(np.linspace(0, 1, max_cycle_length)))
                        if inliers:
                            inliers_array = np.vstack(inliers)
                            mean_c = np.nanmean(inliers_array, axis=0)
                            # calculate standard error band
                            se_c = np.nanstd(inliers_array, axis=0) / np.sqrt(len(inliers))
                            lower_c = mean_c - se_c
                            upper_c = mean_c + se_c
                            x_plot = np.arange(len(mean_c))
                            # plot error band first (behind) - only in se_shading mode
                            if self.graph_display_mode == 'se_shading':
                                ax.fill_between(x_plot, lower_c, upper_c, color=col, alpha=0.15, zorder=2)
                            # plot mean line on top
                            ax.plot(x_plot, mean_c,
                                    color=col, lw=2.2, linestyle=ls, zorder=3,
                                    label=f"{joint.replace('_',' ').title()} V{si+1} mean")

            if self.resample_cycles and self.show_normative and not self.show_outliers_only:
                # scale normative data to the current x range
                norm_x_resampled = np.linspace(0, max_cycle_length, 100)
                
                for jt_key in ('hip', 'knee', 'ankle'):
                    vl = self.joint_visibility.get(f'left_{jt_key}', False)
                    vr = self.joint_visibility.get(f'right_{jt_key}', False)
                    if vl or vr:
                        d = NORMATIVE_GAIT[jt_key]
                        # plot normative mean line only
                        ax.plot(norm_x_resampled, d['mean'], color=C_NORM, lw=1.5, linestyle='-', alpha=0.8, zorder=2,
                                label=f'{jt_key.title()} norm.')
        # set x limits after plotting to avoid autoscaling drift
        if self.show_overlaid_cycles:
            # overlaid cycles use the longest actual stride
            ax.set_xlim(0, self._current_max_cycle_length)
        elif self._ax_xlim_full is not None:
            # continuous view uses the frame range
            ax.set_xlim(self._ax_xlim_full)

        # keep graph size stable across redraws
        self._fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.12)
        self._update_scrollbar()
        self._mpl_canvas.draw_idle()

    # video display
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
                # current format stores a raw path plus pixel landmarks
                if isinstance(entry, tuple):
                    frame = self._cache.get(vi, self.current_frame_idx, store)
                    pixel_lm = entry[1]
                else:
                    # older data may store a frame path or array directly
                    frame = self._cache.get(vi, self.current_frame_idx, store)

            canvas.delete('all')
            if frame is None:
                canvas.create_text(cw//2, ch//2, text="No frame",
                                   fill=SUBTEXT, font=("Helvetica", 10))
                continue

            # draw the skeleton at render time using current joint visibility
            jittery_frames = self.datasets[vi].get('jittery_frames', set()) if self.remove_jitter_frames else set()
            is_jittery = self.current_frame_idx in jittery_frames
            
            # if show_jitter_frames toggle is on, draw jittery frames in red; otherwise skip them
            if pixel_lm is not None:
                if is_jittery and not self.show_jitter_frames:
                    # skip drawing skeleton for jittery frame
                    pass
                else:
                    frame = frame.copy()
                    draw_pose_landmarks_on_frame(
                        frame,
                        pixel_lm,
                        self.joint_visibility,
                        skeleton_thickness=self.skeleton_thickness,
                        draw_jitter_red=is_jittery and self.show_jitter_frames,
                    )

            # rotate the analysis frame so the person appears upright in the gui
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # dim videos that are not active in the current graph view
            show_v1 = self.graph_show_mode in ('v1', 'both')
            show_v2 = self.graph_show_mode in ('v2', 'both')
            is_inactive = (vi == 0 and not show_v1) or (vi == 1 and not show_v2)
            
            if is_inactive:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rgb   = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            fh, fw = rgb.shape[:2]
            scale  = min(cw/fw, ch/fh)
            nw, nh = int(fw*scale), int(fh*scale)
            rgb    = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            img    = ImageTk.PhotoImage(Image.fromarray(rgb))
            canvas._img = img   # keep a reference for tkinter
            canvas.create_image((cw-nw)//2, (ch-nh)//2, anchor='nw', image=img)

    # refresh
    def refresh(self):
        self.redraw_graph()
        self._show_video_frames()
        self._update_metrics_panel()
        self._update_status()

    def _update_status(self):
        ad = self._active_angle_data()
        n  = len(ad) if (ad is not None and not ad.empty) else 0
        t  = self.current_frame_idx / SLOWMO_FPS
        self._frame_lbl.config(
            text=f"Frame {self.current_frame_idx+1}/{n or '?'}  ({t:.2f} s)")
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
            if hib is None:
                fg = TEXT
            elif abs(val) < 1.0:
                fg = SUBTEXT
            elif hib:
                fg = GREEN if val > 0 else RED
            else:
                fg = GREEN if val < 0 else RED
            lbl.config(text=txt, fg=fg)

    # graph mouse
    def _seek_from_event(self, event):
        ad = self._active_angle_data()
        if ad is None or ad.empty or event.xdata is None: return
        fn  = ad['frame_num'].to_numpy()
        idx = int(np.argmin(np.abs(fn - event.xdata)))
        self.current_frame_idx = max(0, min(idx, len(ad)-1))
        # preserve zoom while scrubbing
        current_xlim = self._ax.get_xlim()
        self._show_video_frames()
        self._update_status()
        self.redraw_graph()
        # restore zoom after redraw
        self._ax.set_xlim(current_xlim)
        self._update_scrollbar()

    def _on_graph_click(self, event):
        if event.button == 1 and event.inaxes == self._ax:
            self._graph_dragging = True
            self._seek_from_event(event)
            # keep the current zoom and only move the scrubber
        elif event.button == 3 and event.inaxes == self._ax:
            # start selecting an exclusion with right click
            self._exclusion_selecting = True
            self._exclusion_start = event.xdata

    def _on_graph_drag(self, event):
        if self._graph_dragging and event.inaxes == self._ax:
            self._seek_from_event(event)

    def _on_graph_release(self, event):
        if event.button == 1:
            self._graph_dragging = False
        elif event.button == 3:
            # finish the exclusion on right button release
            if self._exclusion_selecting and self._exclusion_start is not None and event.xdata is not None:
                start_frame = round(self._exclusion_start)
                end_frame = round(event.xdata)
                # keep the range ordered
                if start_frame > end_frame:
                    start_frame, end_frame = end_frame, start_frame
                # ignore one frame selections
                if end_frame > start_frame:
                    target_datasets = []
                    
                    # choose the dataset or datasets that should receive the exclusion
                    if self.graph_show_mode == 'v1' and len(self.datasets) >= 1:
                        target_datasets = [self.datasets[0]]
                    elif self.graph_show_mode == 'v2' and len(self.datasets) >= 2:
                        target_datasets = [self.datasets[1]]
                    elif self.graph_show_mode == 'both' and len(self.datasets) >= 2:
                        # apply to both videos in combined view
                        target_datasets = [self.datasets[0], self.datasets[1]]
                    else:
                        target_datasets = [self._active_ds()]
                    
                    # apply the exclusion to each selected dataset
                    for target_ds in target_datasets:
                        if target_ds:
                            target_ds.setdefault('excluded_regions', []).append((start_frame, end_frame))
                            target_ds['excluded_regions'] = self._merge_exclusion_regions(
                                target_ds['excluded_regions'])
                            self._persist_dataset_markup(target_ds)
                    
                    self._status_msg.set(f"Excluded frames {start_frame}-{end_frame} in both videos")
                    current_xlim = self._ax.get_xlim()
                    self.redraw_graph()
                    self._ax.set_xlim(current_xlim)
                    self._update_scrollbar()
                    #self._recompute_steps()  # recompute with filtered data
            self._exclusion_selecting = False
            self._exclusion_start = None

    def _on_canvas_scroll(self, event):
        if self.angle_data is None or self.angle_data.empty:
            return
        
        # disable scrolling in overlaid cycle view
        if self.show_overlaid_cycles:
            return
        
        # keep the last wheel event for cursor based zoom
        self._last_scroll_event = event
        
        # normalize wheel direction across platforms
        if hasattr(event, 'delta'):
            scroll_dir = 1 if event.delta > 0 else -1
        elif hasattr(event, 'num'):
            scroll_dir = 1 if event.num == 4 else -1
        else:
            return
        
        # ctrl switches the wheel from pan to zoom
        ctrl_held = bool(event.state & 0x0004)  # ctrl mask
        
        if ctrl_held:
            self._on_graph_zoom(scroll_dir)
        else:
            self._on_graph_pan(scroll_dir)
    
    def _on_graph_pan(self, direction):
        cur_xlim = self._ax.get_xlim()
        xlim_full = self._get_current_xlim_full()
        full_min, full_max = xlim_full
        pan_amount = (full_max - full_min) * 0.1 * direction  # move by ten percent of the range
        
        new_xlim_min = cur_xlim[0] - pan_amount
        new_xlim_max = cur_xlim[1] - pan_amount
        
        # clamp the pan range to the full data bounds
        if new_xlim_min < full_min:
            new_xlim_min = full_min
            new_xlim_max = full_min + (cur_xlim[1] - cur_xlim[0])
        if new_xlim_max > full_max:
            new_xlim_max = full_max
            new_xlim_min = full_max - (cur_xlim[1] - cur_xlim[0])
        
        self._ax.set_xlim(new_xlim_min, new_xlim_max)
        self._update_scrollbar()
        self._mpl_canvas.draw_idle()
    
    def _on_graph_zoom(self, direction):
        # read the current axis limits
        cur_xlim = self._ax.get_xlim()
        xlim_full = self._get_current_xlim_full()
        full_min, full_max = xlim_full
        full_range = full_max - full_min
        
        # negative direction zooms in and positive direction zooms out
        zoom_factor = 0.8 if direction < 0 else 1.2
        
        # use the cursor position as the zoom center when possible
        zoom_center = None
        if self._last_scroll_event:
            if hasattr(self._last_scroll_event, 'xdata') and self._last_scroll_event.xdata is not None:
                zoom_center = self._last_scroll_event.xdata
            elif hasattr(self._last_scroll_event, 'x'):
                # convert display pixels into data coordinates
                try:
                    # get the axes bounds in display space
                    bbox = self._ax.get_window_extent()
                    # get the event x position in display space
                    x_pixel = self._last_scroll_event.x
                    # convert into normalized axes space
                    x_normalized = (x_pixel - bbox.x0) / bbox.width
                    # convert the normalized position into data space
                    if 0 <= x_normalized <= 1:
                        zoom_center = cur_xlim[0] + x_normalized * (cur_xlim[1] - cur_xlim[0])
                except:
                    pass
        
        # fall back to the current midpoint when needed
        if zoom_center is None:
            zoom_center = (cur_xlim[0] + cur_xlim[1]) / 2
        
        # calculate the new visible width
        new_width = (cur_xlim[1] - cur_xlim[0]) * zoom_factor
        
        # stop zooming out past the full range
        if new_width >= full_range:
            new_xlim_min = full_min
            new_xlim_max = full_max
        else:
            # preserve the cursor position inside the current window
            rel_pos = (zoom_center - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
            new_xlim_min = zoom_center - new_width * rel_pos
            new_xlim_max = new_xlim_min + new_width
            
            # shift back into bounds when needed
            if new_xlim_min < full_min:
                new_xlim_min = full_min
                new_xlim_max = full_min + new_width
            if new_xlim_max > full_max:
                new_xlim_max = full_max
                new_xlim_min = full_max - new_width
        
        # apply the updated limits
        self._ax.set_xlim(new_xlim_min, new_xlim_max)
        self._update_scrollbar()
        self._mpl_canvas.draw_idle()
    
    def _get_current_xlim_full(self):
        if self.show_overlaid_cycles:
            return (0, getattr(self, '_current_max_cycle_length', self.resample_length))
        return self._ax_xlim_full or (0, 100)
    
    def _update_scrollbar(self):
        xlim_full = self._get_current_xlim_full()
        if xlim_full is None:
            return
        
        cur_xlim = self._ax.get_xlim()
        full_min, full_max = xlim_full
        full_range = full_max - full_min
        
        # calculate the scrollbar thumb position
        first = (cur_xlim[0] - full_min) / full_range
        last = (cur_xlim[1] - full_min) / full_range
        
        # update the scrollbar thumb
        self._graph_hbar.set(first, last)
    
    def _on_scrollbar_drag(self, *args):
        # disable the scrollbar in overlaid cycle view
        if self.show_overlaid_cycles:
            return
        
        xlim_full = self._get_current_xlim_full()
        if xlim_full is None or not args:
            return
        
        full_min, full_max = xlim_full
        full_range = full_max - full_min
        cur_width = self._ax.get_xlim()[1] - self._ax.get_xlim()[0]
        
        # parse the tkinter scrollbar callback arguments
        if args[0] == 'moveto':
            # direct drag command from the scrollbar
            fraction = float(args[1])
            new_xlim_min = full_min + fraction * full_range
            new_xlim_max = new_xlim_min + cur_width
            # clamp the view to the valid range
            if new_xlim_max > full_max:
                new_xlim_max = full_max
                new_xlim_min = full_max - cur_width
        elif args[0] == 'scroll':
            # arrow or page step from the scrollbar
            amount = int(args[1])
            units = args[2] if len(args) > 2 else 'units'
            scroll_amount = full_range * 0.1 * amount
            cur_xlim = self._ax.get_xlim()
            new_xlim_min = cur_xlim[0] - scroll_amount
            new_xlim_max = cur_xlim[1] - scroll_amount
            # clamp the view to the valid range
            if new_xlim_min < full_min:
                new_xlim_min = full_min
                new_xlim_max = full_min + cur_width
            if new_xlim_max > full_max:
                new_xlim_max = full_max
                new_xlim_min = full_max - cur_width
        else:
            return
        
        self._ax.set_xlim(new_xlim_min, new_xlim_max)
        self._mpl_canvas.draw_idle()

    # playback
    def _play_tick(self):
        if not self.playing: return
        if self.current_frame_idx < self._active_max_index():
            self.current_frame_idx += 1
            # preserve zoom during playback
            current_xlim = self._ax.get_xlim()
            self._show_video_frames()
            self.redraw_graph()
            self._ax.set_xlim(current_xlim)
            self._update_scrollbar()
            self._update_status()
            self._play_after_id = self.after(16, self._play_tick)
        else:
            self.playing = False
            self._status_msg.set("Playback finished")


    # controls
    def _prev_frame(self):
        if self.playing: return
        self.current_frame_idx = max(0, self.current_frame_idx - 1)
        if self._marking_phase:
            t = self.current_frame_idx / SLOWMO_FPS
            self._markup_frame_lbl.config(
                text=f"Frame {self.current_frame_idx + 1}  ({t:.2f} s)")
            self._markup_show_frames()
            self._redraw_markup_graph()
            return

        # preserve zoom
        current_xlim = self._ax.get_xlim()
        self._show_video_frames()
        self.redraw_graph()
        self._ax.set_xlim(current_xlim)
        self._update_scrollbar()
        self._update_status()

    def _next_frame(self):
        if self.playing: return
        if self._marking_phase:
            vi = self._marking_video_idx
            max_idx = (len(self.datasets[vi].get('all_landmarks', [])) - 1
                       if vi < len(self.datasets) else self._active_max_index())
            self.current_frame_idx = min(max_idx, self.current_frame_idx + 1)
            t = self.current_frame_idx / SLOWMO_FPS
            self._markup_frame_lbl.config(
                text=f"Frame {self.current_frame_idx + 1}  ({t:.2f} s)")
            self._markup_show_frames()
            self._redraw_markup_graph()
            return
        self.current_frame_idx = min(self._active_max_index(), self.current_frame_idx + 1)
        # preserve zoom
        current_xlim = self._ax.get_xlim()
        self._show_video_frames()
        self.redraw_graph()
        self._ax.set_xlim(current_xlim)
        self._update_scrollbar()
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

    def _reset_zoom(self):
        xlim_full = self._get_current_xlim_full()
        if xlim_full is not None:
            self._ax.set_xlim(xlim_full)
            self._update_scrollbar()
            self._mpl_canvas.draw_idle()
            self._status_msg.set("Zoom reset")

    def _toggle_cycles(self):
        self.show_overlaid_cycles = not self.show_overlaid_cycles
        # enable resampling automatically in overlaid cycle view
        if self.show_overlaid_cycles:
            self.resample_cycles = True
        self._status_msg.set("Overlaid cycles" if self.show_overlaid_cycles else "Continuous view")
        self._update_display_btn_visuals()
        self.redraw_graph()

    def _toggle_resample(self):
        if not self.show_overlaid_cycles: return
        self.resample_cycles = not self.resample_cycles
        self._status_msg.set(f"Resample {'on' if self.resample_cycles else 'off'}")
        self.redraw_graph()

    def _toggle_mean(self):
        if not self.show_overlaid_cycles: return
        self.show_mean = not self.show_mean
        if not self.show_mean:
            self.mean_only = False
        self._update_display_btn_visuals()
        self.redraw_graph()

    def _toggle_confidence(self):
        self.show_confidence = not self.show_confidence
        self._status_msg.set(f"Confidence scores {'shown' if self.show_confidence else 'hidden'}")
        self.redraw_graph()

    def _clear_current_cache(self):
        """Clear cached coordinate/step data for currently loaded videos."""
        if not self.datasets:
            messagebox.showwarning("No Videos", "No videos loaded")
            return
        
        cleared_count = 0
        for i, ds in enumerate(self.datasets):
            cache_key = ds.get('_cache_key')
            if cache_key:
                cache_path = _cache_dir(cache_key)
                try:
                    if os.path.exists(cache_path):
                        shutil.rmtree(cache_path)
                        cleared_count += 1
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to clear cache for video {i+1}: {e}")
                    return
        
        if cleared_count > 0:
            messagebox.showinfo("Cache Cleared", f"Cleared cache for {cleared_count} video(s). Reload to reprocess.")
            self._status_msg.set(f"Cleared cache for {cleared_count} video(s)")
        else:
            messagebox.showwarning("No Cache", "No cached data found to clear")

    def _open_settings(self):
        """Open the settings dialog (singleton pattern)."""
        if self._settings_dialog is not None and self._settings_dialog.winfo_exists():
            self._settings_dialog.lift()
            self._settings_dialog.focus()
        else:
            self._settings_dialog = SettingsDialog(self, self)
            # when the dialog is destroyed, clear the reference
            dialog = self._settings_dialog
            def on_close():
                self._settings_dialog = None
                dialog.destroy()
            self._settings_dialog.protocol("WM_DELETE_WINDOW", on_close)

    def _toggle_world(self):
        global USE_WORLD_LANDMARKS
        USE_WORLD_LANDMARKS = not USE_WORLD_LANDMARKS
        key = 'df_world' if USE_WORLD_LANDMARKS else 'df_pixel_filtered'
        for ds in self.datasets:
            if key in ds: ds['angle_data'] = ds[key]
        self.angle_data = self.df_world if USE_WORLD_LANDMARKS else self.df_pixel_filtered
        self._status_msg.set("World landmarks" if USE_WORLD_LANDMARKS else "Pixel landmarks")
        self._update_display_btn_visuals()
        self.redraw_graph()

    def _toggle_video_view(self, which):
        """Toggle V1 (which=0) or V2 (which=1) independently."""
        show_v1 = self.graph_show_mode in ('v1', 'both')
        show_v2 = self.graph_show_mode in ('v2', 'both')
        if which == 0:
            show_v1 = not show_v1
        else:
            show_v2 = not show_v2
        # prevent hiding both videos at once
        if not show_v1 and not show_v2:
            show_v1 = True
            show_v2 = True
        if show_v1 and show_v2:
            self.graph_show_mode = 'both'
        elif show_v1:
            self.graph_show_mode = 'v1'
        else:
            self.graph_show_mode = 'v2'
        # load the saved zoom range for this view
        if self.graph_show_mode in self._ax_xlim_per_mode:
            self._ax_xlim_full = self._ax_xlim_per_mode[self.graph_show_mode]
        # reset zoom when the view changes
        if self._ax_xlim_full is not None:
            self._ax.set_xlim(self._ax_xlim_full)
        labels = {'both': 'Both', 'v1': 'V1 only', 'v2': 'V2 only'}
        self._status_msg.set(f"Graph: {labels[self.graph_show_mode]}")
        self._update_video_btn_visuals()
        self._show_video_frames()
        self.redraw_graph()

    def _update_video_btn_visuals(self):
        show_v1 = self.graph_show_mode in ('v1', 'both')
        show_v2 = self.graph_show_mode in ('v2', 'both')
        if show_v1:
            self._v1_btn_frame.config(bg=BG3)
            self._v1_toggle_lbl.config(bg=BG3, fg=C_V1)
            self._v1_swatch.config(bg=BG3)
            self._v1_swatch.delete('all')
            self._v1_swatch.create_line(2, 7, 22, 7, fill=C_V1, dash=(4, 3), width=2)
        else:
            self._v1_btn_frame.config(bg=BG2)
            self._v1_toggle_lbl.config(bg=BG2, fg='#555555')
            self._v1_swatch.config(bg=BG2)
            self._v1_swatch.delete('all')
            self._v1_swatch.create_line(2, 7, 22, 7, fill='#555555', dash=(4, 3), width=2)
        if show_v2:
            self._v2_btn_frame.config(bg=BG3)
            self._v2_toggle_lbl.config(bg=BG3, fg=C_V2)
            self._v2_swatch.config(bg=BG3)
            self._v2_swatch.delete('all')
            self._v2_swatch.create_line(2, 7, 22, 7, fill=C_V2, width=2)
        else:
            self._v2_btn_frame.config(bg=BG2)
            self._v2_toggle_lbl.config(bg=BG2, fg='#555555')
            self._v2_swatch.config(bg=BG2)
            self._v2_swatch.delete('all')
            self._v2_swatch.create_line(2, 7, 22, 7, fill='#555555', width=2)

    def _cycle_graph_view(self):
        modes  = ['both', 'v1', 'v2']
        labels = {'both': 'Both', 'v1': 'V1 only', 'v2': 'V2 only'}
        self.graph_show_mode = modes[(modes.index(self.graph_show_mode)+1) % 3]
        # load the saved zoom range for this view
        if self.graph_show_mode in self._ax_xlim_per_mode:
            self._ax_xlim_full = self._ax_xlim_per_mode[self.graph_show_mode]
        # reset zoom when the view changes
        if self._ax_xlim_full is not None:
            self._ax.set_xlim(self._ax_xlim_full)
        self._status_msg.set(f"Graph: {labels[self.graph_show_mode]}")
        self._update_video_btn_visuals()
        self._show_video_frames()
        self.redraw_graph()

    def _toggle_active(self):
        if len(self.datasets) >= 2:
            self.active_dataset_idx = 1 - self.active_dataset_idx
            self._status_msg.set(f"Active: V{self.active_dataset_idx+1}")
            self._show_video_frames()
            self.redraw_graph()

    def _toggle_joint(self, joint):
        self.joint_visibility[joint] = not self.joint_visibility[joint]
        state = 'shown' if self.joint_visibility[joint] else 'hidden'
        self._status_msg.set(f"{joint.replace('_',' ')} {state}")
        self._update_legend_visuals()
        self._show_video_frames()
        self.redraw_graph()

    def _toggle_joint_legend(self, joint):
        self._toggle_joint(joint)

    def _update_legend_visuals(self):
        for joint, item in self._legend_items.items():
            vis = self.joint_visibility.get(joint, True)
            col = item['color']
            if vis:
                item['frame'].config(bg=BG3)
                item['label'].config(bg=BG3, fg=TEXT)
                item['swatch'].config(bg=BG3)
                item['swatch'].delete('all')
                item['swatch'].create_line(2, 9, 16, 9, fill=col, dash=(4, 3), width=2)
                item['swatch'].create_line(20, 9, 34, 9, fill=col, width=2)
            else:
                gray = '#999999'
                item['frame'].config(bg=BG2)
                item['label'].config(bg=BG2, fg=gray)
                item['swatch'].config(bg=BG2)
                item['swatch'].delete('all')
                item['swatch'].create_line(2, 9, 16, 9, fill=gray, dash=(4, 3), width=2)
                item['swatch'].create_line(20, 9, 34, 9, fill=gray, width=2)

    def _toggle_all_joints(self):
        all_on = all(self.joint_visibility.values())
        for k in self.joint_visibility:
            self.joint_visibility[k] = not all_on
        self._update_legend_visuals()
        self._show_video_frames()
        self.redraw_graph()

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

    def _toggle_display_option(self, key):
        if key == 'mean':
            if not self.show_overlaid_cycles:
                return
            self.show_mean = not self.show_mean
            if not self.show_mean:
                self.mean_only = False
        elif key == 'data':
            if not self.show_overlaid_cycles:
                return
            self.show_data = not self.show_data
        elif key == 'normal':
            if not self.show_overlaid_cycles:
                return
            self.show_normative = not self.show_normative
        elif key == 'outliers':
            if not self.show_overlaid_cycles:
                return
            self.show_outliers_only = not self.show_outliers_only
        self._update_display_btn_visuals()
        self.redraw_graph()

    def _update_display_btn_visuals(self):
        active_map = {
            'mean': self.show_mean,
            'data': self.show_data,
            'normal': self.show_normative,
            'outliers': self.show_outliers_only,
        }
        for key, btn in self._display_btns.items():
            if not self.show_overlaid_cycles:
                btn.config(bg=BG2, fg='#999999', state='disabled')
            elif active_map.get(key, False):
                btn.config(bg=ACCENT, fg='white', state='normal')
            else:
                btn.config(bg=BG3, fg=TEXT, state='normal')

        if self.show_overlaid_cycles:
            self._sidebar_toggle_btns['cycles'].config(text="Continuous", bg=ACCENT, fg='white')
        else:
            self._sidebar_toggle_btns['cycles'].config(text="Cycles", bg=ACCENT, fg='white')

        if USE_WORLD_LANDMARKS:
            self._sidebar_toggle_btns['world_px'].config(text="Pixel", bg=ACCENT, fg='white')
        else:
            self._sidebar_toggle_btns['world_px'].config(text="World", bg=BG3, fg=TEXT)

    def _toggle_manual_step(self):
        self.manual_step_mode = not self.manual_step_mode
        self._status_msg.set(
            f"Manual step mode {'ON' if self.manual_step_mode else 'OFF'}  [{self.manual_side}]")

    def _set_manual_side(self, side):
        self.manual_side = side
        self._status_msg.set(f"Manual side: {side.upper()}")

    def _add_manual_step(self):
        # route step input to guided markup when active
        if self._marking_phase:
            self._markup_add_step()
            return
        # choose the dataset from the current graph view
        if self.graph_show_mode == 'v1' and len(self.datasets) >= 1:
            ds = self.datasets[0]
        elif self.graph_show_mode == 'v2' and len(self.datasets) >= 2:
            ds = self.datasets[1]
        else:
            ds = self._active_ds()
        
        if not ds or ds['angle_data'].empty: return
        idx = min(self.current_frame_idx, len(ds['angle_data'])-1)
        fn  = int(ds['angle_data']['frame_num'].iloc[idx])
        
        # infer the foot from the nearest suggestion
        suggested = ds.get('suggested_step_frames', [])
        detected_foot = 'right'  # fallback side
        
        if suggested:
        # find the nearest suggested step
            nearest_idx = min(range(len(suggested)), key=lambda i: abs(suggested[i][0] - fn))
            detected_foot = suggested[nearest_idx][1]
        
        ds.setdefault('step_frames', []).append((fn, detected_foot))
        ds['step_frames'].sort(key=lambda x: x[0])
        self._persist_dataset_markup(ds)
        self._status_msg.set(f"Added {detected_foot} step @ frame {fn}")
        self._update_metrics_panel()
        current_xlim = self._ax.get_xlim()
        self.redraw_graph()
        self._ax.set_xlim(current_xlim)
        self._update_scrollbar()

    def _delete_nearest_step(self):
        if self._marking_phase:
            self._markup_remove_last()
            return

        # choose the dataset from the current graph view
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
        current_xlim = self._ax.get_xlim()
        self.redraw_graph()
        self._ax.set_xlim(current_xlim)
        self._update_scrollbar()

    def _recompute_steps(self):
        if not self.datasets: 
            return
        total_steps = 0
        total_excl = 0
        for ds in self.datasets:
            if ds and 'angle_data' in ds:
                ad = ds.get('angle_data', pd.DataFrame())
                depths = ds.get('landmark_depths', pd.DataFrame())
                suggested, auto_excl, step_meta = detect_steps_robust(
                    ad, depth_df=depths, fps=SLOWMO_FPS)
                ds['suggested_step_frames'] = suggested
                ds['suggested_step_meta'] = step_meta
                ds['excluded_regions'] = self._merge_exclusion_regions(list(auto_excl))
                self._persist_dataset_markup(ds)
                total_steps += len(ds.get('suggested_step_frames', []))
                total_excl += len(ds.get('excluded_regions', []))
        self._status_msg.set(
            f"Auto clean complete: {total_steps} step suggestions, {total_excl} exclusion zones")
        self._update_metrics_panel()
        self.redraw_graph()

    def _toggle_suggestions(self):
        self.show_suggestions = not self.show_suggestions
        status = "shown" if self.show_suggestions else "hidden"
        self._status_msg.set(f"Suggestions {status}")
        self.redraw_graph()

    def _clear_steps(self):
        if not self.datasets: 
            return
        for ds in self.datasets:
            if ds:
                ds['step_frames'] = []
                self._persist_dataset_markup(ds)
        self._status_msg.set("Steps cleared from all videos")
        self._update_metrics_panel()
        self.redraw_graph()

    def _clear_exclusions(self):
        if not self.datasets:
            return
        for ds in self.datasets:
            if ds:
                ds['excluded_regions'] = []
                self._persist_dataset_markup(ds)
        self._status_msg.set("All exclusions cleared")
        #self._recompute_steps()  # recalculate steps with full data
        self.redraw_graph()

    def _restart_marking_wizard(self):
        if len(self.datasets) < 2:
            messagebox.showwarning("No videos loaded",
                                   "Load two videos before reopening the step marking wizard.",
                                   parent=self)
            return

        # Keep existing steps and persist them
        for ds in self.datasets:
            if ds:
                self._persist_dataset_markup(ds)

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

        self._markup_step_lbl = tk.Label(
            top_row, text="STEP  1  OF  4",
            font=("Helvetica", 9, "bold"), bg=BG2, fg=SUBTEXT, anchor='w')
        self._markup_step_lbl.pack(side='left')

        # small badge showing the active side
        self._markup_side_badge = tk.Label(
            top_row, text=" LEFT ",
            font=("Helvetica", 9, "bold"), bg=C_LEFT, fg='white', padx=6, pady=1)
        self._markup_side_badge.pack(side='left', padx=(10, 0))

        self._markup_count_lbl = tk.Label(
            top_row, text="0 steps marked",
            font=("Helvetica", 10, "bold"), bg=BG2, fg=C_LEFT, anchor='e')
        self._markup_count_lbl.pack(side='right')

        # compact title row
        title_row = tk.Frame(self._markup_banner_area, bg=BG2)
        title_row.pack(fill='x', padx=20, pady=(2, 4))

        self._markup_banner_lbl = tk.Label(
            title_row,
            text="MARKING LEFT STEPS — VIDEO 1",
            font=("Coiny Cyrillic", 13,), bg=BG2, fg=ACCENT, anchor='w')
        self._markup_banner_lbl.pack(side='left')

        self._markup_sub_lbl = tk.Label(
            title_row,
            text="· press  SPACE  to mark each LEFT foot strike",
            font=("Helvetica", 9), bg=BG2, fg=SUBTEXT, anchor='w')
        self._markup_sub_lbl.pack(side='left', padx=(12, 0))

        # thin separator
        tk.Frame(self._markup_banner_area, bg=BG3, height=1).pack(fill='x')

        vid_area = tk.Frame(self._markup_frame, bg=BG)
        vid_area.pack(fill='both', expand=True, padx=8, pady=(8, 0))

        vf = tk.Frame(vid_area, bg=BG2, bd=1, relief='flat')
        vf.pack(fill='both', expand=True)
        self._markup_vid_lbl = tk.Label(
            vf, text="VIDEO 1",
            font=("Helvetica", 9, "bold"), bg=BG2, fg=C_V1, anchor='w')
        self._markup_vid_lbl.pack(fill='x', padx=6, pady=(3, 0))
        self._markup_canvas = tk.Canvas(vf, bg=BG_VID, highlightthickness=0)
        self._markup_canvas.pack(fill='both', expand=True)
        # keep the list form for compatibility with existing code
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
        self._markup_mpl_canvas.mpl_connect('button_press_event',  self._on_markup_graph_click)
        self._markup_mpl_canvas.mpl_connect('motion_notify_event', self._on_markup_graph_drag)
        self._markup_mpl_canvas.mpl_connect('button_release_event', self._on_markup_graph_release)
        self._markup_graph_dragging = False

        ctrl_row = tk.Frame(self._markup_frame, bg=BG2, height=52)
        ctrl_row.pack(fill='x', side='bottom')
        ctrl_row.pack_propagate(False)

        tk.Button(
            ctrl_row, text="⌫  Undo Last Step",
            font=("Helvetica", 9), bg=BG3, fg=TEXT,
            relief='flat', padx=12, cursor='hand2',
            command=self._markup_remove_last
        ).pack(side='left', padx=16, pady=10)

        self._markup_frame_lbl = tk.Label(
            ctrl_row, text="Frame 3  (0.01 s)",
            font=("Helvetica", 9, "bold"), bg=BG2, fg=TEXT)
        self._markup_frame_lbl.pack(side='left', padx=(0, 16))

        self._markup_continue_btn = tk.Button(
            ctrl_row,
            text="Done  →  Next",
            font=("Helvetica", 11, "bold"),
            bg=ACCENT, fg='white', relief='flat', padx=20, cursor='hand2',
            activebackground='#5a186a', activeforeground='white')
        self._markup_continue_btn.pack(side='right', padx=20, pady=10)

        tk.Label(
            ctrl_row,
            text="Click graph to seek  ·  SPACE = mark step  ·  1/2 = frame by frame",
            font=("Helvetica", 8), bg=BG2, fg=SUBTEXT
        ).pack(side='left', padx=8)

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
        vid_name = (self.video_names[video_idx]
                    if video_idx < len(self.video_names) else f"Video {vid_num}")
        fg_col = C_V1 if video_idx == 0 else C_V2
        marker_col = C_LEFT if side == 'left' else C_RIGHT

        # choose the next step in the markup flow
        next_phase = self._next_markup_phase(side, video_idx)
        if next_phase is not None:
            next_side, next_video_idx = next_phase
            next_side_label = "LEFT" if next_side == 'left' else "RIGHT"
            continue_txt = f"Done  →  Mark {next_side_label} steps on Video {next_video_idx + 1}"
            continue_cmd = lambda s=next_side, vi=next_video_idx: self._enter_marking_phase(s, vi)
        else:
            continue_txt = "Finish  →  View Gait Analysis"
            continue_cmd = self._exit_marking_phase

        # update the banner widgets
        self._markup_step_lbl.config(text=f"STEP  {phase_num}  OF  {phase_total}")
        self._markup_side_badge.config(text=f"  {noun}  ", bg=marker_col)
        self._markup_banner_lbl.config(
            text=f"MARKING {noun} STEPS — VIDEO {vid_num}")
        self._markup_sub_lbl.config(
            text=f"· press  SPACE  to mark each {noun} foot strike  ({vid_name})")
        self._markup_count_lbl.config(fg=marker_col)
        self._markup_vid_lbl.config(
            text=f"VIDEO {vid_num}  —  {vid_name}", fg=fg_col)
        self._markup_continue_btn.config(text=continue_txt, command=continue_cmd)

        self._markup_count_update()
        self._markup_show_frames()

        # swap the main dashboard out for the markup screen
        self._main_content.pack_forget()
        self._bottom_bar.pack_forget()
        self._markup_frame.pack(fill='both', expand=True)
        self._status_msg.set(
            f"Mark every {noun} foot strike with SPACE — Video {vid_num}")
        # wait briefly so tkinter can resolve widget geometry
        self.after(50, self._markup_show_frames)
        self.after(60, self._redraw_markup_graph)

    def _exit_marking_phase(self):
        self._marking_phase = None
        self._markup_frame.pack_forget()
        self._bottom_bar.pack(fill='x', side='bottom')
        self._main_content.pack(fill='both', expand=True, padx=8, pady=(4, 0))
        self._persist_all_dataset_markup()

        # switch into overlaid cycles when markup is complete
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
                canvas.create_text(cw // 2, ch // 2, text="No frame",
                                   fill=SUBTEXT, font=("Helvetica", 10))
            return

        if pixel_lm is not None:
            jittery_frames = self.datasets[vi].get('jittery_frames', set()) if self.remove_jitter_frames else set()
            is_jittery = self.current_frame_idx in jittery_frames
            
            # if show_jitter_frames toggle is on, draw jittery frames in red; otherwise skip them
            if is_jittery and not self.show_jitter_frames:
                # skip drawing skeleton for jittery frame
                pass
            else:
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
        ax.xaxis.label.set_color(SUBTEXT)

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
        for start_frame, end_frame in excluded:
            excl_mask |= (frames >= start_frame) & (frames < end_frame)

        # shade excluded windows to match the main graph behavior
        for start_frame, end_frame in excluded:
            ax.axvspan(start_frame, end_frame, alpha=0.18, color='#6e6e6e', zorder=1)

        for joint, col in JOINT_COLORS_MPL.items():
            if joint not in ad.columns:
                continue
            if not self.joint_visibility.get(joint, True):
                continue
            vals = ad[joint].values.astype(float)
            if excluded:
                gray_vals = vals.copy()
                gray_vals[~excl_mask] = np.nan
                ax.plot(frames, gray_vals, color='#7a7a7a', lw=1.2, alpha=0.9, zorder=2)
            clean_vals = vals.copy()
            clean_vals[excl_mask] = np.nan
            ax.plot(frames, clean_vals, color=col, lw=1.0, alpha=0.85, zorder=3)

        # draw the confirmed step markers for the current side
        side = self._marking_phase
        if side:
            fn_set = set(int(f) for f in frames)
            for f, s in ds.get('step_frames', []):
                if s == side and int(f) in fn_set:
                    mc = C_LEFT if s == 'left' else C_RIGHT
                    ax.axvline(int(f), color=mc, lw=1.5, alpha=0.9, zorder=5)

        # draw the current frame cursor
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
        if event.xdata is None:
            return
        vi = self._marking_video_idx
        if not self.datasets or vi >= len(self.datasets):
            return
        ad = self.datasets[vi].get('angle_data')
        if ad is None or ad.empty:
            return
        fn  = ad['frame_num'].to_numpy()
        idx = int(np.argmin(np.abs(fn - event.xdata)))
        self.current_frame_idx = max(0, min(idx, len(ad) - 1))
        t = self.current_frame_idx / SLOWMO_FPS
        self._markup_frame_lbl.config(
            text=f"Frame {self.current_frame_idx + 1}  ({t:.2f} s)")
        self._markup_show_frames()
        self._redraw_markup_graph()

    def _markup_add_step(self):
        side = self._marking_phase
        if not side or not self.datasets:
            return
        vi = self._marking_video_idx
        if vi >= len(self.datasets):
            return
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
        if not side or not self.datasets:
            return
        vi = self._marking_video_idx
        if vi >= len(self.datasets):
            return
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
        if not side or self._markup_frame is None:
            return
        vi = self._marking_video_idx
        if vi >= len(self.datasets):
            return
        ds = self.datasets[vi]
        total = sum(1 for _, s in ds.get('step_frames', []) if s == side) if ds else 0
        marker_col = C_LEFT if side == 'left' else C_RIGHT
        noun = "LEFT" if side == 'left' else "RIGHT"
        self._markup_count_lbl.config(
            text=f"{total} {noun} step{'s' if total != 1 else ''} marked",
            fg=marker_col)

    # PDF export
    def _generate_pdf(self, output_path, graphs, limbs, measures):
        """Generate a PDF report with selected graphs and outcome measures."""
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
        
        temp_images = []
        
        try:
            # Create PDF
            doc = SimpleDocTemplate(output_path, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#2c3e50'),
                spaceAfter=6,
                alignment=1  # center
            )
            story.append(Paragraph("Gait Analysis Report", title_style))
            story.append(Spacer(1, 0.1*inch))
            
            # Subtitle with date
            subtitle_style = ParagraphStyle(
                'Subtitle',
                parent=styles['Normal'],
                fontSize=11,
                textColor=colors.HexColor('#7f8c8d'),
                alignment=1
            )
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", subtitle_style))
            story.append(Spacer(1, 0.3*inch))
            
            # Save graphs as images and add to PDF
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
                        print(f"Error adding continuous graph image: {e}")
            
            if 'cycles' in graphs:
                story.append(Paragraph("Overlaid Cycles (Normalized Gait Cycles)", styles['Heading2']))
                img_path = self._capture_graph_image('cycles', graphs['cycles'], limbs)
                if img_path:
                    try:
                        story.append(RLImage(img_path, width=graph_width, height=graph_height))
                        story.append(Spacer(1, 0.2*inch))
                        temp_images.append(img_path)
                    except Exception as e:
                        print(f"Error adding cycles graph image: {e}")
            
            # Outcome measures table
            if any(measures.values()) and len(self.datasets) >= 2:
                story.append(PageBreak())
                story.append(Paragraph("Outcome Measures", styles['Heading2']))
                story.append(Spacer(1, 0.15*inch))
                
                # Compute metrics
                metrics = compute_metrics(self.datasets[0], self.datasets[1])
                
                # Build table data
                table_data = [['Measure', 'Change (%)']  ]
                
                measure_labels = PDFExportDialog.OUTCOME_MEASURES
                for key, label in measure_labels.items():
                    if measures.get(key):
                        value = metrics.get(key, 0)
                        value_str = f"{value:+.1f}%" if isinstance(value, (int, float)) else "N/A"
                        table_data.append([label, value_str])
                
                # Create table
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
            
            # Build PDF
            doc.build(story)
            
        finally:
            # Clean up temporary image files
            for img_path in temp_images:
                try:
                    if os.path.exists(img_path):
                        time.sleep(0.05)  # Small delay to ensure file is released
                        os.unlink(img_path)
                except Exception as e:
                    print(f"Warning: Could not delete temp file {img_path}: {e}")
    
    def _capture_graph_image(self, graph_type, graph_options=None, limbs=None):
        try:
            if graph_options is None:
                graph_options = {'show_versions': 'both', 'include_excluded': True}
            if limbs is None:
                limbs = {k: True for k in ('left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle')}
            
            # Create temp file with a proper path
            import tempfile as tf
            temp_dir = tf.gettempdir()
            temp_path = os.path.join(temp_dir, f'gait_graph_{int(time.time()*1000)}_{graph_type}.png')
            
            # Save current graph state
            orig_mode = self.show_overlaid_cycles
            orig_graph_mode = self.graph_show_mode
            orig_joint_visibility = self.joint_visibility.copy()
            orig_show_excluded = getattr(self, '_show_excluded_in_pdf', True)
            
            try:
                # Set view mode
                if graph_type == 'continuous':
                    self.show_overlaid_cycles = False
                elif graph_type == 'cycles':
                    self.show_overlaid_cycles = True
                else:
                    return None
                
                # Set version display mode
                self.graph_show_mode = graph_options.get('show_versions', 'both')
                
                # Set joint visibility
                self.joint_visibility = limbs.copy()
                
                # Set excluded areas visibility
                self._show_excluded_in_pdf = graph_options.get('include_excluded', True)
                
                self.redraw_graph()
                self.update()
                
                # Capture the matplotlib figure - save with high DPI
                self._fig.savefig(temp_path, dpi=150, bbox_inches='tight', format='png')
                
                # Ensure file is written to disk
                import gc
                gc.collect()
                time.sleep(0.1)
                
                # Verify file exists before returning
                if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                    raise Exception(f"Image file not created or empty: {temp_path}")
                
                return temp_path
            finally:
                # Restore original state
                self.show_overlaid_cycles = orig_mode
                self.graph_show_mode = orig_graph_mode
                self.joint_visibility = orig_joint_visibility
                self._show_excluded_in_pdf = orig_show_excluded
                self.redraw_graph()
        except Exception as e:
            print(f"Error capturing graph: {e}")
            return None

    # close
    def _on_close(self):
        self._stop_pf = True
        if self.playing and self._play_after_id:
            self.after_cancel(self._play_after_id)
        plt.close('all')
        self.destroy()

# session temp directory
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