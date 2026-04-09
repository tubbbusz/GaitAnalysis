# imports
import atexit
import gc
import os
import signal
import sys
import tempfile
import threading
import time
import urllib.request
from collections import OrderedDict, namedtuple

# third party imports
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import pyglet
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks
import tkinter as tk
from tkinter import filedialog, messagebox

pyglet.font.add_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Coiny-Cyrillic.ttf'))


def resource_path(filename):
    base = sys._MEIPASS if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, filename)

# settings
SLOWMO_FPS    = 240
FILTER_CUTOFF = 6
FILTER_ORDER  = 4

# memory / disk
SAVE_HEIGHT  = 540
JPEG_QUALITY = 65
CACHE_FRAMES = 96

# model path
MODEL_PATH = resource_path("pose_landmarker_full.task")
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "pose_landmarker/pose_landmarker_full/float16/1/"
              "pose_landmarker_full.task")

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading PoseLandmarker model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


# landmark indices
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
    (23,25),(24,26),(25,27),(26,28),                            # upper legs
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),           # lower legs / feet
]

# skeleton
DRAW_THICKNESS      = 8
USE_WORLD_LANDMARKS = True
SKELETON_LINE_COL   = (0, 0, 255)   # cyan lines

JOINT_NAME_TO_LANDMARK = {
    'left_hip':    PoseLandmark.LEFT_HIP,
    'right_hip':   PoseLandmark.RIGHT_HIP,
    'left_knee':   PoseLandmark.LEFT_KNEE,
    'right_knee':  PoseLandmark.RIGHT_KNEE,
    'left_ankle':  PoseLandmark.LEFT_ANKLE,
    'right_ankle': PoseLandmark.RIGHT_ANKLE,
}

# reverse mapping: landmark index -> joint name
LANDMARK_TO_JOINT_NAME = {v: k for k, v in JOINT_NAME_TO_LANDMARK.items()}

# joint colors
JOINT_COLORS_MPL = {
    'left_hip':    '#c0392b', 'left_knee':   '#d35400', 'left_ankle':  '#8e44ad',
    'right_hip':   '#2471a3', 'right_knee':  '#1a5276', 'right_ankle': '#148f77',
}

def hex_to_bgr(hex_color):
    """Convert hex color string to BGR tuple for OpenCV."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)  # OpenCV uses BGR, not RGB

# Convert MPL colors to BGR for skeleton drawing
JOINT_COLORS_BGR = {k: hex_to_bgr(v) for k, v in JOINT_COLORS_MPL.items()}
C_RIGHT = '#4a90d9'   # right step marker
C_LEFT  = '#e8913a'   # left step marker

# ui colors
BG      = "#f0f0f0"   # window background
BG2     = "#d6d6d6"   # header / panels
BG3     = "#c8c8c8"   # cards / toolbar
BG_VID  = "#d8d8d8"   # video canvas
BG_PLOT = "#b8b8b8"   # graph axes
BG_INIT = "#c0c0c0"   # graph before data

# text colors
ACCENT  = "#3a083a"   # purple (logo, headers)
TEXT    = "#1a1a1a"
SUBTEXT = "#4a4a4a"
GREEN   = '#27ae60'
RED     = '#c0392b'

# label colors
C_V1      = "#4a1a44"   # video 1 (purple tint)
C_V2      = "#2e6b40"   # video 2 (green tint)
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
    m  = NORMATIVE_GAIT[jt]["mean"]
    sd = np.std(m) * 0.15
    NORMATIVE_GAIT[jt]["lower"] = m - sd
    NORMATIVE_GAIT[jt]["upper"] = m + sd
NORMATIVE_X = np.linspace(0, 100, 100)


SimpleLandmark = namedtuple('SimpleLandmark', ['x', 'y', 'visibility'])

GRAY_BGR = (128, 128, 128)

# analysis functions
def draw_pose_landmarks_on_frame(frame_bgr, pixel_landmarks, joint_visibility=None):
    h, w = frame_bgr.shape[:2]
    default_line_col = (200, 200, 200)  # light gray for non-tracked joints
    
    def _resolve_color(landmark_idx, base_color):
        if landmark_idx not in LANDMARK_TO_JOINT_NAME:
            return base_color
        jname = LANDMARK_TO_JOINT_NAME[landmark_idx]
        if joint_visibility is not None and not joint_visibility.get(jname, True):
            return GRAY_BGR
        return JOINT_COLORS_BGR[jname]

    # Draw lines connecting landmarks
    for s, e in POSE_CONNECTIONS:
        if s < len(pixel_landmarks) and e < len(pixel_landmarks):
            ls, le = pixel_landmarks[s], pixel_landmarks[e]
            if ls.visibility > 0.5 and le.visibility > 0.5:
                # Use color of end joint if tracked, otherwise try start
                if e in LANDMARK_TO_JOINT_NAME:
                    line_color = _resolve_color(e, default_line_col)
                elif s in LANDMARK_TO_JOINT_NAME:
                    line_color = _resolve_color(s, default_line_col)
                else:
                    line_color = default_line_col
                
                cv2.line(frame_bgr,
                         (int(ls.x*w), int(ls.y*h)),
                         (int(le.x*w), int(le.y*h)),
                         line_color, DRAW_THICKNESS)
    
    # Draw joint circles with colors
    for idx, lm in enumerate(pixel_landmarks):
        if lm.visibility > 0.5:
            if idx in LANDMARK_TO_JOINT_NAME:
                joint_color = _resolve_color(idx, (255, 0, 255))
            else:
                joint_color = (255, 0, 255)  # magenta for non-tracked joints
            
            cv2.circle(frame_bgr, (int(lm.x*w), int(lm.y*h)), 4, joint_color, -1)
    
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
    # calculate angle
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


def butter_lowpass_filter(data, cutoff=4.0, fs=240.0, order=4):
    # butterworth filter
    nyq  = 0.5 * fs
    b, a = butter(order, cutoff/nyq, btype='low', analog=False)
    return filtfilt(b, a, data)


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

        # adaptive outlier removal: reject peaks whose interval to the
        # previous accepted peak is less than 50% of the median inter-peak
        # interval.  this eliminates noise blips without cascading rejection.
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


def select_video_paths():
    root = tk.Tk()
    root.withdraw()
    paths = filedialog.askopenfilenames(
        title="Select two videos for comparison",
        filetypes=[("Video files", "*.mov *.mp4 *.avi *.m4v"), ("All files", "*.*")])
    root.destroy()
    return list(paths)


# joint definitions
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
        # smooth directions
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
            
            # add peak/trough analysis per direction using scipy
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
    
    # also log raw per-frame data to a second csv
    raw_path = os.path.join(output_dir, f'raw_angles_{vid_name}_{timestamp}.csv')
    df_raw = df_w.copy()
    df_raw['_smoothed_direction'] = smoothed if '_direction' in df_w.columns else 'unknown'
    df_raw.to_csv(raw_path, index=False)
    
    # write summary csv
    if rows:
        fieldnames = rows[0].keys()
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    return output_path, raw_path


# video processing
def process_video(video_path, ann_dir, progress_cb, status_cb):
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

    world_rows, pixel_rows, landmarks, landmark_depths = [], [], [], []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        progress_cb(frame_count / max(1, total))

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms  = int((frame_count / fps) * 1000)
        result = landmarker.detect_for_video(mp_img, ts_ms)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            pixel_lm = result.pose_landmarks[0]
            world_lm = (result.pose_world_landmarks[0]
                        if result.pose_world_landmarks else pixel_lm)

            # Save raw frame (skeleton drawn at render time for dynamic visibility)
            raw = frame.copy()
            h, w = raw.shape[:2]
            if SAVE_HEIGHT and h > SAVE_HEIGHT:
                nw = int(w * SAVE_HEIGHT / h)
                raw = cv2.resize(raw, (nw, SAVE_HEIGHT), interpolation=cv2.INTER_AREA)
            raw_path = os.path.join(ann_dir, f"raw_{frame_count:06d}.jpg")
            cv2.imwrite(raw_path, raw, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            landmarks.append((raw_path, [SimpleLandmark(lm.x, lm.y, lm.visibility) for lm in pixel_lm]))

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

            # Capture landmark depths (z-coordinates from world landmarks)
            depth_row = {'frame_num': frame_count}
            for i, landmark in enumerate(world_lm):
                depth_row[f'joint_{i}'] = landmark.z
            landmark_depths.append(depth_row)

            world_rows.append(w_row)
            pixel_rows.append(p_row)
        else:
            landmarks.append(None)
            landmark_depths.append(None)

    cap.release()
    landmarker.close()

    df_w = pd.DataFrame(world_rows)
    df_p = pd.DataFrame(pixel_rows)
    df_depths = pd.DataFrame([d for d in landmark_depths if d is not None])

    # Log raw diagnostics BEFORE correction
    try:
        diag_path, raw_path = _log_direction_diagnostics(df_w, df_p, video_path, 0)
        status_cb(f"Diagnostics saved: {os.path.basename(diag_path)}")
    except Exception as e:
        status_cb(f"Diagnostic logging failed: {e}")

    for df in (df_w, df_p):
        for col in df.columns:
            if col not in ('frame_num', '_direction'):
                df[col] = butter_lowpass_filter(df[col], FILTER_CUTOFF, SLOWMO_FPS, FILTER_ORDER)
                df[col] = df[col].astype(np.float32)

    # Drop the helper column before further use
    for df in (df_w, df_p):
        if '_direction' in df.columns:
            df.drop('_direction', axis=1, inplace=True)

    ad    = df_w if USE_WORLD_LANDMARKS else df_p
    suggested_steps = detect_steps(ad, fps=SLOWMO_FPS)
    del world_rows, pixel_rows
    gc.collect()

    return {
        'df_world':       df_w,
        'df_pixel':       df_p,
        'angle_data':     ad,
        'step_frames':    [],
        'suggested_step_frames': suggested_steps,
        'excluded_regions': [],
        'all_landmarks':  landmarks,   # list of (raw_path, pixel_landmarks) tuples
        'landmark_depths': df_depths,
    }


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
        # Handle new (raw_path, pixel_landmarks) tuple format
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
    ("r",             "Generate step suggestions (all videos)"),
    ("g",             "Toggle visibility of suggested steps"),
    ("d",             "Clear manual steps (all videos)"),
    ("",              None),
    ("Exclusions",    None),
    ("Right-click",   "Drag on graph to exclude region"),
    ("Clr Excl btn",  "Clear all excluded regions"),
    ("h / H",         "This help screen"),
]

# dashboard
class GaitAnalysisDashboard(tk.Tk):

    def __init__(self):
        super().__init__()
        self.configure(bg=BG)
        self.title("Gait Analysis")
        self.geometry("1400x860")

        # state
        self.datasets           = []
        self.video_names        = ["Video 1", "Video 2"]
        self.current_frame_idx  = 0
        self.total_frames       = 0
        self.progress           = 0.0
        self._status_msg        = tk.StringVar(value="Loading…")

        self.df_world   = pd.DataFrame()
        self.df_pixel   = pd.DataFrame()
        self.angle_data = pd.DataFrame()

        self.joint_visibility = {k: True for k in
            ('left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle')}

        self.graph_show_mode      = 'both'
        self.show_overlaid_cycles = False
        self.resample_cycles      = False
        self.resample_length      = 100
        self.show_mean            = False
        self.mean_only            = False
        self.show_normative       = True
        self.show_data            = True
        self.active_dataset_idx   = 0
        self.manual_step_mode     = True  # Always active - no toggle needed
        self.manual_side          = 'right'  # Deprecated - auto-detected from suggested steps
        self.show_suggestions     = True
        self.playing              = False
        self._play_after_id       = None
        self._graph_dragging      = False
        self._exclusion_selecting = False
        self._exclusion_start     = None

        self._cache     = FrameCache()
        self._stop_pf   = False
        self._pf_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._pf_thread.start()

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

        tk.Label(hdr, text="novita gait analysis",
                 font=("Coiny Cyrillic", 17), bg=BG2, fg=ACCENT,
                 ).pack(side='left', pady=(6, 0))

        tk.Button(hdr, text="? Help", font=("Helvetica", 9),
                  bg=BG3, fg=TEXT, relief='flat', padx=8,
                  command=self._show_help
                  ).pack(side='right', padx=10, pady=8)

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


        # main layout: graph+videos left, metrics right
        main = tk.Frame(self, bg=BG)
        main.pack(fill='both', expand=True, padx=8, pady=(4, 0))

        left = tk.Frame(main, bg=BG)
        left.pack(side='left', fill='both', expand=True)

        # graph row: graph on left, interactive legend on right
        graph_row = tk.Frame(left, bg=BG2)
        graph_row.pack(fill='x')

        gf = tk.Frame(graph_row, bg=BG2)
        gf.pack(side='left', fill='both', expand=True)

        # add h-scrollbar below the canvas
        self._graph_hbar = tk.Scrollbar(gf, orient='horizontal', command=self._on_scrollbar_drag)
        self._graph_hbar.pack(fill='x', side='bottom')

        self._fig, self._ax = plt.subplots(figsize=(11, 4.2), dpi=100)
        self._ax_xlim_full = None # set after data loads
        self._ax_xlim_per_mode = {}  # Track zoom limits per graph mode
        self._zoom_level = 1.0
        self._last_scroll_event = None  # Store for zoom on cursor position
        self._fig.patch.set_facecolor(BG)
        self._ax.set_facecolor(BG_INIT)
        self._mpl_canvas = FigureCanvasTkAgg(self._fig, master=gf)
        self._mpl_canvas.get_tk_widget().pack(fill='x')
        self._mpl_canvas.mpl_connect('button_press_event',   self._on_graph_click)
        self._mpl_canvas.mpl_connect('motion_notify_event',  self._on_graph_drag)
        self._mpl_canvas.mpl_connect('button_release_event', self._on_graph_release)

        # interactive legend panel
        self._legend_frame = tk.Frame(graph_row, bg=BG2, width=140)
        self._legend_frame.pack(side='right', fill='y', padx=(2, 0))
        self._legend_frame.pack_propagate(False)

        # v1/v2 indicator at top
        ind_frame = tk.Frame(self._legend_frame, bg=BG2)
        ind_frame.pack(fill='x', padx=6, pady=(8, 2))
        # draw dashed line indicator for v1
        c1 = tk.Canvas(ind_frame, width=30, height=10, bg=BG2, highlightthickness=0)
        c1.pack(side='left')
        c1.create_line(2, 5, 28, 5, fill=TEXT, dash=(4, 3), width=2)
        tk.Label(ind_frame, text="V1", font=("Helvetica", 8), bg=BG2, fg=SUBTEXT).pack(side='left', padx=(2, 8))
        # draw solid line indicator for v2
        c2 = tk.Canvas(ind_frame, width=30, height=10, bg=BG2, highlightthickness=0)
        c2.pack(side='left')
        c2.create_line(2, 5, 28, 5, fill=TEXT, width=2)
        tk.Label(ind_frame, text="V2", font=("Helvetica", 8), bg=BG2, fg=SUBTEXT).pack(side='left', padx=2)

        ttk_sep = tk.Frame(self._legend_frame, bg=SUBTEXT, height=1)
        ttk_sep.pack(fill='x', padx=6, pady=(4, 4))

        # joint toggle entries
        self._legend_items = {}
        for joint, col in JOINT_COLORS_MPL.items():
            jf = tk.Frame(self._legend_frame, bg=BG3, cursor='hand2')
            jf.pack(fill='x', padx=4, pady=2)

            # color swatch with line samples
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

        # display toggles: mean, data, normal
        self._display_btns = {}
        for label, key in [("Mean", "mean"), ("Data", "data"), ("Normal", "normal")]:
            btn = tk.Button(self._legend_frame, text=label,
                        font=("Helvetica", 7, "bold"), bg=BG3, fg=TEXT,
                        relief='flat', cursor='hand2',
                        command=lambda k=key: self._toggle_display_option(k))
            btn.pack(fill='x', padx=4, pady=1)
            self._display_btns[key] = btn
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

        # colour key
        tk.Frame(right, bg=BG2, height=1).pack(fill='x', padx=8, pady=(8, 0))
        kf = tk.Frame(right, bg=BG2)
        kf.pack(fill='x', padx=10, pady=4)
        for colour, label in ((C_RIGHT, "Right strike"), (C_LEFT, "Left strike")):
            row = tk.Frame(kf, bg=BG2)
            row.pack(anchor='w', pady=1)
            tk.Canvas(row, width=12, height=12, bg=colour,
                      highlightthickness=0).pack(side='left')
            tk.Label(row, text=f"  {label}", font=("Helvetica", 8),
                     bg=BG2, fg=SUBTEXT).pack(side='left')

        # toolbar / status bar
        bottom = tk.Frame(self, bg=BG2, height=36)
        bottom.pack(fill='x', side='bottom')
        bottom.pack_propagate(False)

        self._prog_canvas = tk.Canvas(bottom, height=4, bg=BG3, highlightthickness=0)
        self._prog_canvas.pack(fill='x', side='top')

        bar = tk.Frame(bottom, bg=BG2)
        bar.pack(fill='x', expand=True, padx=6)

        btn_cfg = dict(bg=BG3, fg=TEXT, relief='flat',
                       font=("Helvetica", 8), padx=5, pady=1, cursor='hand2',
                       activebackground=ACCENT, activeforeground='white')

        buttons = [
            ("Prev",       self._prev_frame),
            ("Next",       self._next_frame),
            ("Play",       self._toggle_play),
            ("Cycles",     self._toggle_cycles),
            ("World/Px",   self._toggle_world),
            ("Graph V",    self._cycle_graph_view),
            ("Active V",   self._toggle_active),
            ("Auto steps", self._recompute_steps),
            ("Show Sugg",  self._toggle_suggestions),
            ("Clr steps",  self._clear_steps),
            ("Clr Excl",   self._clear_exclusions),
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
        self._status_msg.set("Processing videos… 0%")

        session = self._session
        spill1 = os.path.join(session.path, "vid1")
        spill2 = os.path.join(session.path, "vid2")
        os.makedirs(spill1, exist_ok=True)
        os.makedirs(spill2, exist_ok=True)

        results          = [None, None]
        results_progress = [0.0, 0.0]

        def _process(i, path, ann_dir):
            def _prog(p): results_progress[i] = p
            def _stat(s): self.after(0, lambda: self._status_msg.set(s))
            results[i] = process_video(path, ann_dir, _prog, _stat)

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
            self.datasets     = results
            self.df_world     = results[0]['df_world']
            self.df_pixel     = results[0]['df_pixel']
            self.angle_data   = results[0]['angle_data']
            self.total_frames = max(len(results[0]['all_landmarks']),
                                    len(results[1]['all_landmarks']))
            # initialize zoom tracking with range limited to smallest video
            min_video_frames = min(len(results[0]['all_landmarks']),
                                   len(results[1]['all_landmarks']))
            if not self.angle_data.empty:
                data_min = self.angle_data['frame_num'].min()
                self._ax_xlim_full = (data_min, data_min + min_video_frames)
                # store per-view limits
                self._ax_xlim_per_mode['both'] = self._ax_xlim_full
                for i, ds in enumerate(results[:2]):
                    v_frames = len(ds['all_landmarks'])
                    self._ax_xlim_per_mode[f'v{i+1}'] = (data_min, data_min + v_frames)
            self.progress     = 1.0
            self.show_overlaid_cycles = False
            self.resample_cycles = False
            self._status_msg.set("Ready  —  press H for help")
            self.refresh()

        t1 = threading.Thread(target=_process, args=(0, video_paths[0], spill1), daemon=True)
        t2 = threading.Thread(target=_process, args=(1, video_paths[1], spill2), daemon=True)
        t1.start()
        t2.start()
        self.after(300, _poll_loading)

    # key bindings
    def _bind_keys(self):
        # bind scroll events to canvas - works with ctrl for zoom, without ctrl for scrub
        canvas = self._mpl_canvas.get_tk_widget()
        canvas.bind('<MouseWheel>', self._on_canvas_scroll)
        canvas.bind('<Button-4>', self._on_canvas_scroll)  # linux scroll up
        canvas.bind('<Button-5>', self._on_canvas_scroll)  # linux scroll down
        self.bind('<Key-1>',        lambda e: self._prev_frame())
        self.bind('<Key-2>',        lambda e: self._next_frame())
        self.bind('<Key-9>',        lambda e: self._toggle_play())
        self.bind('q',              lambda e: self._on_close())
        self.bind('w',              lambda e: self._toggle_world())
        self.bind('c',              lambda e: self._toggle_cycles())
        self.bind('s',              lambda e: self._toggle_resample())
        self.bind('m',              lambda e: self._toggle_mean())
        self.bind('v',              lambda e: self._cycle_graph_view())
        self.bind('t',              lambda e: self._toggle_active())
        self.bind('r',              lambda e: self._recompute_steps())
        self.bind('g',              lambda e: self._toggle_suggestions())
        self.bind('d',              lambda e: self._clear_steps())
        self.bind('<space>',        lambda e: self._add_manual_step())
        self.bind('<BackSpace>',    lambda e: self._delete_nearest_step())
        self.bind('<Delete>',       lambda e: self._delete_nearest_step())
        self.bind('h',              lambda e: self._show_help())
        self.bind('H',              lambda e: self._show_help())
        self.bind('z',              lambda e: self._reset_zoom())
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

    def _get_filtered_angle_data(self, angle_data, excluded_regions):
        if not excluded_regions or angle_data.empty:
            return angle_data
        
        # create mask for rows not in any excluded region
        mask = pd.Series([True] * len(angle_data), index=angle_data.index)
        for start_frame, end_frame in excluded_regions:
            mask &= ~((angle_data['frame_num'] >= start_frame) & (angle_data['frame_num'] < end_frame))
        
        return angle_data[mask].reset_index(drop=True)

    def _region_crosses_exclusion(self, start_frame, end_frame, excluded_regions):
        for ex_start, ex_end in excluded_regions:
            # check if exclusion overlaps with this region
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
            if s <= prev_e:  # overlapping or adjacent
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

        # calculate max cycle length for overlaid mode before plotting
        max_cycle_length = self.resample_length
        if self.show_overlaid_cycles and self.resample_cycles:
            for ds in dfg:
                ad = ds['angle_data']
                excluded = ds.get('excluded_regions', [])
                ad_filtered = self._get_filtered_angle_data(ad, excluded)
                
                sf = ds.get('step_frames', [])
                norm = _to_fnums(ad_filtered, sf)
                
                for joint in JOINT_COLORS_MPL.keys():
                    side = 'left' if joint.startswith('left_') else 'right'
                    strikes = [f for f, s in norm if s == side]
                    if len(strikes) < 2: continue
                    
                    for i in range(len(strikes)-1):
                        # skip step pairs that cross exclusion regions
                        if self._region_crosses_exclusion(strikes[i], strikes[i+1], excluded):
                            continue
                            
                        seg = ad_filtered[(ad_filtered['frame_num'] >= strikes[i]) & (ad_filtered['frame_num'] <= strikes[i+1])]
                        if not seg.empty:
                            max_cycle_length = max(max_cycle_length, len(seg))

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

                # Build exclusion mask for this dataset
                frames = ad['frame_num'].values
                excl_mask = np.zeros(len(frames), dtype=bool)
                for ex_s, ex_e in excluded:
                    excl_mask |= (frames >= ex_s) & (frames < ex_e)

                for joint, col in JOINT_COLORS_MPL.items():
                    if joint not in ad.columns:
                        continue
                    vis = self.joint_visibility.get(joint, True)
                    values = ad[joint].values.copy().astype(float)

                    if excluded:
                        # Gray line in excluded regions
                        gray_vals = values.copy()
                        gray_vals[~excl_mask] = np.nan
                        ax.plot(frames, gray_vals, color='#999999', lw=1.2,
                                alpha=0.5, linestyle=ls, zorder=2)

                    # Non-excluded data
                    clean_vals = values.copy()
                    if excluded:
                        clean_vals[excl_mask] = np.nan
                    if vis:
                        ax.plot(frames, clean_vals, color=col, lw=1.4,
                                alpha=0.85, linestyle=ls, zorder=3,
                                label=f"{joint.replace('_',' ').title()} V{si+1}")
                    else:
                        ax.plot(frames, clean_vals, color='#999999', lw=1.2,
                                alpha=0.5, linestyle=ls, zorder=2)
                    plotted = True

                nsteps  = _to_fnums(ad_filtered, sf)
                fn_min  = int(ad['frame_num'].min())
                fn_max  = int(ad['frame_num'].max())
                for f, side in nsteps:
                    if fn_min <= f <= fn_max:
                        ax.axvline(f, color=C_RIGHT if side=='right' else C_LEFT,
                                   lw=0.8, alpha=0.7, linestyle=ls)

                # draw suggested steps if enabled (exclude those in excluded regions)
                if self.show_suggestions:
                    ssf = ds.get('suggested_step_frames', [])
                    suggested = _to_fnums(ad_filtered, ssf)
                    
                    # show triangles at top of graph for suggested steps
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

            # secondary time axis (top)
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

                for joint, col in JOINT_COLORS_MPL.items():
                    vis = self.joint_visibility.get(joint, True)
                    side    = 'left' if joint.startswith('left_') else 'right'
                    strikes = [f for f, s in norm if s == side]
                    if len(strikes) < 2: continue

                    cycles, lengths = [], []
                    excluded = ds.get('excluded_regions', [])
                    
                    for i in range(len(strikes)-1):
                        # skip step pairs that cross exclusion regions
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
                    ok  = [0.8*med <= l <= 1.2*med for l in lengths]

                    for (x, y), good in zip(cycles, ok):
                        if not self.show_data: break
                        if self.resample_cycles:
                            t = np.linspace(0, 1, len(y))
                            y = interp1d(t, y)(np.linspace(0, 1, max_cycle_length))
                            x = np.arange(max_cycle_length)
                        if vis:
                            c = col if good else C_OUTLIER
                        else:
                            c = '#999999'
                        ax.plot(x, y, color=c, alpha=0.25, lw=0.8, linestyle=ls)

                    if self.resample_cycles and self.show_mean:
                        inliers = []
                        for (x, y), good in zip(cycles, ok):
                            if not good: continue
                            t = np.linspace(0, 1, len(y))
                            inliers.append(interp1d(t, y)(np.linspace(0, 1, max_cycle_length)))
                        if inliers:
                            mean_c = np.nanmean(np.vstack(inliers), axis=0)
                            mean_col = col if vis else '#999999'
                            ax.plot(np.arange(len(mean_c)), mean_c,
                                    color=mean_col, lw=2.2, linestyle=ls,
                                    label=f"{joint.replace('_',' ').title()} V{si+1} mean" if vis else None)

            if self.resample_cycles and self.show_normative:
                # Create x-axis for normative data matching graph x-range
                norm_x_resampled = np.linspace(0, max_cycle_length, 100)
                
                for jt_key in ('hip', 'knee', 'ankle'):
                    vl = self.joint_visibility.get(f'left_{jt_key}', False)
                    vr = self.joint_visibility.get(f'right_{jt_key}', False)
                    if vl or vr:
                        d = NORMATIVE_GAIT[jt_key]
                        ax.fill_between(norm_x_resampled, d['lower'], d['upper'],
                                        color=C_NORM, alpha=0.12,
                                        label=f'{jt_key.title()} norm.')
        # draw exclusion overlays for all datasets
        for ds in dfg:
            excluded = ds.get('excluded_regions', [])
            for start_frame, end_frame in excluded:
                ax.axvspan(start_frame, end_frame, alpha=0.05, color='darkgray', zorder=1)

        # set axis limits after plotting to prevent auto-scaling
        # determine appropriate limits based on current view
        if self.show_overlaid_cycles and self.resample_cycles:
            # overlaid cycles with resample: x-axis scales to longest cycle
            ax.set_xlim(0, max_cycle_length)
        elif self._ax_xlim_full is not None:
            # continuous view or non-resampled cycles: use frame range
            ax.set_xlim(self._ax_xlim_full)

        # use subplots_adjust instead of tight_layout to maintain static graph size
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
                # New format: (raw_path, pixel_landmarks) tuple
                if isinstance(entry, tuple):
                    frame = self._cache.get(vi, self.current_frame_idx, store)
                    pixel_lm = entry[1]
                else:
                    # Legacy: annotated frame path or numpy array
                    frame = self._cache.get(vi, self.current_frame_idx, store)

            canvas.delete('all')
            if frame is None:
                canvas.create_text(cw//2, ch//2, text="No frame",
                                   fill=SUBTEXT, font=("Helvetica", 10))
                continue

            # Draw skeleton at render time with dynamic joint visibility
            if pixel_lm is not None:
                frame = frame.copy()
                draw_pose_landmarks_on_frame(frame, pixel_lm, self.joint_visibility)

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Determine if this video should be shown based on graph_show_mode
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
            canvas._img = img   # keep reference
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
        # preserve zoom level when scrubbing
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
            # don't reset zoom - just move scrubber line
        elif event.button == 3 and event.inaxes == self._ax:
            # right-click to start exclusion selection
            self._exclusion_selecting = True
            self._exclusion_start = event.xdata

    def _on_graph_drag(self, event):
        if self._graph_dragging and event.inaxes == self._ax:
            self._seek_from_event(event)

    def _on_graph_release(self, event):
        if event.button == 1:
            self._graph_dragging = False
        elif event.button == 3:
            # Right-click release - finalize exclusion
            if self._exclusion_selecting and self._exclusion_start is not None and event.xdata is not None:
                start_frame = round(self._exclusion_start)
                end_frame = round(event.xdata)
                # Ensure start < end
                if start_frame > end_frame:
                    start_frame, end_frame = end_frame, start_frame
                # Only add if selection is meaningful (> 1 frame)
                if end_frame > start_frame:
                    target_datasets = []
                    
                    # Determine which dataset(s) to apply exclusion to
                    if self.graph_show_mode == 'v1' and len(self.datasets) >= 1:
                        target_datasets = [self.datasets[0]]
                    elif self.graph_show_mode == 'v2' and len(self.datasets) >= 2:
                        target_datasets = [self.datasets[1]]
                    elif self.graph_show_mode == 'both' and len(self.datasets) >= 2:
                        # Apply to both videos when in "both" view
                        target_datasets = [self.datasets[0], self.datasets[1]]
                    else:
                        target_datasets = [self._active_ds()]
                    
                    # Apply exclusion to all target datasets
                    for target_ds in target_datasets:
                        if target_ds:
                            target_ds.setdefault('excluded_regions', []).append((start_frame, end_frame))
                            target_ds['excluded_regions'] = self._merge_exclusion_regions(
                                target_ds['excluded_regions'])
                    
                    self._status_msg.set(f"Excluded frames {start_frame}-{end_frame} in both videos")
                    self._recompute_steps()  # Recompute with filtered data
            self._exclusion_selecting = False
            self._exclusion_start = None

    def _on_canvas_scroll(self, event):
        if self.angle_data is None or self.angle_data.empty:
            return
        
        # disable scrolling in overlaid cycles mode (viewing normalized cycles)
        if self.show_overlaid_cycles:
            return
        
        # store event for zoom-on-cursor
        self._last_scroll_event = event
        
        # determine scroll direction (works on windows, mac, and linux)
        if hasattr(event, 'delta'):
            scroll_dir = 1 if event.delta > 0 else -1
        elif hasattr(event, 'num'):
            scroll_dir = 1 if event.num == 4 else -1
        else:
            return
        
        # check if ctrl is held - determines zoom vs pan
        ctrl_held = bool(event.state & 0x0004)  # 0x0004 is ctrl mask
        
        if ctrl_held:
            self._on_graph_zoom(scroll_dir)
        else:
            self._on_graph_pan(scroll_dir)
    
    def _on_graph_pan(self, direction):
        cur_xlim = self._ax.get_xlim()
        xlim_full = self._get_current_xlim_full()
        full_min, full_max = xlim_full
        pan_amount = (full_max - full_min) * 0.1 * direction  # 10% of range
        
        new_xlim_min = cur_xlim[0] - pan_amount
        new_xlim_max = cur_xlim[1] - pan_amount
        
        # clamp to data bounds
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
        # get current axis limits
        cur_xlim = self._ax.get_xlim()
        xlim_full = self._get_current_xlim_full()
        full_min, full_max = xlim_full
        full_range = full_max - full_min
        
        # zoom factor (direction = -1 for zoom in, 1 for zoom out)
        zoom_factor = 0.8 if direction < 0 else 1.2
        
        # get zoom center from cursor position if available
        zoom_center = None
        if self._last_scroll_event:
            if hasattr(self._last_scroll_event, 'xdata') and self._last_scroll_event.xdata is not None:
                zoom_center = self._last_scroll_event.xdata
            elif hasattr(self._last_scroll_event, 'x'):
                # Convert pixel coordinates to data coordinates
                try:
                    # get the axes bounding box in display coordinates
                    bbox = self._ax.get_window_extent()
                    # get event position in display space
                    x_pixel = self._last_scroll_event.x
                    # convert to axes space (0 to 1)
                    x_normalized = (x_pixel - bbox.x0) / bbox.width
                    # convert to data space
                    if 0 <= x_normalized <= 1:
                        zoom_center = cur_xlim[0] + x_normalized * (cur_xlim[1] - cur_xlim[0])
                except:
                    pass
        
        # fallback to center if we couldn't determine cursor position
        if zoom_center is None:
            zoom_center = (cur_xlim[0] + cur_xlim[1]) / 2
        
        # calculate new width
        new_width = (cur_xlim[1] - cur_xlim[0]) * zoom_factor
        
        # clamp to minimum zoom (show full range)
        if new_width >= full_range:
            new_xlim_min = full_min
            new_xlim_max = full_max
        else:
            # calculate position relative to cursor
            rel_pos = (zoom_center - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
            new_xlim_min = zoom_center - new_width * rel_pos
            new_xlim_max = new_xlim_min + new_width
            
            # pan to stay in bounds
            if new_xlim_min < full_min:
                new_xlim_min = full_min
                new_xlim_max = full_min + new_width
            if new_xlim_max > full_max:
                new_xlim_max = full_max
                new_xlim_min = full_max - new_width
        
        # apply new limits
        self._ax.set_xlim(new_xlim_min, new_xlim_max)
        self._update_scrollbar()
        self._mpl_canvas.draw_idle()
    
    def _get_current_xlim_full(self):
        if self.show_overlaid_cycles and self.resample_cycles:
            return (0, self.resample_length)
        return self._ax_xlim_full or (0, 100)
    
    def _update_scrollbar(self):
        xlim_full = self._get_current_xlim_full()
        if xlim_full is None:
            return
        
        cur_xlim = self._ax.get_xlim()
        full_min, full_max = xlim_full
        full_range = full_max - full_min
        
        # calculate scrollbar position
        first = (cur_xlim[0] - full_min) / full_range
        last = (cur_xlim[1] - full_min) / full_range
        
        # set scrollbar
        self._graph_hbar.set(first, last)
    
    def _on_scrollbar_drag(self, *args):
        # disable scrollbar in overlaid cycles mode (viewing normalized cycles)
        if self.show_overlaid_cycles:
            return
        
        xlim_full = self._get_current_xlim_full()
        if xlim_full is None or not args:
            return
        
        full_min, full_max = xlim_full
        full_range = full_max - full_min
        cur_width = self._ax.get_xlim()[1] - self._ax.get_xlim()[0]
        
        # parse tkinter scrollbar callback arguments
        # scrollbar sends either ('moveto', fraction) or ('scroll', amount, 'units')
        if args[0] == 'moveto':
            # direct position command from scrollbar drag
            fraction = float(args[1])
            new_xlim_min = full_min + fraction * full_range
            new_xlim_max = new_xlim_min + cur_width
            # Clamp to bounds
            if new_xlim_max > full_max:
                new_xlim_max = full_max
                new_xlim_min = full_max - cur_width
        elif args[0] == 'scroll':
            # arrow or page scroll
            amount = int(args[1])
            units = args[2] if len(args) > 2 else 'units'
            scroll_amount = full_range * 0.1 * amount
            cur_xlim = self._ax.get_xlim()
            new_xlim_min = cur_xlim[0] - scroll_amount
            new_xlim_max = cur_xlim[1] - scroll_amount
            # Clamp to bounds
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
            # preserve zoom level during playback
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
        # preserve zoom level
        current_xlim = self._ax.get_xlim()
        self._show_video_frames()
        self.redraw_graph()
        self._ax.set_xlim(current_xlim)
        self._update_scrollbar()
        self._update_status()

    def _next_frame(self):
        if self.playing: return
        self.current_frame_idx = min(self._active_max_index(), self.current_frame_idx + 1)
        # preserve zoom level
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
        # auto-enable resampling when entering overlaid cycles mode
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

    def _toggle_world(self):
        global USE_WORLD_LANDMARKS
        USE_WORLD_LANDMARKS = not USE_WORLD_LANDMARKS
        key = 'df_world' if USE_WORLD_LANDMARKS else 'df_pixel'
        for ds in self.datasets:
            if key in ds: ds['angle_data'] = ds[key]
        self.angle_data = self.df_world if USE_WORLD_LANDMARKS else self.df_pixel
        self._status_msg.set("World landmarks" if USE_WORLD_LANDMARKS else "Pixel landmarks")
        self.redraw_graph()

    def _cycle_graph_view(self):
        modes  = ['both', 'v1', 'v2']
        labels = {'both': 'Both', 'v1': 'V1 only', 'v2': 'V2 only'}
        self.graph_show_mode = modes[(modes.index(self.graph_show_mode)+1) % 3]
        # Update zoom limits for this view mode
        if self.graph_show_mode in self._ax_xlim_per_mode:
            self._ax_xlim_full = self._ax_xlim_per_mode[self.graph_show_mode]
        # Reset zoom when switching views
        if self._ax_xlim_full is not None:
            self._ax.set_xlim(self._ax_xlim_full)
        self._status_msg.set(f"Graph: {labels[self.graph_show_mode]}")
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
        self._update_display_btn_visuals()
        self.redraw_graph()

    def _update_display_btn_visuals(self):
        active_map = {
            'mean': self.show_mean,
            'data': self.show_data,
            'normal': self.show_normative,
        }
        for key, btn in self._display_btns.items():
            if not self.show_overlaid_cycles:
                btn.config(bg=BG2, fg='#999999', state='disabled')
            elif active_map[key]:
                btn.config(bg=ACCENT, fg='white', state='normal')
            else:
                btn.config(bg=BG3, fg=TEXT, state='normal')

    def _toggle_manual_step(self):
        self.manual_step_mode = not self.manual_step_mode
        self._status_msg.set(
            f"Manual step mode {'ON' if self.manual_step_mode else 'OFF'}  [{self.manual_side}]")

    def _set_manual_side(self, side):
        self.manual_side = side
        self._status_msg.set(f"Manual side: {side.upper()}")

    def _add_manual_step(self):
        # determine which dataset to add step to based on graph view mode
        if self.graph_show_mode == 'v1' and len(self.datasets) >= 1:
            ds = self.datasets[0]
        elif self.graph_show_mode == 'v2' and len(self.datasets) >= 2:
            ds = self.datasets[1]
        else:
            ds = self._active_ds()
        
        if not ds or ds['angle_data'].empty: return
        idx = min(self.current_frame_idx, len(ds['angle_data'])-1)
        fn  = int(ds['angle_data']['frame_num'].iloc[idx])
        
        # auto-detect foot from nearest suggested step
        suggested = ds.get('suggested_step_frames', [])
        detected_foot = 'right'  # default
        
        if suggested:
        # find nearest suggested step
            nearest_idx = min(range(len(suggested)), key=lambda i: abs(suggested[i][0] - fn))
            detected_foot = suggested[nearest_idx][1]
        
        ds.setdefault('step_frames', []).append((fn, detected_foot))
        ds['step_frames'].sort(key=lambda x: x[0])
        self._status_msg.set(f"Added {detected_foot} step @ frame {fn}")
        self._update_metrics_panel()
        self.redraw_graph()

    def _delete_nearest_step(self):
        # determine which dataset to delete step from based on graph view mode
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
        self._status_msg.set(f"Removed step @ frame {removed[0]}")
        self._update_metrics_panel()
        self.redraw_graph()

    def _recompute_steps(self):
        if not self.datasets: 
            return
        total_steps = 0
        for ds in self.datasets:
            if ds and 'angle_data' in ds:
                # use filtered angle data excluding regions
                excluded = ds.get('excluded_regions', [])
                filtered_ad = self._get_filtered_angle_data(ds['angle_data'], excluded)
                ds['suggested_step_frames'] = detect_steps(
                    filtered_ad, fps=SLOWMO_FPS)
                total_steps += len(ds.get('suggested_step_frames', []))
        self._status_msg.set(f"Auto steps: {total_steps} suggestions generated across all videos")
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
        self._status_msg.set("Steps cleared from all videos")
        self._update_metrics_panel()
        self.redraw_graph()

    def _clear_exclusions(self):
        if not self.datasets:
            return
        for ds in self.datasets:
            if ds:
                ds['excluded_regions'] = []
        self._status_msg.set("All exclusions cleared")
        self._recompute_steps()  # recalculate steps with full data
        self.redraw_graph()

    # help window
    def _show_help(self):
        win = tk.Toplevel(self)
        win.title("Keyboard Shortcuts")
        win.configure(bg=BG)
        win.resizable(False, False)
        for i, (key, desc) in enumerate(HELP_TEXT):
            if desc is None:          # section header
                tk.Label(win, text=key, font=("Helvetica", 10, "bold"),
                         bg=BG, fg=ACCENT, anchor='w'
                         ).grid(row=i, column=0, columnspan=2,
                                sticky='w', padx=16, pady=(10, 2))
            elif key == "":           # spacer
                tk.Label(win, text="", bg=BG, height=0
                         ).grid(row=i, column=0)
            else:
                tk.Label(win, text=key, font=("Courier", 9, "bold"),
                         bg=BG, fg=ACCENT, width=18, anchor='w'
                         ).grid(row=i, column=0, sticky='w', padx=(16, 4), pady=1)
                tk.Label(win, text=desc, font=("Helvetica", 9),
                         bg=BG, fg=TEXT, anchor='w'
                         ).grid(row=i, column=1, sticky='w', padx=(0, 16), pady=1)
        tk.Button(win, text="Close", command=win.destroy,
                  bg=ACCENT, fg='white', relief='flat', padx=14
                  ).grid(row=len(HELP_TEXT), column=0, columnspan=2, pady=14)

    # close
    def _on_close(self):
        self._stop_pf = True
        if self.playing and self._play_after_id:
            self.after_cancel(self._play_after_id)
        plt.close('all')
        self.destroy()

# temp dir
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
    ensure_model()
    app = GaitAnalysisDashboard()
    app._session = SessionTempDir()
    app._status_msg.set("Select two videos to begin")
    app.mainloop()
    app._session.cleanup()


if __name__ == "__main__":
    main()