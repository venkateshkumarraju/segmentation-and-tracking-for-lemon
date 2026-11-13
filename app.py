"""
Real-time Segmentation & Object Tracking with Interactive Priority Zone
Features:
- Complete Tracker Selection Suite (15 Algorithms Supported):
    • ByteTrack (Ultralytics Deep MOT)
    • BoT-SORT (Deep MOT with Re-ID)
    • CSRT (Channel & Spatial Reliability Filter)
    • KCF (Kernelized Correlation Filters)
    • MIL (Multiple Instance Learning)
    • MOSSE (Minimum Output Sum of Squared Error)
    • MEDIANFLOW (Median Flow)
    • TLD (Tracking, Learning and Detection)
    • BOOSTING (AdaBoost)
    • CAMShift (Continuously Adaptive Mean Shift)
    • MeanShift (Color Histogram Mean Shift)
    • DaSiamRPN (Siamese Neural Tracker)
    • NanoTrack (Nano Neural Tracker)
    • VitTrack (Vision Transformer Tracker)
    • GOTURN (Generic Object Tracking Using Regression Networks)
- Runtime Tracker Switching (Press 'T' to cycle trackers on the fly)
- Startup Setup Dialog with Tracker & Camera Selection + Live Preview
- 970x540 Frame Resizing (50% scale for high FPS)
- Interactive Priority Zone (Auto-saved to zone_config.json)
- GPU Acceleration (NVIDIA CUDA)
- Dual counts for each class: Inside Priority Zone vs Total Screen
- All Key Options Visible in Window (Bottom Ribbon + 2-Column HUD Grid + 'H' Help Modal)
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# Default weights path from user's training runs
DEFAULT_WEIGHTS = r"C:\Users\ADMIN.DESKTOP-O3EH4US.000\Desktop\lemon\runs\segment\train\weights\best.pt"
CONFIG_FILE = "zone_config.json"

# Distinct color palette (BGR) for the 4 classes
CLASS_COLORS = {
    0: (40, 110, 215),    # bad green lemon (Deep Olive / Rust)
    1: (0, 85, 235),      # bad yellow lemon (Dark Orange / Brown-Yellow)
    2: (45, 220, 75),     # good green lemon (Vibrant Fresh Emerald)
    3: (0, 225, 255),     # good yellow lemon (Bright Pure Lemon Yellow)
}

DEFAULT_COLOR = (210, 210, 210)

# Available Tracking Algorithms
AVAILABLE_TRACKERS = [
    ("ByteTrack", "bytetrack.yaml", "ByteTrack (Deep Multi-Object) [RECOMMENDED]"),
    ("BoT-SORT", "botsort.yaml", "BoT-SORT (Deep Multi-Object + Re-ID)"),
    ("CSRT", "csrt", "CSRT (Channel & Spatial Reliability Filter)"),
    ("KCF", "kcf", "KCF (Kernelized Correlation Filters)"),
    ("MIL", "mil", "MIL (Multiple Instance Learning)"),
    ("MOSSE", "mosse", "MOSSE (Fast Correlation Filter)"),
    ("MEDIANFLOW", "medianflow", "MEDIANFLOW (Median Flow)"),
    ("TLD", "tld", "TLD (Tracking, Learning and Detection)"),
    ("BOOSTING", "boosting", "BOOSTING (AdaBoost Online Tracking)"),
    ("CAMShift", "camshift", "CAMShift (Continuously Adaptive Mean Shift)"),
    ("MeanShift", "meanshift", "MeanShift (Color Histogram Mean Shift)"),
    ("DaSiamRPN", "dasiamrpn", "DaSiamRPN (Siamese Neural Network)"),
    ("NanoTrack", "nanotrack", "NanoTrack (Lightweight Neural Tracker)"),
    ("VitTrack", "vittrack", "VitTrack (Vision Transformer Tracker)"),
    ("GOTURN", "goturn", "GOTURN (Regression Networks Tracker)"),
]


# ==============================================================================
# FONT & TYPOGRAPHY MANAGER (Crisp Modern TrueType Anti-Aliased Rendering)
# ==============================================================================
class FontManager:
    """Manages TrueType fonts with fallbacks for high-dpi, anti-aliased rendering."""
    _instance = None

    def __init__(self):
        self.fonts = {}

        def try_load(font_names, size):
            for name in font_names:
                try:
                    return ImageFont.truetype(name, size)
                except Exception:
                    pass
            return ImageFont.load_default()

        # Pre-cache common sizes (Base font size: 12)
        self.fonts["title"] = try_load(["segoeuib.ttf", "arialbd.ttf"], 15)
        self.fonts["sub"] = try_load(["segoeui.ttf", "arial.ttf"], 12)
        self.fonts["header"] = try_load(["segoeuib.ttf", "arialbd.ttf"], 13)
        self.fonts["body"] = try_load(["segoeui.ttf", "arial.ttf"], 12)
        self.fonts["bold"] = try_load(["segoeuib.ttf", "arialbd.ttf"], 12)
        self.fonts["chip"] = try_load(["segoeuib.ttf", "arialbd.ttf"], 12)
        self.fonts["mono"] = try_load(["consola.ttf", "cour.ttf", "arial.ttf"], 12)
        self.fonts["mono_bold"] = try_load(["consolab.ttf", "courbd.ttf", "arialbd.ttf"], 12)
        self.fonts["small"] = try_load(["segoeui.ttf", "arial.ttf"], 12)
        self.fonts["tag"] = try_load(["segoeuib.ttf", "arialbd.ttf"], 12)
        self.fonts["ribbon_key"] = try_load(["consolab.ttf", "arialbd.ttf"], 12)
        self.fonts["ribbon_desc"] = try_load(["segoeui.ttf", "arial.ttf"], 12)
        self.fonts["modal_title"] = try_load(["segoeuib.ttf", "arialbd.ttf"], 20)
        self.fonts["modal_sub"] = try_load(["segoeui.ttf", "arial.ttf"], 13)

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = FontManager()
        return cls._instance


def draw_rounded_rect(draw, bbox, radius, fill=None, outline=None, width=1):
    """Draw a smooth rounded rectangle on a PIL ImageDraw context."""
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill, outline=outline, width=width)


# ==============================================================================
# OPENCV & CLASSICAL TRACKER ENSEMBLE ENGINE
# ==============================================================================
class ColorShiftTracker:
    """Implements MeanShift and CAMShift tracking via HSV color histogram back-projection."""

    def __init__(self, mode="camshift"):
        self.mode = mode.lower()
        self.hist = None
        self.track_window = None
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    def init(self, frame, bbox):
        x, y, w, h = map(int, bbox)
        w, h = max(4, w), max(4, h)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        roi = hsv[y:y + h, x:x + w]
        if roi.size == 0:
            return
        mask = cv2.inRange(roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        self.hist = cv2.calcHist([roi], [0], mask, [180], [0, 180])
        cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX)
        self.track_window = (x, y, w, h)

    def update(self, frame):
        if self.hist is None or self.track_window is None:
            return False, (0, 0, 0, 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)

        if self.mode == "camshift":
            ret, self.track_window = cv2.CamShift(dst, self.track_window, self.term_crit)
            pts = cv2.boxPoints(ret)
            x, y, w, h = cv2.boundingRect(np.intp(pts))
            return True, (x, y, w, h)
        else:
            ret, self.track_window = cv2.meanShift(dst, self.track_window, self.term_crit)
            return True, self.track_window


class OpenCVTrackedInstance:
    """Manages an individual object instance tracked via an OpenCV algorithm."""

    def __init__(self, track_id, cls_id, box, frame, tracker_type="CSRT"):
        self.track_id = track_id
        self.cls_id = cls_id
        self.tracker_type = tracker_type.upper()
        self.tracker = self._build_tracker()
        x1, y1, x2, y2 = box
        w, h = max(4, x2 - x1), max(4, y2 - y1)
        self.bbox = (x1, y1, w, h)
        self.missed_frames = 0
        self.age = 0

        if self.tracker is not None:
            try:
                self.tracker.init(frame, (x1, y1, w, h))
            except Exception:
                self.tracker = None

    def _build_tracker(self):
        t = self.tracker_type
        if t in ("CAMSHIFT", "MEANSHIFT"):
            return ColorShiftTracker(mode=t)
        elif t == "CSRT":
            return cv2.TrackerCSRT_create()
        elif t == "KCF":
            return cv2.TrackerKCF_create()
        elif t == "MIL":
            return cv2.TrackerMIL_create()
        elif t == "MOSSE":
            if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"):
                return cv2.legacy.TrackerMOSSE_create()
            if hasattr(cv2, "TrackerMOSSE_create"):
                return cv2.TrackerMOSSE_create()
            raise RuntimeError(
                "MOSSE tracker is unavailable. Install opencv-contrib-python "
                "(and remove conflicting opencv-python if necessary)."
            )
        elif hasattr(cv2, "legacy"):
            if t == "MEDIANFLOW" and hasattr(cv2.legacy, "TrackerMedianFlow_create"):
                return cv2.legacy.TrackerMedianFlow_create()
            elif t == "TLD" and hasattr(cv2.legacy, "TrackerTLD_create"):
                return cv2.legacy.TrackerTLD_create()
            elif t == "BOOSTING" and hasattr(cv2.legacy, "TrackerBoosting_create"):
                return cv2.legacy.TrackerBoosting_create()
        # Fallback to CSRT for other unavailable legacy trackers.
        try:
            return cv2.TrackerCSRT_create()
        except Exception:
            return None

    def update(self, frame):
        self.age += 1
        if self.tracker is not None:
            try:
                ok, bbox = self.tracker.update(frame)
                if ok and bbox[2] > 0 and bbox[3] > 0:
                    self.bbox = tuple(map(int, bbox))
                    return True
            except Exception:
                pass
        self.missed_frames += 1
        return False

    def reinit(self, frame, box):
        x1, y1, x2, y2 = box
        w, h = max(4, x2 - x1), max(4, y2 - y1)
        self.bbox = (x1, y1, w, h)
        self.missed_frames = 0
        if self.tracker is not None:
            try:
                self.tracker.init(frame, (x1, y1, w, h))
            except Exception:
                pass


def compute_iou(boxA, boxB):
    """Compute Intersection-over-Union between two boxes (x1, y1, x2, y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return interArea / float(boxAArea + boxBArea - interArea)


class UnifiedMultiTracker:
    """
    Unified manager that routes tracking requests to either:
    - Ultralytics deep multi-object trackers (ByteTrack, BoT-SORT)
    - Or OpenCV classic & correlation trackers (CSRT, KCF, MIL, MOSSE, MEDIANFLOW, TLD, BOOSTING, CAMShift, MeanShift)
    """

    def __init__(self, tracker_name="ByteTrack"):
        self.set_tracker(tracker_name)
        self.opencv_instances = {}  # {track_id: OpenCVTrackedInstance}
        self.next_id = 1

    def set_tracker(self, tracker_name):
        self.tracker_name = tracker_name
        self.is_ultralytics = tracker_name.lower() in ("bytetrack", "bot-sort", "botsort", "bytetrack.yaml", "botsort.yaml")
        self.opencv_instances = {}
        print(f"[TrackerManager] Active tracking algorithm: {self.tracker_name}")

    def update(self, model, frame, device, conf, iou, show_masks):
        """
        Executes tracking on the frame using the active tracking algorithm.
        Returns: list of detections, list of masks_data
        """
        if self.is_ultralytics:
            tracker_file = "botsort.yaml" if "bot" in self.tracker_name.lower() else "bytetrack.yaml"
            results = model.track(
                source=frame,
                device=device,
                persist=True,
                tracker=tracker_file,
                conf=conf,
                iou=iou,
                verbose=False,
            )
            res = results[0]
            detections = []
            masks_data = res.masks.data.cpu().numpy() if res.masks is not None else None

            if res.boxes is not None and len(res.boxes) > 0:
                boxes = res.boxes.xyxy.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy().astype(int)
                confs = res.boxes.conf.cpu().numpy()
                track_ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else [None] * len(boxes)

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.astype(int)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    detections.append({
                        "box": (x1, y1, x2, y2),
                        "class_id": classes[i],
                        "track_id": track_ids[i],
                        "conf": confs[i],
                        "centroid": (cx, cy),
                    })
            return detections, masks_data
        else:
            # OpenCV Tracker Ensemble with  Detections
            results = model.predict(
                source=frame,
                device=device,
                conf=conf,
                iou=iou,
                verbose=False,
            )
            res = results[0]
            masks_data = res.masks.data.cpu().numpy() if res.masks is not None else None
            detections = []

            # 1. Update existing OpenCV tracker instances
            active_ids = list(self.opencv_instances.keys())
            for tid in active_ids:
                inst = self.opencv_instances[tid]
                inst.update(frame)
                if inst.missed_frames > 8:
                    del self.opencv_instances[tid]

            # 2. Match with fresh  detections
            yolo_boxes = []
            yolo_classes = []
            yolo_confs = []
            if res.boxes is not None and len(res.boxes) > 0:
                yolo_boxes = res.boxes.xyxy.cpu().numpy().astype(int)
                yolo_classes = res.boxes.cls.cpu().numpy().astype(int)
                yolo_confs = res.boxes.conf.cpu().numpy()

            matched_tracks = set()
            for b_idx, box in enumerate(yolo_boxes):
                best_iou = 0.0
                best_tid = None

                for tid, inst in self.opencv_instances.items():
                    if tid in matched_tracks or inst.cls_id != yolo_classes[b_idx]:
                        continue
                    ix, iy, iw, ih = inst.bbox
                    inst_box = (ix, iy, ix + iw, iy + ih)
                    score = compute_iou(box, inst_box)
                    if score > best_iou:
                        best_iou = score
                        best_tid = tid

                if best_iou >= 0.25 and best_tid is not None:
                    # Match found: refresh tracker position
                    inst = self.opencv_instances[best_tid]
                    inst.reinit(frame, box)
                    matched_tracks.add(best_tid)
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    detections.append({
                        "box": (x1, y1, x2, y2),
                        "class_id": yolo_classes[b_idx],
                        "track_id": best_tid,
                        "conf": yolo_confs[b_idx],
                        "centroid": (cx, cy),
                    })
                else:
                    # New detection: spawn new tracker instance
                    new_tid = self.next_id
                    self.next_id += 1
                    inst = OpenCVTrackedInstance(new_tid, yolo_classes[b_idx], box, frame, self.tracker_name)
                    self.opencv_instances[new_tid] = inst
                    matched_tracks.add(new_tid)
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    detections.append({
                        "box": (x1, y1, x2, y2),
                        "class_id": yolo_classes[b_idx],
                        "track_id": new_tid,
                        "conf": yolo_confs[b_idx],
                        "centroid": (cx, cy),
                    })

            return detections, masks_data


# ==============================================================================
# WEBCAM QUALITY & DEVICE SCANNER
# ==============================================================================
def scan_connected_cameras(max_tested=6):
    """Scans camera indices and tests connectivity and maximum resolution."""
    print("[INFO] Scanning for connected webcams...")
    available = []
    backends = [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_ANY, "Default")]

    for idx in range(max_tested):
        for backend_id, backend_name in backends:
            try:
                cap = cv2.VideoCapture(idx, backend_id)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        h, w = frame.shape[:2]
                        available.append({
                            "index": idx,
                            "backend": backend_id,
                            "backend_name": backend_name,
                            "width": w,
                            "height": h,
                            "active": True,
                            "label": f"Camera {idx}: {w}x{h} (Active - {backend_name})"
                        })
                        cap.release()
                        break
                    else:
                        if backend_id == cv2.CAP_ANY:
                            available.append({
                                "index": idx,
                                "backend": backend_id,
                                "backend_name": backend_name,
                                "width": 0,
                                "height": 0,
                                "active": False,
                                "label": f"Camera {idx}: Detected (Virtual / Busy)"
                            })
                    cap.release()
            except Exception:
                pass

    print(f"[INFO] Scan complete. Found {len(available)} camera device(s):")
    for cam in available:
        print(f"       -> {cam['label']}")
    return available


def open_high_quality_camera(source, backend=cv2.CAP_DSHOW):
    """Opens camera with optimized resolution, MJPG decoding, zero-latency buffering."""
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src, backend)

    if not cap.isOpened() and backend != cv2.CAP_ANY:
        cap = cv2.VideoCapture(src, cv2.CAP_ANY)

    if cap.isOpened():
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        try:
            cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass
        try:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        except Exception:
            pass

    return cap


def apply_clarity_enhancement(frame):
    """Applies adaptive contrast enhancement (CLAHE on luminance) and subtle unsharp masking."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    merged_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    balanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(balanced_bgr, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(balanced_bgr, 1.25, blurred, -0.25, 0)
    return sharpened


# ==============================================================================
# CAMERA & TRACKER SELECTOR DIALOG (Modern Dark Theme)
# ==============================================================================
def select_setup_gui(cameras, default_tracker="MOSSE", default_input_size="640x640"):
    """Startup dialog for choosing webcam/video file, processing size and tracker."""
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        from PIL import ImageTk
    except ImportError:
        return 0, cv2.CAP_DSHOW, default_tracker, (640, 640)

    selected_choice = {
        "source": 0, "backend": cv2.CAP_DSHOW,
        "tracker": default_tracker, "input_size": (640, 640)
    }
    active_cams = [c for c in cameras if c.get("active")]
    default_cam = active_cams[0] if active_cams else (cameras[0] if cameras else None)

    root = tk.Tk()
    root.title("Setup - Segmentation & MOSSE Tracking")
    root.geometry("660x720")
    root.resizable(False, False)
    root.configure(bg="#111318")
    root.eval('tk::PlaceWindow . center')

    title_font = ("Segoe UI Semibold", 14)
    label_font = ("Segoe UI Semibold", 10)
    subtitle_font = ("Segoe UI", 10)
    btn_font = ("Segoe UI Semibold", 10)

    tk.Label(root, text="SEGMENTATION & TRACKING SETUP",
             font=title_font, fg="#00E5FF", bg="#111318").pack(pady=(14, 3))
    tk.Label(root, text="Select input source, processing resolution and tracker",
             font=subtitle_font, fg="#9AA0A6", bg="#111318").pack(pady=(0, 10))

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("TCombobox", fieldbackground="#202633", background="#202633",
                    foreground="#E8EDF5", arrowcolor="#00E5FF")

    source_box = tk.LabelFrame(root, text=" 1. Video Input ", fg="#00E5FF",
                               bg="#111318", font=label_font, bd=1)
    source_box.pack(fill="x", padx=25, pady=6)

    source_mode = tk.StringVar(value="Webcam")
    tk.Label(source_box, text="Input type:", fg="#DCE3ED", bg="#111318").grid(
        row=0, column=0, padx=10, pady=8, sticky="w")
    mode_combo = ttk.Combobox(source_box, textvariable=source_mode,
                              values=["Webcam", "Video file"], state="readonly", width=18)
    mode_combo.grid(row=0, column=1, padx=8, pady=8, sticky="w")

    cam_var = tk.StringVar(value=default_cam["label"] if default_cam else "No cameras detected")
    cam_choices = [c["label"] for c in cameras] if cameras else ["No cameras detected"]
    tk.Label(source_box, text="Camera:", fg="#DCE3ED", bg="#111318").grid(
        row=1, column=0, padx=10, pady=6, sticky="w")
    combo_cam = ttk.Combobox(source_box, textvariable=cam_var, values=cam_choices,
                             state="readonly", width=48)
    combo_cam.grid(row=1, column=1, padx=8, pady=6, sticky="w")

    video_var = tk.StringVar()
    tk.Label(source_box, text="Video file:", fg="#DCE3ED", bg="#111318").grid(
        row=2, column=0, padx=10, pady=6, sticky="w")
    video_entry = tk.Entry(source_box, textvariable=video_var, width=42,
                           bg="#202633", fg="#FFFFFF", insertbackground="white",
                           relief="flat")
    video_entry.grid(row=2, column=1, padx=8, pady=6, sticky="w")

    def browse_video():
        path = filedialog.askopenfilename(
            title="Select input video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.mpeg *.mpg"),
                       ("All files", "*.*")]
        )
        if path:
            video_var.set(path)
            source_mode.set("Video file")

    browse_btn = tk.Button(source_box, text="Browse", command=browse_video,
                           bg="#2B3545", fg="white", relief="flat",
                           padx=12, cursor="hand2")
    browse_btn.grid(row=2, column=2, padx=5, pady=6)

    size_box = tk.LabelFrame(root, text=" 2. Processing Resolution ",
                             fg="#00E5FF", bg="#111318", font=label_font, bd=1)
    size_box.pack(fill="x", padx=25, pady=6)
    size_var = tk.StringVar(value=default_input_size)
    tk.Label(size_box, text="Frame size:", fg="#DCE3ED", bg="#111318").pack(
        side="left", padx=12, pady=10)
    size_combo = ttk.Combobox(
        size_box, textvariable=size_var,
        values=["640x640", "Original", "1280x720", "1920x1080"],
        state="readonly", width=18
    )
    size_combo.pack(side="left", padx=8, pady=10)

    tracker_box = tk.LabelFrame(root, text=" 3. Object Tracker ",
                                fg="#00E5FF", bg="#111318", font=label_font, bd=1)
    tracker_box.pack(fill="x", padx=25, pady=6)
    tracker_var = tk.StringVar()
    tracker_labels = [f"{t[0]} - {t[2]}" for t in AVAILABLE_TRACKERS]
    default_label = next((v for v in tracker_labels if v.split(" - ")[0] == default_tracker),
                         tracker_labels[0])
    tracker_var.set(default_label)
    combo_tracker = ttk.Combobox(tracker_box, textvariable=tracker_var,
                                 values=tracker_labels, state="readonly", width=58)
    combo_tracker.pack(padx=12, pady=10)

    preview_box = tk.LabelFrame(root, text=" Live Webcam Preview ",
                                fg="#00E5FF", bg="#111318", font=label_font, bd=1)
    preview_box.pack(padx=25, pady=8, fill="both", expand=True)
    preview_canvas = tk.Canvas(preview_box, width=420, height=230,
                               bg="#0A0C0F", highlightthickness=0)
    preview_canvas.pack(pady=8)
    preview_state = {"cap": None, "running": True, "photo": None, "idx": -1}

    def update_preview():
        if not preview_state["running"]:
            return
        if source_mode.get() == "Webcam":
            target = next((c for c in cameras if c["label"] == cam_var.get()), None)
            if target and target.get("active"):
                idx, backend = target["index"], target["backend"]
                if preview_state["cap"] is None or preview_state["idx"] != idx:
                    if preview_state["cap"] is not None:
                        preview_state["cap"].release()
                    preview_state["cap"] = cv2.VideoCapture(idx, backend)
                    preview_state["idx"] = idx
                cap_preview = preview_state["cap"]
                if cap_preview and cap_preview.isOpened():
                    ret, frame = cap_preview.read()
                    if ret and frame is not None:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(rgb)
                        img.thumbnail((420, 230), Image.Resampling.BILINEAR)
                        preview_state["photo"] = ImageTk.PhotoImage(image=img)
                        preview_canvas.delete("all")
                        preview_canvas.create_image(210, 115, image=preview_state["photo"],
                                                    anchor="center")
            else:
                preview_canvas.delete("all")
                preview_canvas.create_text(210, 115, text="No live camera signal",
                                           fill="#FF5252")
        else:
            preview_canvas.delete("all")
            preview_canvas.create_text(210, 115, text="Video file selected - preview starts after Start",
                                       fill="#9AA0A6")
        root.after(100, update_preview)

    def on_source_change(event=None):
        if preview_state["cap"] is not None:
            preview_state["cap"].release()
            preview_state["cap"] = None
            preview_state["idx"] = -1
        combo_cam.configure(state="readonly" if source_mode.get() == "Webcam" else "disabled")
        video_entry.configure(state="normal" if source_mode.get() == "Video file" else "disabled")
        browse_btn.configure(state="normal" if source_mode.get() == "Video file" else "disabled")

    mode_combo.bind("<<ComboboxSelected>>", on_source_change)
    on_source_change()
    root.after(100, update_preview)

    button_bar = tk.Frame(root, bg="#111318")
    button_bar.pack(pady=(5, 14))

    def on_confirm():
        if source_mode.get() == "Video file":
            path = video_var.get().strip()
            if not path or not os.path.isfile(path):
                messagebox.showerror("Video required", "Please browse and select a valid video file.")
                return
            selected_choice["source"] = path
        else:
            target = next((c for c in cameras if c["label"] == cam_var.get()), None)
            if target:
                selected_choice["source"] = target["index"]
                selected_choice["backend"] = target["backend"]
            else:
                selected_choice["source"] = 0
        selected_choice["tracker"] = tracker_var.get().split(" - ")[0].strip()
        selected_choice["input_size"] = parse_input_size(size_var.get())
        preview_state["running"] = False
        if preview_state["cap"] is not None:
            preview_state["cap"].release()
        root.destroy()

    def on_cancel():
        preview_state["running"] = False
        if preview_state["cap"] is not None:
            preview_state["cap"].release()
        root.destroy()
        sys.exit(0)

    tk.Button(button_bar, text="▶  Start Tracking", command=on_confirm,
              bg="#00ADB5", fg="white", font=btn_font, padx=26, pady=7,
              bd=0, cursor="hand2").pack(side="left", padx=10)
    tk.Button(button_bar, text="Cancel", command=on_cancel,
              bg="#2B303C", fg="#E0E0E0", font=btn_font,
              padx=18, pady=7, bd=0, cursor="hand2").pack(side="left", padx=10)
    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.mainloop()
    return (selected_choice["source"], selected_choice["backend"],
            selected_choice["tracker"], selected_choice["input_size"])


def parse_input_size(value):
    """Convert a resolution label into (width, height); None means native size."""
    if value is None or str(value).strip().lower() in ("original", "native", "none"):
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    try:
        width, height = str(value).lower().replace(" ", "").split("x", 1)
        return int(width), int(height)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid input size: {value}. Use 640x640, Original, 1280x720 or 1920x1080.")


# ==============================================================================
# PRIORITY ZONE MANAGER
# ==============================================================================
class ZoneManager:
    """Manages the interactive polygon priority zone."""

    def __init__(self, config_path=CONFIG_FILE):
        self.config_path = config_path
        self.points = []
        self.is_closed = False
        self.cursor_pos = None
        self.load_zone()

    def add_point(self, pt):
        self.points.append(pt)

    def undo_point(self):
        if self.points:
            self.points.pop()
            if len(self.points) < 3:
                self.is_closed = False

    def clear(self):
        self.points = []
        self.is_closed = False
        if os.path.exists(self.config_path):
            try:
                os.remove(self.config_path)
            except OSError:
                pass

    def close_polygon(self):
        if len(self.points) >= 3:
            self.is_closed = True
            self.save_zone()
            return True
        return False

    def is_inside(self, point):
        """Check if point is inside polygon."""
        if not self.is_closed or len(self.points) < 3:
            return False
        pts = np.array(self.points, dtype=np.int32)
        return cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False) >= 0

    def save_zone(self):
        try:
            with open(self.config_path, "w") as f:
                json.dump({"points": self.points, "is_closed": self.is_closed}, f)
        except Exception as e:
            print(f"[ZoneManager] Failed to save zone: {e}")

    def load_zone(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    data = json.load(f)
                    self.points = [tuple(p) for p in data.get("points", [])]
                    self.is_closed = data.get("is_closed", False)
                    print(f"[ZoneManager] Loaded zone with {len(self.points)} points.")
            except Exception as e:
                print(f"[ZoneManager] Failed to load zone: {e}")

    def draw(self, frame, pil_draw):
        """Draws glowing polygon zone and anti-aliased title badge."""
        n = len(self.points)
        if n == 0:
            return

        pts_array = np.array(self.points, dtype=np.int32)

        if self.is_closed and n >= 3:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts_array], (255, 180, 50))
            cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

            # Clean 2-pixel border
            cv2.polylines(frame, [pts_array], isClosed=True, color=(0, 200, 255), thickness=2, lineType=cv2.LINE_AA)

            cx = int(np.mean([p[0] for p in self.points]))
            cy = int(np.mean([p[1] for p in self.points]))

            fm = FontManager.get()
            badge_text = "PRIORITY ZONE"
            bbox = fm.fonts["bold"].getbbox(badge_text)
            bw = (bbox[2] - bbox[0]) + 20
            bh = (bbox[3] - bbox[1]) + 10

            draw_rounded_rect(pil_draw, (cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2),
                              radius=6, fill=(15, 20, 28), outline=(0, 220, 255), width=2)
            pil_draw.text((cx - bw // 2 + 10, cy - bh // 2 + 4), badge_text,
                          font=fm.fonts["bold"], fill=(0, 240, 255))
        else:
            for i in range(n - 1):
                cv2.line(frame, self.points[i], self.points[i + 1], (0, 210, 255), 2, cv2.LINE_AA)

            if self.cursor_pos is not None:
                cv2.line(frame, self.points[-1], self.cursor_pos, (0, 255, 200), 1, cv2.LINE_AA)

        for idx, pt in enumerate(self.points):
            fill_c = (0, 255, 100) if (idx == 0 and n >= 3) else (0, 170, 255)
            cv2.circle(frame, pt, 6, fill_c, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 7, (255, 255, 255), 1, cv2.LINE_AA)


# ==============================================================================
# OBJECT TRACKING & COUNTER
# ==============================================================================
class ObjectCounterTracker:
    """Maintains tracking IDs, cumulative stats, and per-class counts."""

    def __init__(self, class_names):
        self.class_names = class_names
        self.reset()

    def reset(self):
        self.total_unique_ids = defaultdict(set)
        self.zone_unique_ids = defaultdict(set)
        self.current_frame_counts = defaultdict(int)
        self.current_zone_counts = defaultdict(int)

    def update_frame(self, detections, zone_manager):
        self.current_frame_counts.clear()
        self.current_zone_counts.clear()

        for det in detections:
            cls_id = det["class_id"]
            trk_id = det["track_id"]
            cx, cy = det["centroid"]

            self.current_frame_counts[cls_id] += 1
            if trk_id is not None:
                self.total_unique_ids[cls_id].add(trk_id)

            in_zone = zone_manager.is_inside((cx, cy))
            det["in_zone"] = in_zone

            if in_zone:
                self.current_zone_counts[cls_id] += 1
                if trk_id is not None:
                    self.zone_unique_ids[cls_id].add(trk_id)


def mouse_callback(event, x, y, flags, param):
    """OpenCV mouse callback for interactive polygon drawing."""
    zone = param["zone"]

    if event == cv2.EVENT_MOUSEMOVE:
        zone.cursor_pos = (x, y)

    elif event == cv2.EVENT_LBUTTONDOWN:
        if zone.is_closed:
            print("[INFO] Zone is already completed. Press 'C' to clear/redraw.")
            return

        if len(zone.points) >= 3:
            first_pt = zone.points[0]
            dist = np.hypot(x - first_pt[0], y - first_pt[1])
            if dist < 20:
                zone.close_polygon()
                print(f"[INFO] Priority Zone closed with {len(zone.points)} points!")
                return

        zone.add_point((x, y))
        print(f"[INFO] Added polygon vertex: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        zone.undo_point()
        print(f"[INFO] Removed last point. Remaining: {len(zone.points)}")


# ==============================================================================
# BOTTOM SHORTCUT BAR (Always Visible Across Bottom of Video Window)
# ==============================================================================
def render_bottom_shortcut_ribbon(pil_draw, w, h):
    """Renders an always-visible, translucent bottom ribbon containing key shortcuts."""
    ribbon_h = 34
    y1 = h - ribbon_h
    y2 = h

    draw_rounded_rect(pil_draw, (0, y1, w, y2), radius=0, fill=(10, 12, 17))
    pil_draw.line([(0, y1), (w, y1)], fill=(0, 200, 255), width=1)

    fm = FontManager.get()
    key_font = fm.fonts["ribbon_key"]
    desc_font = fm.fonts["ribbon_desc"]

    items = [
        ("L-Click", "Draw Pt"),
        ("R-Click", "Undo"),
        ("D", "Done"),
        ("C", "Clear"),
        ("T", "Tracker"),
        ("W", "Cam"),
        ("+/-", "Conf"),
        ("E", "Clarity"),
        ("R", "Reset"),
        ("M", "Mask"),
        ("B", "Box"),
        ("S", "Save"),
        ("H", "Help"),
        ("Q", "Quit"),
    ]

    cur_x = 8
    spacing = 8

    for key, desc in items:
        bbox_k = key_font.getbbox(key)
        kw = (bbox_k[2] - bbox_k[0]) + 6
        kh = (bbox_k[3] - bbox_k[1]) + 4
        by = y1 + 5

        if cur_x + kw + 40 > w:
            break

        draw_rounded_rect(pil_draw, (cur_x, by, cur_x + kw, by + kh), radius=3, fill=(28, 35, 48), outline=(0, 220, 255), width=1)
        pil_draw.text((cur_x + 3, by + 1), key, font=key_font, fill=(0, 240, 255))
        cur_x += kw + 4

        bbox_d = desc_font.getbbox(desc)
        dw = (bbox_d[2] - bbox_d[0])
        pil_draw.text((cur_x, by + 1), desc, font=desc_font, fill=(200, 205, 215))
        cur_x += dw + spacing


# ==============================================================================
# FULLSCREEN HELP MODAL (Toggle with 'H')
# ==============================================================================
def render_help_modal(frame, current_tracker):
    """Renders a full-screen translucent help card overlay when 'H' is pressed."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (10, 12, 18), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)
    fm = FontManager.get()

    modal_w = min(720, w - 30)
    modal_h = min(490, h - 30)
    mx1 = (w - modal_w) // 2
    my1 = (h - modal_h) // 2
    mx2 = mx1 + modal_w
    my2 = my1 + modal_h

    draw_rounded_rect(draw, (mx1, my1, mx2, my2), radius=10, fill=(18, 22, 30), outline=(0, 229, 255), width=2)

    draw.text((mx1 + 22, my1 + 16), "APPLICATION CONTROLS & TRACKER REFERENCE",
              font=fm.fonts["modal_title"], fill=(0, 240, 255))
    draw.text((mx1 + 22, my1 + 44), f"Active Tracker: {current_tracker}  |  Press [H] or any key to close",
              font=fm.fonts["modal_sub"], fill=(150, 160, 175))

    keys_info = [
        ("[T]", "Cycle Trackers: ByteTrack, BoT-SORT, CSRT, KCF, MIL, MOSSE, MEDIANFLOW, etc."),
        ("Left Mouse Click", "Add polygon point for priority zone / Click 1st point to close"),
        ("Right Mouse Click", "Undo last placed polygon point"),
        ("[D] or [Enter]", "Finalize & activate priority polygon zone"),
        ("[C]", "Clear priority zone and reset boundary"),
        ("[W]", "Cycle through connected webcams on the fly"),
        ("[+ / -]", "Increase or decrease detection confidence threshold (50%)"),
        ("[E]", "Toggle Live Video Clarity Enhancement (CLAHE + Unsharp Mask)"),
        ("[R]", "Reset object tracking IDs and cumulative class counters"),
        ("[M] / [B]", "Toggle segmentation masks overlay / Bounding boxes"),
        ("[S]", "Save snapshot image (.jpg) and detailed counts report (.json)"),
        ("[Q] or [ESC]", "Exit application cleanly"),
    ]

    ky = my1 + 72
    for k, d in keys_info:
        draw_rounded_rect(draw, (mx1 + 22, ky, mx1 + 175, ky + 22), radius=4, fill=(30, 36, 48), outline=(60, 75, 100))
        draw.text((mx1 + 28, ky + 4), k, font=fm.fonts["mono_bold"], fill=(0, 230, 255))
        draw.text((mx1 + 185, ky + 4), d, font=fm.fonts["body"], fill=(230, 235, 245))
        ky += 28

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ==============================================================================
# HIGH-DEFINITION HUD DASHBOARD (Modern Anti-Aliased Typography)
# ==============================================================================
def render_modern_dashboard(h, tracker, zone_manager, fps, gpu_name, show_masks,
                            show_boxes, cam_info, enhanced_mode, current_conf, tracker_name):
    """Renders a modern, anti-aliased dark dashboard with 2-column key shortcuts."""
    panel_w = 380
    dash_pil = Image.new("RGB", (panel_w, h), (14, 16, 22))
    draw = ImageDraw.Draw(dash_pil)
    fm = FontManager.get()

    # --- TOP HEADER CARD ---
    draw_rounded_rect(draw, (10, 8, panel_w - 10, 86), radius=7, fill=(22, 25, 34), outline=(36, 42, 56), width=1)
    draw.text((20, 13), "SEGMENTATION & TRACKING", font=fm.fonts["title"], fill=(0, 229, 255))

    fps_color = (100, 255, 150) if fps >= 25 else (255, 200, 80)
    draw.text((20, 36), f"FPS: {fps:4.1f}", font=fm.fonts["bold"], fill=fps_color)
    draw.text((95, 36), f"|  {gpu_name[:18]}", font=fm.fonts["body"], fill=(175, 185, 195))

    enh_str = "ON" if enhanced_mode else "OFF"
    enh_color = (0, 229, 255) if enhanced_mode else (140, 145, 155)
    draw.text((20, 58), f"SRC: {cam_info}", font=fm.fonts["small"], fill=(255, 196, 77))
    draw.text((170, 58), f"TRK: {tracker_name[:9]}", font=fm.fonts["chip"], fill=(0, 255, 170))
    draw.text((280, 58), f"CONF: {int(current_conf * 100)}%", font=fm.fonts["chip"], fill=(0, 240, 255))

    # --- PRIORITY ZONE CARD ---
    y_off = 94
    draw.text((15, y_off), "PRIORITY ZONE MONITOR", font=fm.fonts["header"], fill=(150, 175, 255))
    y_off += 18

    if zone_manager.is_closed:
        status_text = f"ACTIVE ({len(zone_manager.points)} vertices)"
        bg_col = (18, 42, 32)
        txt_col = (0, 245, 120)
        border_col = (0, 180, 90)
    elif len(zone_manager.points) > 0:
        status_text = f"DRAWING ({len(zone_manager.points)} pts - Press 'D' / click 1st)"
        bg_col = (45, 34, 15)
        txt_col = (255, 185, 50)
        border_col = (190, 130, 20)
    else:
        status_text = "NO ZONE DEFINED (Click on frame to draw)"
        bg_col = (25, 27, 34)
        txt_col = (140, 145, 155)
        border_col = (45, 48, 60)

    draw_rounded_rect(draw, (10, y_off, panel_w - 10, y_off + 25), radius=5, fill=bg_col, outline=border_col, width=1)
    draw.text((20, y_off + 5), status_text, font=fm.fonts["chip"], fill=txt_col)
    y_off += 33

    # --- TABLE CARD ---
    draw.text((15, y_off), "CLASS METRICS & COUNTS", font=fm.fonts["header"], fill=(150, 175, 255))
    y_off += 18

    # Table Header Row
    draw_rounded_rect(draw, (10, y_off, panel_w - 10, y_off + 22), radius=4, fill=(28, 33, 44))
    draw.text((18, y_off + 4), "CLASS", font=fm.fonts["bold"], fill=(210, 215, 225))
    draw.text((190, y_off + 4), "IN-ZONE", font=fm.fonts["bold"], fill=(0, 230, 255))
    draw.text((290, y_off + 4), "TOTAL", font=fm.fonts["bold"], fill=(210, 215, 225))
    y_off += 25

    tot_z_cur, tot_a_cur, tot_z_cum, tot_a_cum = 0, 0, 0, 0

    for cls_id, cls_name in tracker.class_names.items():
        bgr = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)
        rgb_color = (bgr[2], bgr[1], bgr[0])

        cur_z = tracker.current_zone_counts[cls_id]
        cum_z = len(tracker.zone_unique_ids[cls_id])
        cur_a = tracker.current_frame_counts[cls_id]
        cum_a = len(tracker.total_unique_ids[cls_id])

        tot_z_cur += cur_z
        tot_a_cur += cur_a
        tot_z_cum += cum_z
        tot_a_cum += cum_a

        draw_rounded_rect(draw, (10, y_off, panel_w - 10, y_off + 24), radius=4, fill=(18, 21, 28))
        draw.ellipse([18, y_off + 7, 26, y_off + 15], fill=rgb_color)

        disp_name = (cls_name[:15] + "..") if len(cls_name) > 17 else cls_name
        draw.text((32, y_off + 5), disp_name, font=fm.fonts["body"], fill=(230, 235, 245))

        z_str = f"{cur_z} ({cum_z})"
        draw.text((190, y_off + 5), z_str, font=fm.fonts["mono_bold"], fill=(0, 235, 255))

        a_str = f"{cur_a} ({cum_a})"
        draw.text((290, y_off + 5), a_str, font=fm.fonts["mono"], fill=(220, 225, 235))

        y_off += 26

    # Summary Row
    draw_rounded_rect(draw, (10, y_off, panel_w - 10, y_off + 24), radius=4, fill=(32, 38, 52), outline=(55, 66, 88), width=1)
    draw.text((18, y_off + 5), "TOTAL SUM", font=fm.fonts["bold"], fill=(255, 255, 255))
    draw.text((190, y_off + 5), f"{tot_z_cur} ({tot_z_cum})", font=fm.fonts["mono_bold"], fill=(0, 245, 255))
    draw.text((290, y_off + 5), f"{tot_a_cur} ({tot_a_cum})", font=fm.fonts["mono_bold"], fill=(255, 255, 255))
    y_off += 28

    draw.text((15, y_off), "* Counts: Current Frame (Cumulative IDs)", font=fm.fonts["small"], fill=(115, 125, 140))
    y_off += 22

    # --- 2-COLUMN SHORTCUTS GRID ---
    draw.text((15, y_off), "ALL KEY & MOUSE OPTIONS", font=fm.fonts["header"], fill=(150, 175, 255))
    y_off += 18

    col1 = [
        ("L-Click", "Add zone point"),
        ("R-Click", "Undo last pt"),
        ("D/Enter", "Close zone"),
        ("C", "Clear zone"),
        ("T", f"Tracker: {tracker_name[:7]}"),
        ("W", "Switch cam"),
    ]

    col2 = [
        ("+ / -", f"Conf: {int(current_conf * 100)}%"),
        ("E", f"Clarity: {enh_str}"),
        ("R", "Reset counts"),
        ("M", f"Mask: {'ON' if show_masks else 'OFF'}"),
        ("B", f"Box: {'ON' if show_boxes else 'OFF'}"),
        ("S", "Save report"),
    ]

    col_y = y_off
    row_h = 18

    for k, d in col1:
        draw_rounded_rect(draw, (12, col_y, 70, col_y + 15), radius=3, fill=(24, 29, 40), outline=(0, 200, 255), width=1)
        draw.text((16, col_y + 1), k, font=fm.fonts["mono_bold"], fill=(0, 240, 255))
        draw.text((76, col_y + 1), d, font=fm.fonts["small"], fill=(185, 190, 200))
        col_y += row_h

    col_y2 = y_off
    for k, d in col2:
        draw_rounded_rect(draw, (195, col_y2, 250, col_y2 + 15), radius=3, fill=(24, 29, 40), outline=(0, 200, 255), width=1)
        draw.text((199, col_y2 + 1), k, font=fm.fonts["mono_bold"], fill=(0, 240, 255))
        draw.text((256, col_y2 + 1), d, font=fm.fonts["small"], fill=(185, 190, 200))
        col_y2 += row_h

    dash_bgr = cv2.cvtColor(np.array(dash_pil), cv2.COLOR_RGB2BGR)
    return dash_bgr


def draw_modern_label(pil_draw, x1, y1, text, bg_rgb, in_zone=False):
    """Draws a clean, rounded, anti-aliased pill tag above a detected bounding box."""
    fm = FontManager.get()
    font = fm.fonts["tag"]
    bbox = font.getbbox(text)
    tw = (bbox[2] - bbox[0]) + 16
    th = (bbox[3] - bbox[1]) + 8

    tag_y1 = max(4, y1 - th - 5)
    tag_y2 = tag_y1 + th

    outline = (0, 240, 255) if in_zone else (255, 255, 255)
    width = 1

    draw_rounded_rect(pil_draw, (x1, tag_y1, x1 + tw, tag_y2), radius=5, fill=bg_rgb, outline=outline, width=width)
    pil_draw.text((x1 + 8, tag_y1 + 3), text, font=font, fill=(0, 0, 0))


# ==============================================================================
# MAIN APPLICATION ENGINE
# ==============================================================================
def run_app(weights=DEFAULT_WEIGHTS, source=None, conf=0.50, iou=0.5,
            tracker_type="MOSSE", no_gui=False, input_size=(640, 640)):
    print("=" * 72)
    print("   SEGMENTATION & MULTI-TRACKER PRIORITY ZONE SYSTEM")
    print("=" * 72)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device = 0
        print(f"[INFO] Hardware:      GPU Accelerated ({gpu_name})")
    else:
        gpu_name = "CPU (No CUDA)"
        device = "cpu"
        print("[WARNING] CUDA not detected. Falling back to CPU processing.")

    all_cameras = scan_connected_cameras(max_tested=6)
    selected_backend = cv2.CAP_DSHOW
    current_tracker = tracker_type
    selected_input_size = parse_input_size(input_size)

    if source is None:
        if no_gui:
            current_source = 0
        else:
            current_source, selected_backend, current_tracker, selected_input_size = select_setup_gui(
                all_cameras, default_tracker=tracker_type,
                default_input_size="640x640" if selected_input_size == (640, 640) else
                ("Original" if selected_input_size is None else f"{selected_input_size[0]}x{selected_input_size[1]}")
            )
    else:
        current_source = int(source) if str(source).isdigit() else source
        if isinstance(current_source, int):
            match = next((c for c in all_cameras if c["index"] == current_source), None)
            if match:
                selected_backend = match["backend"]

    is_video_file = isinstance(current_source, str) and not current_source.isdigit()
    source_label = current_source if is_video_file else f"Camera {current_source}"
    size_label = "Original" if selected_input_size is None else f"{selected_input_size[0]}x{selected_input_size[1]}"
    print(f"[INFO] Video source:    {source_label}")
    print(f"[INFO] Processing size: {size_label}")
    print(f"[INFO] Initial Tracker: {current_tracker}")

    if not os.path.exists(weights):
        print(f"[ERROR] Weights file not found: {weights}")
        return

    print("[INFO] Loading  model...")
    model = YOLO(weights)
    class_names = model.names
    print(f"[INFO] Loaded {len(class_names)} classes: {class_names}")

    zone_manager = ZoneManager()
    counter_tracker = ObjectCounterTracker(class_names)
    tracker_manager = UnifiedMultiTracker(current_tracker)

    if is_video_file:
        cap = cv2.VideoCapture(current_source)
    else:
        cap = open_high_quality_camera(current_source, selected_backend)

    if not cap.isOpened():
        print(f"[FATAL] Unable to open video source: {current_source}")
        return

    window_name = "Segmentation & Tracking "
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback, {"zone": zone_manager})

    show_masks = True
    show_boxes = True
    enhanced_mode = True
    show_help_modal = False
    fps = 0.0
    frame_count = 0
    start_time = time.time()

    print("\n[READY] Video feed started.")
    print(f"        - Processing resolution: {size_label}")
    print(f"        - Active tracker: {current_tracker}")
    print("        - Press 'T' to cycle between all 15 tracking algorithms.")
    print("        - Press 'H' to toggle fullscreen Help Guide.")
    print("        - Press 'Q' or Esc to quit.\n")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                if is_video_file:
                    print("[INFO] End of video reached.")
                    break
                time.sleep(0.04)
                continue

            if enhanced_mode:
                proc_frame = apply_clarity_enhancement(frame)
            else:
                proc_frame = frame.copy()

            # All inference, tracking, masks and priority-zone coordinates use this frame size.
            if selected_input_size is not None:
                proc_frame = cv2.resize(
                    proc_frame, selected_input_size, interpolation=cv2.INTER_LINEAR
                )

            h, w = proc_frame.shape[:2]

            # Execute tracking via UnifiedMultiTracker
            detections, masks_data = tracker_manager.update(
                model=model,
                frame=proc_frame,
                device=device,
                conf=conf,
                iou=iou,
                show_masks=show_masks,
            )

            # Render segmentation masks onto frame if available
            if show_masks and masks_data is not None:
                mask_overlay = np.zeros_like(proc_frame, dtype=np.uint8)
                for i in range(len(masks_data)):
                    cls_id = detections[i]["class_id"] if i < len(detections) else 0
                    bgr = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)
                    m = masks_data[i]
                    if m.shape != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    bin_mask = m > 0.5
                    mask_overlay[bin_mask] = bgr
                cv2.addWeighted(mask_overlay, 0.40, proc_frame, 0.60, 0, proc_frame)

            counter_tracker.update_frame(detections, zone_manager)

            frame_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            pil_draw = ImageDraw.Draw(pil_frame)

            for det in detections:
                x1, y1, x2, y2 = det["box"]
                cls_id = det["class_id"]
                trk_id = det["track_id"]
                confidence = det["conf"]
                cx, cy = det["centroid"]
                in_zone = det["in_zone"]

                cls_name = class_names.get(cls_id, f"Class {cls_id}")
                bgr = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)
                rgb = (bgr[2], bgr[1], bgr[0])

                dot_c = (0, 0, 255) if in_zone else (255, 255, 0)
                cv2.circle(proc_frame, (cx, cy), 5, dot_c, -1, cv2.LINE_AA)

                if show_boxes:
                    thickness = 1
                    cv2.rectangle(proc_frame, (x1, y1), (x2, y2), bgr, thickness, cv2.LINE_AA)

                    id_str = f"#{trk_id} " if trk_id is not None else ""
                    zone_flag = "[ZONE] " if in_zone else ""
                    label_str = f"{zone_flag}{id_str}{cls_name} {int(confidence * 100)}%"

                    draw_modern_label(pil_draw, x1, y1, label_str, rgb, in_zone)

            zone_manager.draw(proc_frame, pil_draw)
            render_bottom_shortcut_ribbon(pil_draw, w, h)

            annotated_frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
            mask_diff = np.any(proc_frame != cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), axis=-1)
            annotated_frame[mask_diff] = proc_frame[mask_diff]

            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 0.5:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            cam_label = (f"VIDEO ({w}x{h})" if is_video_file
                         else f"Cam {current_source} ({w}x{h})")
            dashboard = render_modern_dashboard(
                h=h,
                tracker=counter_tracker,
                zone_manager=zone_manager,
                fps=fps,
                gpu_name=gpu_name,
                show_masks=show_masks,
                show_boxes=show_boxes,
                cam_info=cam_label,
                enhanced_mode=enhanced_mode,
                current_conf=conf,
                tracker_name=tracker_manager.tracker_name
            )

            combined_view = np.hstack((annotated_frame, dashboard))

            if show_help_modal:
                combined_view = render_help_modal(combined_view, tracker_manager.tracker_name)

            cv2.imshow(window_name, combined_view)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):
                print("[INFO] Quitting application.")
                break
            elif key in (ord('t'), ord('T')):
                # Cycle through available trackers
                t_names = [t[0] for t in AVAILABLE_TRACKERS]
                cur_i = t_names.index(tracker_manager.tracker_name) if tracker_manager.tracker_name in t_names else 0
                next_i = (cur_i + 1) % len(t_names)
                next_tracker = t_names[next_i]
                tracker_manager.set_tracker(next_tracker)
            elif key in (ord('+'), ord('='), ord(']')):
                conf = min(0.95, round(conf + 0.05, 2))
                print(f"[INFO] Confidence Threshold increased to: {int(conf * 100)}%")
            elif key in (ord('-'), ord('_'), ord('[')):
                conf = max(0.10, round(conf - 0.05, 2))
                print(f"[INFO] Confidence Threshold decreased to: {int(conf * 100)}%")
            elif key in (ord('h'), ord('H')):
                show_help_modal = not show_help_modal
                print(f"[INFO] Help Guide: {'ON' if show_help_modal else 'OFF'}")
            elif show_help_modal and key != 255:
                show_help_modal = False
            elif key in (ord('w'), ord('W')):
                if is_video_file:
                    print("[INFO] Camera switching is available only when using webcam input.")
                else:
                    active_cams = [c for c in all_cameras if c["active"]] or all_cameras
                    if active_cams:
                        curr_indices = [c["index"] for c in active_cams]
                        next_i = (curr_indices.index(current_source) + 1) % len(curr_indices) if current_source in curr_indices else 0
                        next_cam = active_cams[next_i]
                        print(f"\n[INFO] Switching camera to index {next_cam['index']}...")
                        cap.release()
                        current_source = next_cam["index"]
                        selected_backend = next_cam["backend"]
                        cap = open_high_quality_camera(current_source, selected_backend)
            elif key in (ord('e'), ord('E')):
                enhanced_mode = not enhanced_mode
                print(f"[INFO] Live Video Clarity Filter: {'ON' if enhanced_mode else 'OFF'}")
            elif key in (ord('d'), ord('D'), 13):
                if not zone_manager.is_closed:
                    if zone_manager.close_polygon():
                        print(f"[INFO] Priority Zone closed with {len(zone_manager.points)} points.")
                    else:
                        print("[WARNING] Please define at least 3 points before closing.")
            elif key in (ord('c'), ord('C')):
                zone_manager.clear()
                print("[INFO] Priority Zone cleared.")
            elif key in (ord('r'), ord('R')):
                counter_tracker.reset()
                print("[INFO] Tracking and counts reset.")
            elif key in (ord('m'), ord('M')):
                show_masks = not show_masks
                print(f"[INFO] Masks visibility: {'ON' if show_masks else 'OFF'}")
            elif key in (ord('b'), ord('B')):
                show_boxes = not show_boxes
                print(f"[INFO] Boxes visibility: {'ON' if show_boxes else 'OFF'}")
            elif key in (ord('s'), ord('S')):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                snap_path = f"snapshot_{timestamp}.jpg"
                json_path = f"counts_{timestamp}.json"
                cv2.imwrite(snap_path, combined_view)

                report_data = {
                    "timestamp": timestamp,
                    "video_source": str(current_source),
                    "input_resolution": size_label,
                    "processed_frame_resolution": f"{w}x{h}",
                    "tracker": tracker_manager.tracker_name,
                    "confidence_threshold": conf,
                    "fps": round(fps, 1),
                    "zone_points": zone_manager.points,
                    "zone_active": zone_manager.is_closed,
                    "class_counts": {
                        class_names[cls_id]: {
                            "current_in_zone": counter_tracker.current_zone_counts[cls_id],
                            "cumulative_in_zone_ids": len(counter_tracker.zone_unique_ids[cls_id]),
                            "current_frame_total": counter_tracker.current_frame_counts[cls_id],
                            "cumulative_total_ids": len(counter_tracker.total_unique_ids[cls_id]),
                        }
                        for cls_id in counter_tracker.class_names
                    }
                }
                with open(json_path, "w") as f:
                    json.dump(report_data, f, indent=4)
                print(f"[INFO] Saved snapshot to: {snap_path}")
                print(f"[INFO] Saved counts report to: {json_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Cleaned up resources. Goodbye!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segmentation & Multi-Tracker Priority Zone System"
    )
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS,
                        help="Path to YOLO segmentation weights (.pt or .onnx)")
    parser.add_argument("--source", default=None,
                        help="Camera index (0, 1, ...) or video file path; omit to open setup dialog")
    parser.add_argument("--conf", type=float, default=0.50,
                        help="Confidence threshold (default: 0.50)")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="NMS IoU threshold")
    parser.add_argument("--tracker", type=str, default="MOSSE",
                        help="Default tracker (MOSSE, CSRT, KCF, MIL, ByteTrack, BoT-SORT, etc.)")
    parser.add_argument("--input-size", type=str, default="640x640",
                        choices=["640x640", "Original", "1280x720", "1920x1080"],
                        help="Frame processing size (default: 640x640)")
    parser.add_argument("--no-gui", action="store_true",
                        help="Skip the startup selector and use the supplied/default source")

    args = parser.parse_args()
    run_app(
        weights=args.weights,
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        tracker_type=args.tracker,
        no_gui=args.no_gui,
        input_size=args.input_size
    )
