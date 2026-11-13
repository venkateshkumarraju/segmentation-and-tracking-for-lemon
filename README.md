#  Segmentation & Priority Zone Multi-Tracker System

High-performance real-time application combining **YOLO Segmentation**, **Interactive Priority Zones**, and a comprehensive suite of **15 Object Tracking Algorithms** with CUDA GPU acceleration.

---

## 🎯 Supported Tracking Algorithms

You can choose any of the following trackers upon startup or switch dynamically at runtime by pressing **`T`**:

| Tracker | Category | Best Used For |
|---|---|---|
| **ByteTrack** *(Default / Recommended)* | Deep Multi-Object (MOT) | High accuracy multi-object detection tracking under clutter |
| **BoT-SORT** | Deep Multi-Object (MOT) | Re-identification & camera motion compensation |
| **CSRT** | Correlation Filter (OpenCV) | High visual accuracy with channel/spatial reliability |
| **KCF** | Kernelized Correlation (OpenCV) | Fast tracking with kernel trick |
| **MIL** | Online Learning (OpenCV) | Multiple instance learning, robust against partial occlusion |
| **MOSSE** | Fast Correlation (OpenCV) | Ultra-high frame rate lightweight tracking |
| **MEDIANFLOW** | Optical Flow (OpenCV) | Smooth, predictable object trajectories |
| **TLD** | Learning & Detection (OpenCV) | Recovers from track loss using online detection |
| **BOOSTING** | Boosting (OpenCV) | AdaBoost-based visual classifier |
| **CAMShift** | Color Histogram (OpenCV) | Continuously adaptive mean shift color tracking |
| **MeanShift** | Color Histogram (OpenCV) | Convergence to color distribution modes |
| **DaSiamRPN** | Deep Siamese (OpenCV DNN) | Siamese Region Proposal Network |
| **NanoTrack** | Lightweight Neural (OpenCV DNN) | Real-time lightweight deep tracker |
| **VitTrack** | Transformer (OpenCV DNN) | Vision Transformer tracking |
| **GOTURN** | Regression CNN (OpenCV DNN) | Generic offline trained deep tracker |

---

## 🚀 Key Features

- **Dynamic Tracker Switching (`T`)**: Cycle through all tracking algorithms on the fly during live video.
- **Startup Setup Dialog**: Choose both your connected webcam and preferred tracker with a live preview.
- **Optimized 970 × 540 Resolution**: Cuts pixel bandwidth by 50% for maximum frame rates.
- **50% Confidence Threshold**: Filters out low-confidence detections. Tune live with **`+`** / **`-`**.
- **Interactive Priority Polygon Zone**: Left-click to draw points, right-click to undo, `D` or `Enter` to close.
- **Dual Counts per Class**:
  - In-Zone (Current & Cumulative unique IDs).
  - Screen Total (Current & Cumulative unique IDs).

---

## 🎮 Keyboard & Mouse Shortcuts

| Key / Mouse | Action |
|---|---|
| **`T`** | **Cycle between all 15 Tracking Algorithms** |
| **`+` / `-`** | Adjust confidence threshold up or down (50%) |
| **`W`** | Switch / cycle connected webcams |
| **`E`** | Toggle Live Video Clarity Filter (CLAHE + Unsharp Mask) |
| **`H`** | Toggle fullscreen Controls & Tracker Help Modal |
| **Left Click** | Add polygon point for priority zone / close near 1st point |
| **Right Click** | Undo last polygon point |
| **`D`** or **`Enter`** | Close & activate priority zone |
| **`C`** | Clear priority zone |
| **`R`** | Reset tracking IDs and cumulative counters |
| **`M`** | Toggle segmentation mask overlay |
| **`B`** | Toggle bounding box overlay |
| **`S`** | Save snapshot image (`.jpg`) and detailed count report (`.json`) |
| **`Q`** or **`ESC`** | Quit application |
