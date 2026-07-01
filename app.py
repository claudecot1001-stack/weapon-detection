"""
app.py
Flask + Socket.IO app untuk deteksi objek YOLO.

Fitur:
  - Webcam via browser  → getUserMedia → Socket.IO → YOLO → hasil balik ke browser
  - CCTV / IP Camera    → RTSP/HTTP → YOLO → Socket.IO broadcast
  - Upload gambar/video → YOLO → hasil bisa didownload

Install:
    pip install flask flask-socketio ultralytics opencv-python eventlet

Jalankan:
    python app.py
"""

import os
import cv2
import time
import uuid
import base64
import shutil
import subprocess
import threading
import numpy as np
from datetime import datetime
from flask import Flask, Response, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# ====================== KONFIGURASI ======================
MODEL_PATH      = "Model/best5.pt"
CONF_THRESHOLD  = 0.5
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480

UPLOAD_FOLDER     = "static/uploads"
RESULT_FOLDER     = "static/results"
SCREENSHOT_FOLDER = "static/screenshots"
ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "bmp", "webp"}
ALLOWED_VIDEO_EXT = {"mp4", "avi", "mov", "mkv"}
MAX_CONTENT_LENGTH      = 200 * 1024 * 1024  # 200 MB
SCREENSHOT_INTERVAL_SEC = 3
CCTV_CONNECT_TIMEOUT    = 10
# =========================================================

app = Flask(__name__)
app.config["SECRET_KEY"]          = os.environ.get("SECRET_KEY", "yolo-secret-key")
app.config["MAX_CONTENT_LENGTH"]  = MAX_CONTENT_LENGTH

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    max_http_buffer_size=10 * 1024 * 1024,  # 10 MB per frame
    ping_timeout=60,
    ping_interval=25,
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)

print(f"[INFO] Loading model dari {MODEL_PATH} ...")
model = YOLO(MODEL_PATH)
print("[INFO] Model berhasil di-load.")

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None

# ===================== GLOBAL STATE =====================
cctv_url     = None
cctv_camera  = None
cctv_lock    = threading.Lock()
cctv_thread  = None
cctv_running = False

latest_detections = []
screenshot_history = []
screenshot_lock    = threading.Lock()
# ========================================================


# ─────────────────────── helpers ────────────────────────

def save_screenshot(frame, label, confidence, source_name):
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{label.replace(' ', '_')}_{ts}.jpg"
    cv2.imwrite(os.path.join(SCREENSHOT_FOLDER, filename), frame)
    entry = {
        "filename":   filename,
        "url":        f"/static/screenshots/{filename}",
        "label":      label,
        "confidence": confidence,
        "source":     source_name,
        "time":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with screenshot_lock:
        screenshot_history.insert(0, entry)
        if len(screenshot_history) > 200:
            screenshot_history.pop()
    return entry


def allowed_file(filename, allowed_ext):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_ext


def run_yolo(frame):
    """Jalankan YOLO pada satu frame. Return (annotated_frame, detections)."""
    results   = model(frame, conf=CONF_THRESHOLD, verbose=False)
    annotated = results[0].plot()
    dets = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label  = model.names[cls_id]
        conf   = round(float(box.conf[0]), 2)
        dets.append({"label": label, "confidence": conf})
    return annotated, dets


def frame_to_b64(frame, quality=75):
    """Encode OpenCV frame → base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


# ──────────────────── Socket.IO — webcam ────────────────────

@socketio.on("connect")
def on_connect():
    print(f"[WS] Client terhubung: {request.sid}")


@socketio.on("disconnect")
def on_disconnect():
    print(f"[WS] Client terputus: {request.sid}")


@socketio.on("webcam_frame")
def on_webcam_frame(data):
    """
    Terima frame dari browser:
      { "image": "data:image/jpeg;base64,..." }
    Kirim balik hasil deteksi ke client yang sama:
      { "image": "data:image/jpeg;base64,...", "detections": [...] }
    """
    global latest_detections
    try:
        img_data = data.get("image", "")
        if img_data.startswith("data:image"):
            img_data = img_data.split(",", 1)[1]

        img_bytes = base64.b64decode(img_data)
        np_arr    = np.frombuffer(img_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            emit("error", {"message": "Frame tidak valid."})
            return

        annotated, detections = run_yolo(frame)
        latest_detections = detections

        emit("detection_result", {
            "image":      frame_to_b64(annotated),
            "detections": detections,
        })

    except Exception as e:
        print(f"[ERROR] webcam_frame: {e}")
        emit("error", {"message": str(e)})


# ──────────────────── CCTV background thread ────────────────────

def cctv_stream_loop(cap, source_label):
    """
    Baca frame dari CCTV, proses YOLO, broadcast ke semua
    client yang sedang di tab CCTV via Socket.IO event 'cctv_frame'.
    """
    global latest_detections, cctv_running

    retry    = 0
    max_retry = 10
    last_ss  = {}

    while cctv_running:
        with cctv_lock:
            if cctv_camera is None or not cctv_camera.isOpened():
                break

        ok, frame = cap.read()
        if not ok or frame is None:
            retry += 1
            if retry >= max_retry:
                print("[ERROR] CCTV koneksi terputus permanen.")
                socketio.emit("cctv_status", {"status": "disconnected"})
                break
            time.sleep(0.5)
            continue

        retry = 0
        annotated, detections = run_yolo(frame)
        latest_detections = detections

        # Screenshot periodik per label
        now = time.time()
        for d in detections:
            lbl = d["label"]
            if now - last_ss.get(lbl, 0) >= SCREENSHOT_INTERVAL_SEC:
                save_screenshot(annotated, lbl, d["confidence"], source_label)
                last_ss[lbl] = now

        socketio.emit("cctv_frame", {
            "image":      frame_to_b64(annotated, quality=70),
            "detections": detections,
        })

        time.sleep(0.067)   # ~15 fps cap

    cctv_running = False
    print("[INFO] CCTV stream loop berhenti.")


# ──────────────────── Image / video processing ────────────────────

def process_image(filepath, filename):
    results   = model(filepath, conf=CONF_THRESHOLD, verbose=False)
    annotated = results[0].plot()

    result_filename = f"result_{filename}"
    cv2.imwrite(os.path.join(RESULT_FOLDER, result_filename), annotated)

    detections, seen = [], set()
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label  = model.names[cls_id]
        conf   = round(float(box.conf[0]), 2)
        detections.append({"label": label, "confidence": conf})
        if label not in seen:
            seen.add(label)
            save_screenshot(annotated, label, conf, filename)

    return result_filename, detections


def reencode_with_ffmpeg(input_path):
    if not FFMPEG_AVAILABLE:
        return input_path
    base, _     = os.path.splitext(input_path)
    output_path = f"{base}_h264.mp4"
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path,
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-preset", "fast", "-movflags", "+faststart", output_path],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0 or not os.path.exists(output_path):
            return input_path
        os.remove(input_path)
        return output_path
    except Exception as e:
        print(f"[WARNING] ffmpeg error: {e}")
        return input_path


def process_video(filepath, filename):
    cap    = cv2.VideoCapture(filepath)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    result_filename = f"result_{os.path.splitext(filename)[0]}.mp4"
    result_path     = os.path.join(RESULT_FOLDER, result_filename)
    writer          = None

    for fourcc_code, suffix in [("mp4v", ".mp4"), ("avc1", ".mp4"), ("XVID", ".avi")]:
        if suffix != ".mp4":
            result_filename = f"result_{os.path.splitext(filename)[0]}{suffix}"
            result_path     = os.path.join(RESULT_FOLDER, result_filename)
        writer = cv2.VideoWriter(
            result_path, cv2.VideoWriter_fourcc(*fourcc_code), fps, (width, height)
        )
        if writer.isOpened():
            break

    if not writer or not writer.isOpened():
        cap.release()
        raise RuntimeError("Tidak ada codec video yang berhasil dibuka.")

    detection_counts = {}
    last_ss_time     = {}
    frame_idx        = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cur_sec   = frame_idx / fps
        annotated, frame_dets = run_yolo(frame)

        if annotated.shape[1] != width or annotated.shape[0] != height:
            annotated = cv2.resize(annotated, (width, height))
        writer.write(annotated)

        for d in frame_dets:
            lbl  = d["label"]
            conf = d["confidence"]
            detection_counts[lbl] = detection_counts.get(lbl, 0) + 1
            if cur_sec - last_ss_time.get(lbl, -999) >= SCREENSHOT_INTERVAL_SEC:
                save_screenshot(annotated, lbl, conf, filename)
                last_ss_time[lbl] = cur_sec

        frame_idx += 1

    cap.release()
    writer.release()

    final_path      = reencode_with_ffmpeg(result_path)
    result_filename = os.path.basename(final_path)
    print(f"[INFO] Video '{filename}' selesai. Frame: {frame_idx}.")
    return result_filename, [{"label": k, "count": v} for k, v in detection_counts.items()]


# ──────────────────── HTTP Routes ────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detections")
def detections():
    return jsonify(latest_detections)


@app.route("/connect_cctv", methods=["POST"])
def connect_cctv():
    global cctv_camera, cctv_url, cctv_thread, cctv_running

    data = request.get_json(silent=True)
    if not data or not data.get("url"):
        return jsonify({"error": "URL CCTV tidak diberikan."}), 400

    url = data["url"].strip()

    # Hentikan thread lama
    cctv_running = False
    time.sleep(0.3)
    with cctv_lock:
        if cctv_camera is not None:
            cctv_camera.release()
            cctv_camera = None
            cctv_url    = None

    print(f"[INFO] Mencoba membuka CCTV: {url}")
    cap      = cv2.VideoCapture(url)
    deadline = time.time() + CCTV_CONNECT_TIMEOUT
    opened   = False

    while time.time() < deadline:
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                opened = True
                break
        time.sleep(0.5)

    if not opened:
        cap.release()
        return jsonify({
            "error": (
                f"Tidak dapat membuka stream dari '{url}'. "
                "Periksa URL, kredensial, dan pastikan kamera dapat "
                "diakses dari jaringan server."
            )
        }), 503

    with cctv_lock:
        cctv_camera = cap
        cctv_url    = url

    cctv_running = True
    cctv_thread  = threading.Thread(
        target=cctv_stream_loop, args=(cap, url), daemon=True
    )
    cctv_thread.start()

    print(f"[INFO] CCTV berhasil terhubung: {url}")
    return jsonify({"status": "connected", "url": url})


@app.route("/disconnect_cctv", methods=["POST"])
def disconnect_cctv():
    global cctv_camera, cctv_url, cctv_running
    cctv_running = False
    with cctv_lock:
        if cctv_camera is not None:
            cctv_camera.release()
            cctv_camera = None
            cctv_url    = None
    return jsonify({"status": "disconnected"})


@app.route("/upload", methods=["GET"])
def upload_page():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang dikirim"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong"}), 400

    original_name = secure_filename(file.filename)
    unique_name   = f"{uuid.uuid4().hex[:8]}_{original_name}"
    upload_path   = os.path.join(UPLOAD_FOLDER, unique_name)

    if allowed_file(original_name, ALLOWED_IMAGE_EXT):
        file.save(upload_path)
        result_filename, dets = process_image(upload_path, unique_name)
        new_ss = [s for s in screenshot_history if s["source"] == unique_name]
        return jsonify({"type": "image",
                        "result_url": f"/static/results/{result_filename}",
                        "detections": dets, "screenshots": new_ss})

    elif allowed_file(original_name, ALLOWED_VIDEO_EXT):
        file.save(upload_path)
        result_filename, dets = process_video(upload_path, unique_name)
        new_ss = [s for s in screenshot_history if s["source"] == unique_name]
        return jsonify({"type": "video",
                        "result_url": f"/static/results/{result_filename}",
                        "detections": dets, "screenshots": new_ss})

    return jsonify({"error": "Format file tidak didukung"}), 400


@app.route("/screenshots")
def get_screenshots():
    with screenshot_lock:
        return jsonify(screenshot_history)


# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[INFO] Server berjalan di port {port}")
    socketio.run(
        app,
        host="0.0.0.0",
        port=port,
        debug=False,
        allow_unsafe_werkzeug=True,
    )