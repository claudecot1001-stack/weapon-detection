"""
app.py
Flask app untuk deteksi objek real-time menggunakan model YOLO (model/best.pt)
dan kamera yang aktif di komputer/server, di-stream sebagai MJPEG ke halaman web.

Cara jalankan:
    pip install flask ultralytics opencv-python
    python app.py

Lalu buka di browser: http://localhost:5000
"""

import os
import cv2
import time
import uuid
import shutil
import subprocess
import threading
from datetime import datetime
from flask import Flask, Response, render_template, jsonify, request
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# ====================== KONFIGURASI ======================
MODEL_PATH = "Model/best.pt"   # path ke model YOLO kamu
CAMERA_INDEX = 0               # 0 = kamera default. Ganti 1, 2, dst kalau ada banyak kamera
CONF_THRESHOLD = 0.5           # ambang batas confidence deteksi
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

UPLOAD_FOLDER = "static/uploads"        # file asli yang diupload
RESULT_FOLDER = "static/results"        # hasil setelah dideteksi YOLO
SCREENSHOT_FOLDER = "static/screenshots"  # screenshot otomatis tiap deteksi penting
ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "bmp", "webp"}
ALLOWED_VIDEO_EXT = {"mp4", "avi", "mov", "mkv"}
MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # batas upload 200 MB
SCREENSHOT_INTERVAL_SEC = 3   # jarak minimum antar screenshot untuk label yang sama (video)

# Timeout (detik) saat mencoba membuka URL CCTV sebelum dianggap gagal
CCTV_CONNECT_TIMEOUT = 10
# ===========================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)

# Load model YOLO sekali saat startup (biar tidak reload tiap frame)
print(f"[INFO] Loading model dari {MODEL_PATH} ...")
model = YOLO(MODEL_PATH)
print("[INFO] Model berhasil di-load.")

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
if FFMPEG_AVAILABLE:
    print("[INFO] ffmpeg ditemukan -> video hasil akan di-re-encode ke H.264.")
else:
    print("[WARNING] ffmpeg TIDAK ditemukan. Video hasil mungkin tidak bisa "
          "di-preview langsung di browser.")

# ===================== GLOBAL STATE =====================
camera = None
camera_lock = threading.Lock()

# CCTV state — URL aktif dan objek VideoCapture-nya (terpisah dari kamera lokal)
cctv_url = None
cctv_camera = None
cctv_lock = threading.Lock()

latest_detections = []
screenshot_history = []
screenshot_lock = threading.Lock()
# ========================================================


# -------------------- helpers --------------------

def save_screenshot(frame, label, confidence, source_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    safe_label = label.replace(" ", "_")
    filename = f"{safe_label}_{timestamp}.jpg"
    filepath = os.path.join(SCREENSHOT_FOLDER, filename)
    cv2.imwrite(filepath, frame)

    entry = {
        "filename": filename,
        "url": f"/static/screenshots/{filename}",
        "label": label,
        "confidence": confidence,
        "source": source_name,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with screenshot_lock:
        screenshot_history.insert(0, entry)
        if len(screenshot_history) > 200:
            screenshot_history.pop()

    return entry


def init_camera():
    """Inisialisasi kamera lokal."""
    global camera
    try:
        cam = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    except Exception:
        cam = cv2.VideoCapture(CAMERA_INDEX)

    if not cam.isOpened():
        cam = cv2.VideoCapture(CAMERA_INDEX)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera = cam
    return cam


def allowed_file(filename, allowed_ext):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_ext


# -------------------- generators --------------------

def generate_frames():
    """Generator MJPEG dari kamera lokal."""
    global latest_detections

    with camera_lock:
        if camera is None or not camera.isOpened():
            init_camera()

    while True:
        with camera_lock:
            if camera is None or not camera.isOpened():
                import numpy as np
        
                frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        
                cv2.putText(
                    frame,
                    "Camera is Unactive",
                    (120, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
        
                _, buffer = cv2.imencode(".jpg", frame)
        
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )
        
                time.sleep(0.1)
                continue
            
            success, frame = camera.read()
    

        if not success or frame is None:
            print("[WARNING] Gagal membaca frame dari kamera, retry...")
            time.sleep(0.5)
            continue

        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        annotated_frame = results[0].plot()

        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            detections.append({"label": label, "confidence": round(conf, 2)})
        latest_detections = detections

        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        if not ret:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


def generate_cctv_frames():
    """Generator MJPEG dari sumber CCTV / IP Camera.

    Membaca frame dari `cctv_camera`, menjalankan inferensi YOLO, lalu
    men-stream hasilnya. Kalau koneksi putus, generator berhenti (klien
    akan mendapat respons kosong / stream terputus).
    """
    global latest_detections

    with cctv_lock:
        cap = cctv_camera

    if cap is None or not cap.isOpened():
        print("[WARNING] CCTV camera belum terhubung, generator berhenti.")
        return

    retry = 0
    max_retry = 10

    while True:
        with cctv_lock:
            cap = cctv_camera
            if cap is None or not cap.isOpened():
                print("[INFO] CCTV camera di-release, stream berhenti.")
                break

        success, frame = cap.read()

        if not success or frame is None:
            retry += 1
            print(f"[WARNING] CCTV frame gagal dibaca (retry {retry}/{max_retry})...")
            if retry >= max_retry:
                print("[ERROR] CCTV koneksi terputus permanen.")
                break
            time.sleep(0.5)
            continue

        retry = 0  # reset kalau berhasil

        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        annotated_frame = results[0].plot()

        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            detections.append({"label": label, "confidence": round(conf, 2)})
        latest_detections = detections

        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        if not ret:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


# -------------------- image / video processing --------------------

def process_image(filepath, filename):
    results = model(filepath, conf=CONF_THRESHOLD, verbose=False)
    annotated = results[0].plot()

    result_filename = f"result_{filename}"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, annotated)

    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = round(float(box.conf[0]), 2)
        detections.append({"label": label, "confidence": conf})

    if detections:
        seen_labels = set()
        for d in detections:
            if d["label"] in seen_labels:
                continue
            seen_labels.add(d["label"])
            save_screenshot(annotated, d["label"], d["confidence"], filename)

    return result_filename, detections


def reencode_with_ffmpeg(input_path):
    if not FFMPEG_AVAILABLE:
        return input_path

    base, _ = os.path.splitext(input_path)
    output_path = f"{base}_h264.mp4"

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", input_path,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                "-movflags", "+faststart",
                output_path,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0 or not os.path.exists(output_path):
            print(f"[WARNING] ffmpeg re-encode gagal: {result.stderr[-500:]}")
            return input_path

        os.remove(input_path)
        return output_path

    except subprocess.TimeoutExpired:
        print("[WARNING] ffmpeg re-encode timeout, memakai file asli.")
        return input_path
    except Exception as e:
        print(f"[WARNING] ffmpeg re-encode error: {e}")
        return input_path


def process_video(filepath, filename):
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    result_filename = f"result_{os.path.splitext(filename)[0]}.mp4"
    result_path = os.path.join(RESULT_FOLDER, result_filename)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(result_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print("[WARNING] Codec mp4v gagal, fallback ke avc1...")
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(result_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print("[WARNING] Codec avc1 gagal, fallback ke XVID (.avi)...")
        result_filename = f"result_{os.path.splitext(filename)[0]}.avi"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(result_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Tidak ada codec video yang berhasil dibuka.")

    detection_counts = {}
    last_screenshot_time = {}
    frame_idx = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        current_sec = frame_idx / fps
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        annotated = results[0].plot()

        if annotated.shape[1] != width or annotated.shape[0] != height:
            annotated = cv2.resize(annotated, (width, height))

        writer.write(annotated)

        frame_best_conf = {}
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = round(float(box.conf[0]), 2)
            detection_counts[label] = detection_counts.get(label, 0) + 1
            if label not in frame_best_conf or conf > frame_best_conf[label]:
                frame_best_conf[label] = conf

        for label, conf in frame_best_conf.items():
            last_time = last_screenshot_time.get(label)
            if last_time is None or (current_sec - last_time) >= SCREENSHOT_INTERVAL_SEC:
                save_screenshot(annotated, label, conf, filename)
                last_screenshot_time[label] = current_sec

        frame_idx += 1

    cap.release()
    writer.release()

    final_path = reencode_with_ffmpeg(result_path)
    result_filename = os.path.basename(final_path)

    total_detections = sum(detection_counts.values())
    print(f"[INFO] Video '{filename}' selesai. Frame: {frame_idx}, deteksi: {total_detections}.")

    detections = [{"label": k, "count": v} for k, v in detection_counts.items()]
    return result_filename, detections


# ========================= ROUTES =========================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """Stream MJPEG dari kamera lokal."""
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ---- CCTV routes ----

@app.route("/connect_cctv", methods=["POST"])
def connect_cctv():
    """Menerima URL CCTV dari frontend, mencoba membuka koneksi OpenCV,
    dan mengembalikan status berhasil / gagal.

    Payload JSON: { "url": "rtsp://user:pass@192.168.1.100:554/stream1" }
    Response JSON:
      - sukses : { "status": "connected", "url": "<url>" }
      - gagal  : { "error": "<pesan>" }, HTTP 400 / 503
    """
    global cctv_camera, cctv_url

    data = request.get_json(silent=True)
    if not data or not data.get("url"):
        return jsonify({"error": "URL CCTV tidak diberikan."}), 400

    url = data["url"].strip()

    # Tutup koneksi lama kalau ada
    with cctv_lock:
        if cctv_camera is not None:
            cctv_camera.release()
            cctv_camera = None
            cctv_url = None

    print(f"[INFO] Mencoba membuka CCTV: {url}")

    cap = cv2.VideoCapture(url)

    # Beri sedikit waktu agar koneksi RTSP/HTTP terbuka
    deadline = time.time() + CCTV_CONNECT_TIMEOUT
    opened = False
    while time.time() < deadline:
        if cap.isOpened():
            # Coba baca satu frame untuk memastikan stream aktif
            ok, _ = cap.read()
            if ok:
                opened = True
                break
        time.sleep(0.5)

    if not opened:
        cap.release()
        print(f"[WARNING] Gagal terhubung ke CCTV: {url}")
        return jsonify({
            "error": (
                f"Tidak dapat membuka stream dari '{url}'. "
                "Periksa kembali URL, kredensial, port, dan pastikan "
                "kamera dapat diakses dari jaringan server ini."
            )
        }), 503

    with cctv_lock:
        cctv_camera = cap
        cctv_url = url

    print(f"[INFO] CCTV berhasil terhubung: {url}")
    return jsonify({"status": "connected", "url": url})


@app.route("/cctv_feed")
def cctv_feed():
    """Stream MJPEG dari sumber CCTV / IP Camera yang sudah terhubung."""
    with cctv_lock:
        if cctv_camera is None or not cctv_camera.isOpened():
            return Response("CCTV belum terhubung.", status=503, mimetype="text/plain")

    return Response(generate_cctv_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/disconnect_cctv", methods=["POST"])
def disconnect_cctv():
    """Memutus koneksi CCTV dan melepas resource."""
    global cctv_camera, cctv_url
    with cctv_lock:
        if cctv_camera is not None:
            cctv_camera.release()
            cctv_camera = None
            cctv_url = None
    print("[INFO] CCTV diputus.")
    return jsonify({"status": "disconnected"})


# ---- existing routes ----

@app.route("/detections")
def detections():
    return jsonify(latest_detections)


@app.route("/start_camera")
def start_camera():
    with camera_lock:
        if camera is None or not camera.isOpened():
            init_camera()
    return jsonify({"status": "camera started"})


@app.route("/stop_camera")
def stop_camera():
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
    return jsonify({"status": "camera stopped"})


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
    unique_name = f"{uuid.uuid4().hex[:8]}_{original_name}"
    upload_path = os.path.join(UPLOAD_FOLDER, unique_name)

    if allowed_file(original_name, ALLOWED_IMAGE_EXT):
        file.save(upload_path)
        result_filename, dets = process_image(upload_path, unique_name)
        new_screenshots = [s for s in screenshot_history if s["source"] == unique_name]
        return jsonify({
            "type": "image",
            "result_url": f"/static/results/{result_filename}",
            "detections": dets,
            "screenshots": new_screenshots,
        })

    elif allowed_file(original_name, ALLOWED_VIDEO_EXT):
        file.save(upload_path)
        result_filename, dets = process_video(upload_path, unique_name)
        new_screenshots = [s for s in screenshot_history if s["source"] == unique_name]
        return jsonify({
            "type": "video",
            "result_url": f"/static/results/{result_filename}",
            "detections": dets,
            "screenshots": new_screenshots,
        })

    else:
        return jsonify({"error": "Format file tidak didukung"}), 400


@app.route("/screenshots")
def get_screenshots():
    with screenshot_lock:
        return jsonify(screenshot_history)


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
    finally:
        if camera is not None:
            camera.release()
        if cctv_camera is not None:
            cctv_camera.release()