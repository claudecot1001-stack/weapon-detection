"""
app.py
Flask app untuk deteksi objek real-time menggunakan model YOLO (model/best.pt).

Fitur:
  - Upload gambar / video → deteksi YOLO → hasil bisa didownload
  - CCTV / IP Camera via RTSP atau HTTP → stream MJPEG + deteksi real-time
  - Kamera lokal TIDAK didukung di cloud hosting (Railway, Render, dll.)
    → ditampilkan frame placeholder informatif

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
import numpy as np
from datetime import datetime
from flask import Flask, Response, render_template, jsonify, request
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# ====================== KONFIGURASI ======================
MODEL_PATH = "Model/best.pt"
CONF_THRESHOLD = 0.5
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
SCREENSHOT_FOLDER = "static/screenshots"
ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "bmp", "webp"}
ALLOWED_VIDEO_EXT = {"mp4", "avi", "mov", "mkv"}
MAX_CONTENT_LENGTH = 200 * 1024 * 1024   # 200 MB
SCREENSHOT_INTERVAL_SEC = 3

# Timeout (detik) saat mencoba membuka URL CCTV
CCTV_CONNECT_TIMEOUT = 10

# Deteksi apakah berjalan di environment cloud (tidak ada /dev/video*)
IS_CLOUD = not any(
    os.path.exists(f"/dev/video{i}") for i in range(4)
)
# ===========================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)

print(f"[INFO] Loading model dari {MODEL_PATH} ...")
model = YOLO(MODEL_PATH)
print("[INFO] Model berhasil di-load.")

if IS_CLOUD:
    print("[INFO] Berjalan di environment cloud — kamera lokal dinonaktifkan.")
else:
    print("[INFO] Berjalan di environment lokal — kamera lokal tersedia.")

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
if FFMPEG_AVAILABLE:
    print("[INFO] ffmpeg ditemukan -> video hasil akan di-re-encode ke H.264.")
else:
    print("[WARNING] ffmpeg TIDAK ditemukan.")

# ===================== GLOBAL STATE =====================
camera = None
camera_lock = threading.Lock()
camera_on = False          # False di cloud, bisa di-toggle di lokal

cctv_url = None
cctv_camera = None
cctv_lock = threading.Lock()

latest_detections = []
screenshot_history = []
screenshot_lock = threading.Lock()
# ========================================================


# -------------------- helpers --------------------

def make_placeholder_frame(message: str, sub: str = "") -> bytes:
    """Buat frame hitam dengan teks pesan sebagai placeholder."""
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    # Border tipis
    cv2.rectangle(frame, (10, 10), (FRAME_WIDTH - 10, FRAME_HEIGHT - 10),
                  (40, 40, 40), 1)

    # Icon kamera (lingkaran)
    cx, cy = FRAME_WIDTH // 2, FRAME_HEIGHT // 2 - 40
    cv2.circle(frame, (cx, cy), 36, (60, 60, 60), -1)
    cv2.circle(frame, (cx, cy), 22, (30, 30, 30), -1)
    cv2.circle(frame, (cx, cy), 8, (80, 80, 80), -1)

    # Teks utama — wrap otomatis per kata
    words = message.split()
    lines, current = [], ""
    for word in words:
        test = (current + " " + word).strip()
        if cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0] < FRAME_WIDTH - 60:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    y_start = FRAME_HEIGHT // 2 + 20
    for i, line in enumerate(lines):
        text_w = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
        x = (FRAME_WIDTH - text_w) // 2
        cv2.putText(frame, line, (x, y_start + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

    # Sub-teks (lebih kecil, biru)
    if sub:
        sub_w = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
        x = (FRAME_WIDTH - sub_w) // 2
        cv2.putText(frame, sub, (x, y_start + len(lines) * 28 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 160, 220), 1, cv2.LINE_AA)

    _, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes()


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
    """Inisialisasi kamera lokal (hanya di environment non-cloud)."""
    global camera, camera_on
    if IS_CLOUD:
        return None

    try:
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    except Exception:
        cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        cam = cv2.VideoCapture(0)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera = cam
    camera_on = cam.isOpened()
    return cam


def allowed_file(filename, allowed_ext):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_ext


# -------------------- generators --------------------

def generate_frames():
    """Generator MJPEG dari kamera lokal.

    Di cloud: stream placeholder statis.
    Di lokal: stream live + deteksi YOLO.
    """
    global latest_detections

    # ── Cloud mode ──────────────────────────────────────────────────────────
    if IS_CLOUD:
        frame_bytes = make_placeholder_frame(
            "Kamera lokal tidak tersedia di cloud hosting.",
            "Gunakan fitur CCTV / IP Camera atau Upload File."
        )
        while True:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n"
                   + frame_bytes + b"\r\n")
            time.sleep(2)
        return

    # ── Local mode ───────────────────────────────────────────────────────────
    with camera_lock:
        if camera is None or not camera.isOpened():
            init_camera()

    while True:
        with camera_lock:
            if not camera_on or camera is None or not camera.isOpened():
                frame_bytes = make_placeholder_frame(
                    "Kamera dimatikan.",
                    "Klik 'Nyalakan Kamera' untuk memulai stream."
                )
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n"
                       + frame_bytes + b"\r\n")
                time.sleep(0.5)
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
               b"Content-Type: image/jpeg\r\n\r\n"
               + buffer.tobytes() + b"\r\n")


def generate_cctv_frames():
    """Generator MJPEG dari sumber CCTV / IP Camera."""
    global latest_detections

    with cctv_lock:
        cap = cctv_camera

    if cap is None or not cap.isOpened():
        print("[WARNING] CCTV camera belum terhubung.")
        frame_bytes = make_placeholder_frame(
            "CCTV belum terhubung.",
            "Klik 'Hubungkan CCTV' dan masukkan URL stream."
        )
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n"
               + frame_bytes + b"\r\n")
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
                frame_bytes = make_placeholder_frame(
                    "Koneksi CCTV terputus.",
                    "Coba hubungkan ulang stream."
                )
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n"
                       + frame_bytes + b"\r\n")
                break
            time.sleep(0.5)
            continue

        retry = 0

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
               b"Content-Type: image/jpeg\r\n\r\n"
               + buffer.tobytes() + b"\r\n")


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

    total = sum(detection_counts.values())
    print(f"[INFO] Video '{filename}' selesai. Frame: {frame_idx}, deteksi: {total}.")

    detections = [{"label": k, "count": v} for k, v in detection_counts.items()]
    return result_filename, detections


# ========================= ROUTES =========================

@app.route("/")
def index():
    return render_template("index.html", is_cloud=IS_CLOUD)


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ---- CCTV routes ----

@app.route("/connect_cctv", methods=["POST"])
def connect_cctv():
    global cctv_camera, cctv_url

    data = request.get_json(silent=True)
    if not data or not data.get("url"):
        return jsonify({"error": "URL CCTV tidak diberikan."}), 400

    url = data["url"].strip()

    with cctv_lock:
        if cctv_camera is not None:
            cctv_camera.release()
            cctv_camera = None
            cctv_url = None

    print(f"[INFO] Mencoba membuka CCTV: {url}")

    cap = cv2.VideoCapture(url)

    deadline = time.time() + CCTV_CONNECT_TIMEOUT
    opened = False
    while time.time() < deadline:
        if cap.isOpened():
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
    with cctv_lock:
        if cctv_camera is None or not cctv_camera.isOpened():
            # Kembalikan placeholder stream alih-alih error 503
            def placeholder():
                fb = make_placeholder_frame(
                    "CCTV belum terhubung.",
                    "Klik 'Hubungkan CCTV' dan masukkan URL stream."
                )
                while True:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + fb + b"\r\n")
                    time.sleep(2)
            return Response(placeholder(),
                            mimetype="multipart/x-mixed-replace; boundary=frame")

    return Response(generate_cctv_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/disconnect_cctv", methods=["POST"])
def disconnect_cctv():
    global cctv_camera, cctv_url
    with cctv_lock:
        if cctv_camera is not None:
            cctv_camera.release()
            cctv_camera = None
            cctv_url = None
    print("[INFO] CCTV diputus.")
    return jsonify({"status": "disconnected"})


# ---- Camera routes (lokal saja, no-op di cloud) ----

@app.route("/start_camera")
def start_camera():
    global camera, camera_on
    if IS_CLOUD:
        return jsonify({"status": "unavailable", "reason": "cloud environment"}), 200

    with camera_lock:
        if camera is None or not camera.isOpened():
            init_camera()
        camera_on = True
    return jsonify({"status": "camera started"})


@app.route("/stop_camera")
def stop_camera():
    global camera, camera_on
    if IS_CLOUD:
        return jsonify({"status": "unavailable", "reason": "cloud environment"}), 200

    with camera_lock:
        camera_on = False
        if camera is not None:
            camera.release()
            camera = None
    return jsonify({"status": "camera stopped"})


# ---- Env info route (untuk frontend menyesuaikan UI) ----

@app.route("/env_info")
def env_info():
    return jsonify({"is_cloud": IS_CLOUD})


# ---- Upload & detection routes ----

@app.route("/detections")
def detections():
    return jsonify(latest_detections)


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


# ===========================================================

if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 5000))
        app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
    finally:
        with camera_lock:
            if camera is not None:
                camera.release()
        with cctv_lock:
            if cctv_camera is not None:
                cctv_camera.release()