import os
import json
import base64
import time
import re
import urllib.parse
from flask import Flask, send_from_directory, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DESCRIPTORS_FILE = os.path.join(BASE_DIR, "labeled_descriptors.json")
DATASET_INFO_FILE = os.path.join(BASE_DIR, "dataset_info.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LIB_DIR = os.path.join(BASE_DIR, "lib")

os.makedirs(DATASET_DIR, exist_ok=True)

app = Flask(__name__, static_folder=BASE_DIR)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit for uploading images/descriptors

# -------------------------------------------------------------
# CORS Middleware
# -------------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

# -------------------------------------------------------------
# Static files routing
# -------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/dataset/<path:filename>")
def serve_dataset(filename):
    return send_from_directory(DATASET_DIR, filename)

@app.route("/models/<path:filename>")
def serve_models(filename):
    return send_from_directory(MODELS_DIR, filename)

@app.route("/lib/<path:filename>")
def serve_lib(filename):
    return send_from_directory(LIB_DIR, filename)

@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(BASE_DIR, path)

# -------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------
@app.route("/api/descriptors", methods=["GET"])
def get_descriptors():
    if os.path.exists(DESCRIPTORS_FILE):
        try:
            with open(DESCRIPTORS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({
        "metadata": {"totalPersons": 0, "totalImages": 0},
        "descriptors": []
    }), 200

@app.route("/api/save-descriptors", methods=["POST"])
def save_descriptors():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"success": False, "error": "No JSON payload provided"}), 400

        with open(DESCRIPTORS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
        print(f"Saved {len(data.get('descriptors', []))} face descriptors to {DESCRIPTORS_FILE}")
        return jsonify({"success": True, "message": "Model dataset berhasil disimpan!"})
    except Exception as e:
        print("Error saving descriptors:", e)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/dataset-info", methods=["GET"])
def get_dataset_info():
    if not os.path.exists(DATASET_DIR):
        return jsonify({"persons": [], "totalImages": 0})
    
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    persons = []
    total_images = 0

    try:
        for name in sorted(os.listdir(DATASET_DIR)):
            p_dir = os.path.join(DATASET_DIR, name)
            if os.path.isdir(p_dir):
                files = [
                    img for img in sorted(os.listdir(p_dir))
                    if os.path.splitext(img)[1].lower() in valid_exts
                ]
                images = [
                    f"/dataset/{urllib.parse.quote(name)}/{urllib.parse.quote(img)}" 
                    for img in files
                ]
                persons.append({
                    "name": name,
                    "imageCount": len(images),
                    "images": images
                })
                total_images += len(images)

        result = {"persons": persons, "totalImages": total_images}
        
        # Cache to dataset_info.json
        try:
            with open(DATASET_INFO_FILE, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
        except Exception:
            pass

        return jsonify(result)
    except Exception as e:
        if os.path.exists(DATASET_INFO_FILE):
            with open(DATASET_INFO_FILE, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        return jsonify({"persons": [], "totalImages": 0, "error": str(e)})

@app.route("/api/upload-images", methods=["POST"])
def upload_images():
    try:
        payload = request.get_json(force=True)
        person_name = payload.get("personName", "").strip()
        images = payload.get("images", [])

        if not person_name or not images:
            return jsonify({"success": False, "error": "personName atau images tidak valid"}), 400

        safe_name = re.sub(r'[<>:"/\\|?*]', '_', person_name)
        person_dir = os.path.join(DATASET_DIR, safe_name)
        os.makedirs(person_dir, exist_ok=True)

        saved_count = 0
        ts = int(time.time() * 1000)

        for idx, img_data in enumerate(images):
            ext = ".jpg"
            b64_data = img_data
            match = re.match(r"^data:image/([a-zA-Z0-9+]+);base64,(.+)$", img_data)
            if match:
                mime_ext = match.group(1).lower()
                ext = ".jpg" if mime_ext == "jpeg" else f".{mime_ext}"
                b64_data = match.group(2)

            file_bytes = base64.b64decode(b64_data)
            filename = f"photo_{ts}_{idx + 1}{ext}"
            file_path = os.path.join(person_dir, filename)
            with open(file_path, "wb") as f:
                f.write(file_bytes)
            saved_count += 1

        print(f"Uploaded {saved_count} photos for {safe_name}")
        return jsonify({"success": True, "count": saved_count, "personName": safe_name})
    except Exception as e:
        print("Upload error:", e)
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 55)
    print("   FACE RECOGNITION SERVER (TEACHABLE MACHINE STYLE)")
    print("=" * 55)
    print(f" Server berjalan di: http://localhost:{port}")
    print(" Tekan Ctrl+C untuk menghentikan server\n")
    app.run(host="0.0.0.0", port=port, debug=False)
