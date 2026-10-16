import os
import json
from flask import Flask, send_from_directory, jsonify, request

app = Flask(__name__, static_folder=".")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DESCRIPTORS_FILE = os.path.join(BASE_DIR, "labeled_descriptors.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LIB_DIR = os.path.join(BASE_DIR, "lib")

os.makedirs(DATASET_DIR, exist_ok=True)

# -------------------------------------------------------------
# Static files routing
# -------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

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
        with open(DESCRIPTORS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return jsonify({"success": True, "message": "Model dataset berhasil disimpan!"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/dataset-info", methods=["GET"])
def get_dataset_info():
    if not os.path.exists(DATASET_DIR):
        return jsonify({"persons": [], "totalImages": 0})
    
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    persons = []
    total_images = 0

    for name in sorted(os.listdir(DATASET_DIR)):
        p_dir = os.path.join(DATASET_DIR, name)
        if os.path.isdir(p_dir):
            images = [
                f"/dataset/{name}/{img}" for img in sorted(os.listdir(p_dir))
                if os.path.splitext(img)[1].lower() in valid_exts
            ]
            persons.append({
                "name": name,
                "imageCount": len(images),
                "images": images
            })
            total_images += len(images)

    return jsonify({"persons": persons, "totalImages": total_images})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 55)
    print("   FACE RECOGNITION SERVER (TEACHABLE MACHINE STYLE)")
    print("=" * 55)
    print(f" Server berjalan di: http://localhost:{port}")
    print(" Buka link di atas di browser Anda (Google Chrome / Edge)")
    print(" Tekan Ctrl+C untuk menghentikan server\n")
    app.run(host="0.0.0.0", port=port, debug=False)
