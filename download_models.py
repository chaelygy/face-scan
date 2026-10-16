import os
import sys
import urllib.request

# Ensure UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
LIB_DIR = os.path.join(os.path.dirname(__file__), "lib")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LIB_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

BASE_CDN_MODEL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/"
BASE_CDN_LIB = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/dist/face-api.js"

MODEL_FILES = [
    "ssd_mobilenetv1_model-weights_manifest.json",
    "ssd_mobilenetv1_model.bin",
    "face_landmark_68_model-weights_manifest.json",
    "face_landmark_68_model.bin",
    "face_recognition_model-weights_manifest.json",
    "face_recognition_model.bin",
]

print("Mengunduh face-api.js library...")
lib_path = os.path.join(LIB_DIR, "face-api.js")
if not os.path.exists(lib_path):
    urllib.request.urlretrieve(BASE_CDN_LIB, lib_path)
    print("   [OK] face-api.js selesai diunduh.")
else:
    print("   [INFO] face-api.js sudah ada.")

print("\nMengunduh model AI (SSD MobileNet + Landmarks + ResNet-34)...")
for f in MODEL_FILES:
    dest = os.path.join(MODELS_DIR, f)
    if not os.path.exists(dest):
        url = BASE_CDN_MODEL + f
        print(f"   Downloading {f}...")
        urllib.request.urlretrieve(url, dest)
        print(f"   [OK] {f} selesai.")
    else:
        print(f"   [INFO] {f} sudah ada.")

print("\nSemua model dan library AI siap secara offline!")
