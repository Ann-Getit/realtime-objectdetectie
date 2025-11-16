# ai_server.py
import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
os.makedirs("/tmp/Ultralytics", exist_ok=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import numpy as np
import torch
from ultralytics import YOLO


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "weights", "best.pt"))
print("💡 Volledig MODEL_PATH:", MODEL_PATH)
# Controleer eerst of het model echt bestaat
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Het model kon niet gevonden worden: {MODEL_PATH}")

DEVICE = os.getenv("DEVICE", "cpu")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
print("💡 Volledig MODEL_PATH:", MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else DEVICE


model = YOLO(MODEL_PATH)
model.to(device)
print("✅ Model geladen:", device)




# Alleen requests vanaf http://localhost:3000 toestaan
CORS(app, resources={r"/*": {"origins": [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://localhost:3000", 
    "http://127.0.0.1:3000",
    "https://ann-getit.github.io/realtime-objectdetectie/"
    ]}})  #HIER LATER DE FRONTEND URL INVOEREN---------------


print("MODEL_PATH:", MODEL_PATH)
print("DEVICE:", DEVICE)
print("CONFIDENCE_THRESHOLD:", CONFIDENCE_THRESHOLD)

#"http://localhost:3000"

 


# YOLO-model laden
#model = YOLO(MODEL_PATH)
#model.to(device)
#print("Device in gebruik:", device)

@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.get_json()
        img_base64 = data.get("image")
        if not img_base64:
            return jsonify({"error": "Geen 'image' veld"}), 400
    
        # ---- Alleen voor testen (tijdelijk) ----
        # print("Ontvangen afbeelding:", len(img_base64))
        # return jsonify({"detections": []})
        

    # Base64 → PIL → numpy
        img_bytes = base64.b64decode(img_base64)

        try:
             img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            return jsonify({"error": "Kon afbeelding niet openen"}), 400


        frame = np.array(img)

        print("Frame shape:", frame.shape)

  # YOLO detectie
        with torch.no_grad():  # geen gradients voor inferentie
            # In ai_server.py
            results = model(frame, device=device, imgsz=384)  # laat model zelf resize doen

   
        # Resultaten parsen
            detections = []
            detected_classes = []

            for r in results:
                for box in r.boxes:
                    class_name = model.names[int(box.cls)]
                    if class_name == "not_apple":  # filter
                       continue

        # Voeg alleen de gefilterde classes toe
                    detected_classes.append(class_name)
                    detections.append({
                        "class": class_name,
                        "confidence": float(box.conf),
                        "bbox": box.xyxy[0].tolist()
                     })

            app.logger.info(f"Frame shape: {frame.shape},  Detecties: {len(detections)}, Classes: {detected_classes}")
        
        print("YOLO intern shape:", results[0].path, results[0].orig_shape, results[0].boxes.shape)
        print(model.names)



         # VRAM cleanup voor M1 GPU
        if device == "mps":
            torch.mps.empty_cache()

    
        return jsonify({"detections": detections})
    
    except Exception as e:
        # Zorg dat de server niet crasht bij fouten
        return jsonify({"error": "Flask server error", "details": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050)) # 5000 is alleen fallback voor lokaal testen 
    app.run(host="0.0.0.0", port=port)


