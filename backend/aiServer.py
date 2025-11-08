# ai_server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import base64
import numpy as np
import torch

app = Flask(__name__)
# Alleen requests vanaf http://localhost:3000 toestaan
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}})#"http://localhost:3000" voegen
 
# Gebruik GPU/Metal als beschikbaar
device = "mps" if torch.backends.mps.is_available() else "cpu"


# YOLO-model laden
model = YOLO("/Users/anna-elisetweeboom/datasets/final_combined2/training_runs/apple_banana_notapple_final_safe/weights/best.pt")
model.to(device)
print("Device in gebruik:", device)

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
    app.run(host="0.0.0.0", port=5001)


