from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
import json
import uvicorn
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from core.model import load_model
from core.preprocess import preprocess
from utils.image_loader import load_image_from_bytes, load_image_from_url

"""
Class ID ของสัตว์ทั้งหมดที่ YOLO รู้จักใน COCO Dataset
14: นก (Bird)
15: แมว (Cat)
16: หมา (Dog)
17: ม้า (Horse)
18: แกะ (Sheep)
19: วัว (Cow)
20: ช้าง (Elephant)
21: หมี (Bear)
22: ม้าลาย (Zebra)
23: ยีราฟ (Giraffe)
"""

ANIMAL_CLASSES = [14, 15, 16]
MINIMUM_THRESHOLD = 0.65

app = FastAPI(title="WildSnap AI Service")
model = load_model()
yolov11_model = YOLO('yolo11n.pt')


BASE_DIR = Path(__file__).resolve().parent
CLASSES_PATH = BASE_DIR / "model" / "classes.json"

if CLASSES_PATH.exists():
    with open(CLASSES_PATH, "r") as f:
        class_names = json.load(f)
else:
    print("⚠️ Warning: classes.json not found!")
    class_names = []

class PredictUrlRequest(BaseModel):
    image_url: str

def predict_image(image):
    # Preprocess
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    if confidence.item() < MINIMUM_THRESHOLD:
        return {
            "class_id": None,
            "class_name": "Uncertain Prediction or Unrecognized Species",
            "confidence": round(confidence.item() * 100, 2)
        }

    class_id = predicted_idx.item()
    
    predicted_name = "Unknown"
    if 0 <= class_id < len(class_names):
        predicted_name = class_names[class_id]

    return {
        "class_id": class_id,
        "class_name": predicted_name,
        "confidence": round(confidence.item() * 100, 2)
    }

def detect_and_predict(image: Image.Image):
    results = yolov11_model.predict(source=image, conf=0.5)
    
    if len(results[0].boxes) == 0:
        img_w, img_h = image.size
        shortest_edge = min(img_w, img_h)
        crop_size = int(shortest_edge * 0.8)
        
        x1 = (img_w - crop_size) // 2
        y1 = (img_h - crop_size) // 2
        x2 = x1 + crop_size
        y2 = y1 + crop_size
    else:
        box = results[0].boxes[0].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
    
    cropped_animal_image = image.crop((x1, y1, x2, y2))
    species_result = predict_image(cropped_animal_image)
    
    return {"status": "success", **species_result}

def calculate_blur(image_cv):
    """Calculate the variance of the Laplacian to estimate blur."""
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def smart_predict_pipeline(image: Image.Image):
    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    img_h, img_w = img_bgr.shape[:2]
    image_area = img_w * img_h

    results = yolov11_model.predict(source=img_bgr, classes=ANIMAL_CLASSES, verbose=False)
    
    best_box = None
    yolo_conf = 0.0

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > yolo_conf:
                yolo_conf = conf
                best_box = box.xyxy[0].cpu().numpy()

    if best_box is None:
        shortest_edge = min(img_w, img_h)
        crop_size = int(shortest_edge * 0.8)
        
        x1 = (img_w - crop_size) // 2
        y1 = (img_h - crop_size) // 2
        x2 = x1 + crop_size
        y2 = y1 + crop_size
        
        best_box = [x1, y1, x2, y2]
        yolo_conf = 0.0

    x1, y1, x2, y2 = map(int, best_box)
    box_w, box_h = x2 - x1, y2 - y1
    size_ratio = (box_w * box_h) / image_area
    size_score = min(size_ratio / 0.15, 1.0) # Animal size > 15% of image area is considered good

    cropped_bgr = img_bgr[y1:y2, x1:x2]
    blur_val = calculate_blur(cropped_bgr)
    quality_score = min(blur_val / 200.0, 1.0) # Blur value > 200 is considered sharp

    cropped_pil = image.crop((x1, y1, x2, y2))
    
    input_tensor = preprocess(cropped_pil)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.softmax(output[0], dim=0)
        eff_conf_tensor, predicted_idx = torch.max(probabilities, 0)

    eff_conf = eff_conf_tensor.item()
    class_id = predicted_idx.item()
    
    predicted_name = "Unknown"
    if 0 <= class_id < len(class_names):
        predicted_name = class_names[class_id]

    final_score = (eff_conf * 0.5) + (yolo_conf * 0.2) + (size_score * 0.15) + (quality_score * 0.15)
    is_reliable = bool(final_score >= MINIMUM_THRESHOLD)

    return {
        "status": "success",
        "class_id": class_id,
        "class_name": predicted_name,
        "scores": {
            "efficientnet_confidence": round(eff_conf * 100, 2),
            "yolo_match_confidence": round(yolo_conf * 100, 2),
            "size_score": round(size_score * 100, 2),
            "quality_score": round(quality_score * 100, 2),
            "FINAL_SCORE": round(final_score * 100, 2)
        },
        "is_reliable": is_reliable,
        "recommendation": "Predict Success" if is_reliable else "The image is blurry or too far away. Please take a clearer photo."
    }


@app.get("/")
def health_check():
    return {"status": "ok", "service": "WildSnap AI (EfficientNet-B1)"}


@app.post("/predict")
async def predict_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = load_image_from_bytes(contents)
        result = detect_and_predict(image)

        return {"status": "success", **result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/predict/url")
async def predict_url(payload: PredictUrlRequest):
    try:
        image = load_image_from_url(payload.image_url)
        result = detect_and_predict(image)
        return {"status": "success", **result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/smart-predict")
async def smart_predict_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = load_image_from_bytes(contents)
        result = smart_predict_pipeline(image)
        return result 
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/smart-predict/url")
async def smart_predict_url_endpoint(payload: PredictUrlRequest):
    try:
        image = load_image_from_url(payload.image_url)
        result = smart_predict_pipeline(image)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)