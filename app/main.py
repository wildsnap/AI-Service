from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
import json
import uvicorn
from PIL import Image

from core.model import load_model
from core.preprocess import preprocess
from utils.image_loader import load_image_from_bytes, load_image_from_url

app = FastAPI(title="WildSnap AI Service")
model = load_model()

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

    class_id = predicted_idx.item()
    
    predicted_name = "Unknown"
    if 0 <= class_id < len(class_names):
        predicted_name = class_names[class_id]

    return {
        "class_id": class_id,
        "class_name": predicted_name,
        "confidence": round(confidence.item() * 100, 2)
    }

@app.get("/")
def health_check():
    return {"status": "ok", "service": "WildSnap AI (EfficientNet-B1)"}

@app.post("/predict")
async def predict_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = load_image_from_bytes(contents)
        result = predict_image(image)
        return {"status": "success", **result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/predict/url")
async def predict_url(payload: PredictUrlRequest):
    try:
        image = load_image_from_url(payload.image_url)
        result = predict_image(image)
        return {"status": "success", **result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)