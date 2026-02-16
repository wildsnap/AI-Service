from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "wildsnap_model_v4.pth" 
NUM_CLASSES = 28

def load_model():
    model = models.efficientnet_b1(weights=None)

    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, NUM_CLASSES)
    )

    try:
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location="cpu")
        )
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise e

    model.eval()
    return model