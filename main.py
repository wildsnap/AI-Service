import io
import json
from pathlib import Path
from typing import Dict

import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from torchvision.models import resnet18

# Initialize FastAPI app
app = FastAPI(
    title="AI Image Classification Service",
    description="PyTorch-based image classification microservice",
    version="1.0.0"
)

# Global variables for model and class labels
model = None
class_labels = None


def load_model():
    """Load the pre-trained ResNet18 model with custom weights."""
    global model
    
    model_path = Path("model.pth")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. Please ensure model.pth exists."
        )
    
    # Initialize ResNet18 architecture
    model = resnet18(weights=None)
    
    # Load the custom weights
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        print("Model loaded successfully from model.pth")
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {str(e)}")


def load_class_labels():
    """Load ImageNet class labels."""
    global class_labels
    
    # ImageNet class labels (using a subset for demonstration)
    # In production, you would load the full 1000 classes from a file
    labels_path = Path("imagenet_classes.json")
    
    if labels_path.exists():
        with open(labels_path, 'r') as f:
            class_labels = json.load(f)
    else:
        # Default to indices if labels file doesn't exist
        class_labels = {str(i): f"class_{i}" for i in range(1000)}
    
    print(f"Loaded {len(class_labels)} class labels")


@app.on_event("startup")
async def startup_event():
    """Initialize model and labels on startup."""
    try:
        load_model()
        load_class_labels()
    except Exception as e:
        print(f"Warning: {str(e)}")
        print("Server started but model may not be available.")


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "AI Image Classification Service",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model input."""
    # Define the same transforms used in ImageNet training
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Convert image to RGB if it's not
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms and add batch dimension
    return transform(image).unsqueeze(0)


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    """
    Predict the class of an uploaded image.
    
    Args:
        file: Image file uploaded by the user
        
    Returns:
        Dictionary containing class_id, class_name, and confidence
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model.pth exists and restart the service."
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, etc.)"
        )
    
    try:
        # Read image file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess image
        input_tensor = preprocess_image(image)
        
        # Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top prediction
            confidence, class_id = torch.max(probabilities, dim=0)
            
            class_id = class_id.item()
            confidence = confidence.item()
        
        # Get class name
        class_name = class_labels.get(str(class_id), f"class_{class_id}")
        
        return {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(confidence, 4)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
