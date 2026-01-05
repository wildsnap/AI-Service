# AI-Service

A Python-based AI microservice using FastAPI to serve a PyTorch image classification model.

## Features

- **FastAPI Framework**: Modern, fast web framework for building APIs
- **PyTorch Model**: Pre-trained ResNet18 for image classification
- **ImageNet Classes**: Supports 1000 ImageNet object classes
- **REST API**: Simple POST endpoint for image predictions
- **Easy Deployment**: Ready to deploy with uvicorn server

## Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/wildsnap/AI-Service.git
cd AI-Service
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained model and class labels:
```bash
python download_model.py
python download_labels.py
```

This will create two files:
- `model.pth`: Pre-trained ResNet18 weights
- `imagenet_classes.json`: ImageNet class labels

## Usage

### Start the Server

Run the server using uvicorn:
```bash
uvicorn main:app --reload
```

Or run directly with Python:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, you can access:
- Interactive API docs: `http://localhost:8000/docs`
- Alternative API docs: `http://localhost:8000/redoc`

### Endpoints

#### GET /
Returns service information and status.

**Response:**
```json
{
  "service": "AI Image Classification Service",
  "version": "1.0.0",
  "status": "running",
  "model_loaded": true
}
```

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### POST /predict
Classify an uploaded image.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: Image file (JPEG, PNG, etc.)

**Response:**
```json
{
  "class_id": 281,
  "class_name": "tabby cat",
  "confidence": 0.8532
}
```

### Example Usage

#### Using the Test Script (Recommended):
```bash
# Download a sample image (or use your own)
curl "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg" -o dog.jpg

# Run the test script
python test_api.py dog.jpg
```

The test script will check the health endpoint and then classify the image, displaying results in a user-friendly format.

#### Using cURL:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg"
```

#### Using Python requests:
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

#### Using the Interactive Docs:
1. Navigate to `http://localhost:8000/docs`
2. Click on the `/predict` endpoint
3. Click "Try it out"
4. Upload an image file
5. Click "Execute"

## Project Structure

```
AI-Service/
├── main.py                    # FastAPI application
├── download_model.py          # Script to download pre-trained model
├── download_labels.py         # Script to download ImageNet labels
├── test_api.py                # Test script to validate API functionality
├── requirements.txt           # Python dependencies
├── model.pth                  # Pre-trained model weights (generated)
├── imagenet_classes.json      # ImageNet class labels (generated)
└── README.md                  # This file
```

## Technical Details

### Model Architecture
- **Model**: ResNet18
- **Framework**: PyTorch
- **Pre-trained**: ImageNet (1000 classes)
- **Input Size**: 224x224 RGB images

### Image Preprocessing
Images are automatically preprocessed with:
- Resize to 256x256
- Center crop to 224x224
- Normalization using ImageNet mean and std

### Response Format
The API returns:
- `class_id`: Integer ID of the predicted class (0-999)
- `class_name`: Human-readable class name
- `confidence`: Confidence score (0.0-1.0)

## Error Handling

The API handles common errors:
- **400**: Invalid file type (non-image)
- **500**: Error processing image
- **503**: Model not loaded

## Development

### Running in Development Mode
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Testing the API
You can use the interactive documentation at `/docs` to test the API, or use tools like cURL, Postman, or Python requests library.

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.