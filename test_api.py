"""
Simple test script to demonstrate the API usage.
This script sends a test image to the /predict endpoint and displays the result.
"""
import requests
import sys
from pathlib import Path


def test_predict_endpoint(image_path: str):
    """
    Test the /predict endpoint with an image file.
    
    Args:
        image_path: Path to the image file to classify
    """
    url = "http://localhost:8000/predict"
    
    # Check if the file exists
    if not Path(image_path).exists():
        print(f"Error: File '{image_path}' not found.")
        sys.exit(1)
    
    # Send the request
    print(f"Sending image to {url}...")
    try:
        with open(image_path, 'rb') as f:
            files = {"file": (Path(image_path).name, f, "image/jpeg")}
            response = requests.post(url, files=files)
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            print("\n✓ Prediction successful!")
            print(f"  Class ID:    {result['class_id']}")
            print(f"  Class Name:  {result['class_name']}")
            print(f"  Confidence:  {result['confidence']:.2%}")
        else:
            print(f"\n✗ Error: {response.status_code}")
            print(f"  {response.json()}")
    
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to the API server.")
        print("  Make sure the server is running with: uvicorn main:app --reload")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        sys.exit(1)


def test_health_endpoint():
    """Test the /health endpoint."""
    url = "http://localhost:8000/health"
    
    print(f"Checking health endpoint at {url}...")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Server is healthy")
            print(f"  Model loaded: {result['model_loaded']}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("✗ Error: Could not connect to the API server.")
        print("  Make sure the server is running with: uvicorn main:app --reload")
        sys.exit(1)


if __name__ == "__main__":
    # Test health endpoint
    test_health_endpoint()
    print()
    
    # Test predict endpoint
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <image_path>")
        print("\nExample:")
        print("  python test_api.py dog.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    test_predict_endpoint(image_path)
