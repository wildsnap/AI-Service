"""
Script to download a pre-trained ResNet18 model and save it as model.pth
This script should be run once to prepare the model for the service.
"""
import torch
from torchvision.models import resnet18, ResNet18_Weights


def download_and_save_model():
    """Download pre-trained ResNet18 and save weights to model.pth"""
    print("Downloading pre-trained ResNet18 model...")
    
    # Load pre-trained ResNet18 with ImageNet weights
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    print("Saving model weights to model.pth...")
    torch.save(model.state_dict(), "model.pth")
    
    print("Model saved successfully!")
    print("You can now start the API service with: uvicorn main:app --reload")


if __name__ == "__main__":
    download_and_save_model()
