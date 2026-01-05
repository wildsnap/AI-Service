"""
Script to download ImageNet class labels and save them as imagenet_classes.json
"""
import json
import urllib.request


def download_imagenet_labels():
    """Download ImageNet class labels from a reliable source."""
    print("Downloading ImageNet class labels...")
    
    # URL for ImageNet class labels
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    
    try:
        with urllib.request.urlopen(url) as response:
            labels_list = json.loads(response.read().decode())
        
        # Convert list to dictionary with index as key
        labels_dict = {str(i): label for i, label in enumerate(labels_list)}
        
        print("Saving labels to imagenet_classes.json...")
        with open("imagenet_classes.json", "w") as f:
            json.dump(labels_dict, f, indent=2)
        
        print(f"Successfully saved {len(labels_dict)} ImageNet class labels!")
        
    except Exception as e:
        print(f"Error downloading labels: {e}")
        print("Creating default labels...")
        
        # Create default labels if download fails
        default_labels = {str(i): f"class_{i}" for i in range(1000)}
        with open("imagenet_classes.json", "w") as f:
            json.dump(default_labels, f, indent=2)
        
        print("Created default labels file.")


if __name__ == "__main__":
    download_imagenet_labels()
