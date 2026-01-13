import io
import requests
from PIL import Image

def load_image_from_bytes(contents: bytes) -> Image.Image:
    return Image.open(io.BytesIO(contents)).convert("RGB")

def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")
