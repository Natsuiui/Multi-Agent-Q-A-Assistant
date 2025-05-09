import os
import requests
from pathlib import Path

MODEL_FILENAME = "mistral-7b-instruct-v0.1.Q2_K.gguf"
MODEL_PATH = Path(__file__).resolve().parent / MODEL_FILENAME

# Replace with your actual file ID
MODEL_URL = "https://drive.google.com/uc?export=download&id=1BZYGTX0vxTsXtWmQUk4ZgkXWwfyK8DD4"

def download_model():
    if MODEL_PATH.exists():
        print(f"✅ Model already exists at: {MODEL_PATH}")
        return

    print(f"⬇️ Downloading model from: {MODEL_URL}")
    with requests.get(MODEL_URL, stream=True) as r:
        if r.status_code != 200:
            raise RuntimeError(f"Failed to download model (status {r.status_code})")

        with open(MODEL_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print("✅ Model downloaded successfully.")

if __name__ == "__main__":
    download_model()
