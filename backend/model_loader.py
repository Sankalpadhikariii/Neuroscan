import os
import requests
import torch

MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://huggingface.co/Sankalpadhikari/vgg19_final_20260110_154609/commit/e63356096d85486ba665c9d854f15da3accdf692"
)

MODEL_PATH = "/app/models/vgg19.pth"

device = torch.device("cpu")
_model = None


def download_model():
    if os.path.exists(MODEL_PATH):
        return

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    with requests.get(MODEL_URL, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def get_model():
    global _model
    if _model is not None:
        return _model

    download_model()
    _model = torch.load(MODEL_PATH, map_location=device)
    _model.eval()
    return _model
