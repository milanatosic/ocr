"""
Predikcija teksta na novoj slici reda.
Pokretanje:
    python predict.py --img putanja/do/slike.jpg
"""

import torch
import cv2
import argparse
from pathlib import Path

from crnn_model import CRNN
from dataset import NUM_CLASSES, preprocess_image, decode_prediction

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = str(BASE_DIR / "checkpoints" / "best_model.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model = CRNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


def predict(model, img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Ne mogu da učitam sliku: {img_path}")
        return ""

    processed = preprocess_image(img)
    if processed is None:
        return ""

    img_tensor = torch.tensor(processed).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]

    with torch.no_grad():
        logits = model(img_tensor)  # [T, 1, C]
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)
        log_probs_np = log_probs.squeeze(1).cpu().numpy()  # [T, C]

    return decode_prediction(log_probs_np)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="Putanja do slike reda")
    args = parser.parse_args()

    model = load_model()
    result = predict(model, args.img)
    print(f"Prepoznati tekst: {result}")