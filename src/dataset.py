"""
Dataset i collate_fn za CRNN OCR.

Promene:
- collate_fn vraća 4 vrednosti (label_strs ispravno raspakovan)
- Jača augmentacija
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import random

# ── Karakteri ──────────────────────────────────────────────────────────────────
CHARS = (
    " !\"#$%&'()*+,-./:;<=>?@"
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "ČčĆćŠšŽžĐđ"
    "БВГДЂЖЗИЛЉНЊПРСЋУФХЦЧЏШ"
    "бвгдђжзиклљмњпрстћуфхцчџш"
)
BLANK = 0
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1


def encode_label(text):
    return [CHAR_TO_IDX[ch] for ch in text if ch in CHAR_TO_IDX]


def decode_prediction(log_probs):
    indices = np.argmax(log_probs, axis=1)
    chars = []
    prev_idx = -1
    for idx in indices:
        if idx != 0 and idx != prev_idx:
            if idx in IDX_TO_CHAR:
                chars.append(IDX_TO_CHAR[idx])
        prev_idx = idx
    return ''.join(chars)


def augment_image(img):
    """Jača augmentacija za OCR."""
    # Rotacija
    if random.random() < 0.4:
        angle = random.uniform(-3, 3)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Osvetljenje/kontrast
    if random.random() < 0.5:
        alpha = random.uniform(0.7, 1.3)
        beta = random.uniform(-30, 30)
        img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # Blur
    if random.random() < 0.25:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # Šum
    if random.random() < 0.3:
        sigma = random.uniform(5, 15)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Erozija/dilatacija
    if random.random() < 0.2:
        kernel = np.ones((2, 2), np.uint8)
        if random.random() < 0.5:
            img = cv2.erode(img, kernel, iterations=1)
        else:
            img = cv2.dilate(img, kernel, iterations=1)

    # JPEG kompresija
    if random.random() < 0.2:
        quality = random.randint(50, 85)
        _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

    return img


class OCRDataset(Dataset):
    def __init__(self, csv_path, base_dir, img_height=48, min_height=15, augment=False):
        self.df = pd.read_csv(csv_path)
        self.base_dir = Path(base_dir)
        self.img_height = img_height
        self.min_height = min_height
        self.augment = augment

        # Filtriraj nevalidne
        valid = []
        for idx, row in self.df.iterrows():
            img_path = self.base_dir / row['image_path']
            if img_path.exists():
                valid.append(idx)
        self.df = self.df.loc[valid].reset_index(drop=True)
        print(f"Dataset učitan: {len(self.df)} validnih uzoraka")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.base_dir / row['image_path']
        label = str(row['label'])

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return self.__getitem__(0)

        h, w = img.shape
        if h < self.min_height:
            return self.__getitem__(0)

        # Skaliraj po visini
        new_h = self.img_height
        new_w = max(int(new_h * w / h), 1)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if self.augment:
            img = augment_image(img)

        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).unsqueeze(0)

        label_encoded = encode_label(label)
        if len(label_encoded) == 0:
            return self.__getitem__(0)
        label_tensor = torch.tensor(label_encoded, dtype=torch.long)

        return img_tensor, label_tensor, label


def collate_fn(batch):
    """Vraća 4 vrednosti: imgs, labels_concat, label_lengths, label_strs."""
    batch = [(img, lbl, s) for img, lbl, s in batch
             if img is not None and lbl is not None and len(lbl) > 0]
    if len(batch) == 0:
        return None, None, None, None

    images, labels, label_strs = zip(*batch)

    # Padding po širini
    max_width = max(img.shape[2] for img in images)
    padded = []
    for img in images:
        c, h, w = img.shape
        if w < max_width:
            pad = torch.zeros(c, h, max_width - w)
            padded.append(torch.cat([img, pad], dim=2))
        else:
            padded.append(img)

    images_batch = torch.stack(padded, dim=0)
    label_lengths = torch.tensor([len(lbl) for lbl in labels], dtype=torch.long)
    labels_concat = torch.cat(list(labels))

    return images_batch, labels_concat, label_lengths, list(label_strs)