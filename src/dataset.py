"""
Dataset i collate_fn za CRNN OCR.
Ključna promena: slike se skaliraju samo po visini,
a širina se pamti i padding se radi u collate_fn.
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
import random

# ── Karakteri ──────────────────────────────────────────────────────────────────
CHARS = (
    " !\"#$%&'()*+,-./:;<=>?@"  # specijalni
    "0123456789"                  # brojevi
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # latinica velika
    "abcdefghijklmnopqrstuvwxyz"  # latinica mala
    "ČčĆćŠšŽžĐđ"                 # srpska latinica
    "БВГДЂЖЗИЛЉНЊПРСЋУФХЦЧЏШ"   # ćirilica velika
    "бвгдђжзиклљмњпрстћуфхцчџш"  # ćirilica mala
)
BLANK = 0  # CTC blank token
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 je blank
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # +1 za blank


def encode_label(text):
    """Enkodira tekst u niz indeksa (bez praznog stringa)."""
    encoded = []
    for ch in text:
        if ch in CHAR_TO_IDX:
            encoded.append(CHAR_TO_IDX[ch])
    return encoded


def decode_prediction(log_probs):
    """CTC dekodiranje: argmax + uklanjanje blank-a i ponavljanja."""
    indices = np.argmax(log_probs, axis=1)
    chars = []
    prev_idx = -1
    for idx in indices:
        if idx != 0 and idx != prev_idx:  # 0 = blank
            if idx in IDX_TO_CHAR:
                chars.append(IDX_TO_CHAR[idx])
        prev_idx = idx
    return ''.join(chars)


# ── Augmentacije ─────────────────────────────────────────────────────────────

def augment_image(img):
    """Nasumične augmentacije za trening."""
    # Blaga rotacija
    if random.random() < 0.3:
        angle = random.uniform(-2, 2)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Promena osvetljenja
    if random.random() < 0.3:
        alpha = random.uniform(0.8, 1.2)
        beta = random.uniform(-20, 20)
        img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # Gaussian blur
    if random.random() < 0.2:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # Šum
    if random.random() < 0.2:
        noise = np.random.normal(0, 10, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


# ── Dataset ──────────────────────────────────────────────────────────────────

class OCRDataset(Dataset):
    def __init__(self, csv_path, base_dir, img_height=48, min_height=15, augment=False):
        self.df = pd.read_csv(csv_path)
        self.base_dir = Path(base_dir)
        self.img_height = img_height
        self.min_height = min_height
        self.augment = augment

        # Filtriraj nevalidne
        valid_rows = []
        for idx, row in self.df.iterrows():
            img_path = self.base_dir / row['image_path']
            if img_path.exists():
                valid_rows.append(idx)
        self.df = self.df.loc[valid_rows].reset_index(drop=True)
        print(f"Dataset učitan: {len(self.df)} validnih uzoraka")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.base_dir / row['image_path']
        label = str(row['label'])

        # Učitaj sliku
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback: vraćamo prvi uzorak
            return self.__getitem__(0)

        h, w = img.shape

        # Preskoči premale slike
        if h < self.min_height:
            return self.__getitem__(0)

        # ── KLJUČNO: skaliraj SAMO po visini, čuvaj aspect ratio ──
        new_height = self.img_height
        aspect_ratio = w / h
        new_width = max(int(new_height * aspect_ratio), 1)  # minimum 1px širine

        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Augmentacija (samo za trening)
        if self.augment:
            img = augment_image(img)

        # Normalizacija na [0, 1]
        img = img.astype(np.float32) / 255.0

        # Konvertuj u tensor [1, H, W]
        img_tensor = torch.from_numpy(img).unsqueeze(0)

        # Enkodiraj labelu
        label_encoded = encode_label(label)
        label_tensor = torch.tensor(label_encoded, dtype=torch.long)

        return img_tensor, label_tensor, label, new_width


def collate_fn(batch):
    """
    Padding po širini da sve slike u batch-u budu iste širine.
    
    Returns:
        images_padded:  [B, 1, H, max_W]
        labels_concat:  [sum_label_lengths]  — sve labele spojene
        label_lengths:  [B]
        input_lengths:  [B]  — T za svaki uzorak (posle CNN-a)
        label_strs:     list of strings
    """
    # Filtriraj nevalidne uzorke
    batch = [(img, lbl, s, w) for img, lbl, s, w in batch if img is not None and lbl is not None and len(lbl) > 0]
    if len(batch) == 0:
        return None, None, None, None, None

    images, labels, label_strs, widths = zip(*batch)

    # ── Padding po širini ──
    # Pronađi maksimalnu širinu u batch-u
    max_width = max(img.shape[2] for img in images)  # img shape: [1, H, W]

    # Pad sve slike na istu širinu
    padded_images = []
    for img in images:
        c, h, w = img.shape
        if w < max_width:
            # Pad desno nulama
            padding = torch.zeros(c, h, max_width - w)
            padded = torch.cat([img, padding], dim=2)
        else:
            padded = img
        padded_images.append(padded)

    images_batch = torch.stack(padded_images, dim=0)  # [B, 1, H, max_W]

    # ── Labele: spojene za CTC ──
    label_lengths = torch.tensor([len(lbl) for lbl in labels], dtype=torch.long)
    labels_concat = torch.cat(labels)  # Sve labele jedna za drugom

    # ── input_lengths za CTC ──
    # Izračunaj T (temporalnu dimenziju) prolaskom kroz CNN arhitekturu
    # Ovo je APPROKSIMACIJA — tačna vrednost zavisi od CNN slojeva
    # Za standardni CRNN sa 4 conv sloja (stride 2 po visini, 1 po širini):
    #   T ≈ max_width (jer se širina ne smanjuje poolingom po širini)
    #   ILI T ≈ max_width / (product svih horizontalnih stride-ova)
    #
    # Za sada koristimo T = max_width, a korigovaćemo posle prvog prolaska
    # (ovo se radi u train.py)

    # Placeholder — tačne dužine se izračunavaju u train.py
    input_lengths = torch.full((len(batch),), max_width, dtype=torch.long)

    return images_batch, labels_concat, label_lengths, input_lengths, list(label_strs)