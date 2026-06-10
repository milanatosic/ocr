"""
Dataset i preprocessing za CRNN OCR model.
"""

import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path


# ── Karakter set ──────────────────────────────────────────────────────────────
CHARS = (
    " !\"#$%&'()*+,-./:;<=>?@"  # specijalni
    "0123456789"                  # brojevi
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # latinica velika
    "abcdefghijklmnopqrstuvwxyz"  # latinica mala
    "ČčĆćŠšŽžĐđ"                 # srpska latinica
    "БВГДЂЖЗИЛЉНЊПРСЋУФХЦЧЏШ"  # ćirilica velika
    "бвгдђжзиклљмњпрстћуфхцчџш"    # ćirilica mala
)
BLANK = 0  # CTC blank token, ne znaci razmak, vec nema karaktera ovde
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 je blank
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # +1 za blank

IMG_HEIGHT = 48
IMG_WIDTH = 768  # max širina, manje slike se padduju
CNN_REDUCTION_FACTOR = 4  # Većina CRNN modela smanjuje širinu 4 puta kroz Pooling slojeve


def encode_label(text):
    """Pretvori string u listu indeksa karaktera."""
    indices = []
    for c in text:
        if c in CHAR_TO_IDX:
            indices.append(CHAR_TO_IDX[c])
        # Nepoznate karaktere preskačemo
    return indices


def decode_prediction(logits):
    """
    CTC greedy dekodiranje.
    logits: [T, num_classes] numpy array ili tensor
    Za svaki vremenski korak uzmi karakter sa najvecom verovatnocom
    Spoji uzastopne karaktere u jedan
    Ukloni blank tokene
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()

    pred_indices = np.argmax(logits, axis=1)  # [T]

    # Ukloni ponovljene i blank tokene
    chars = []
    prev = None
    for idx in pred_indices:
        if idx != prev:
            if idx != BLANK:
                chars.append(IDX_TO_CHAR.get(idx, ""))
        prev = idx

    return "".join(chars)


def preprocess_image(img_bgr, target_height=IMG_HEIGHT, max_width=IMG_WIDTH):
    """
    Preprocess slike za CRNN:
    Vraća prerađenu sliku I novu validnu širinu (pre paddinga).
    """
    # Grayscale
    if len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr

    h, w = gray.shape
    if h == 0 or w == 0:
        return None, 0

    # Resize na target_height
    new_w = int(w * target_height / h)
    new_w = max(1, min(new_w, max_width))
    resized = cv2.resize(gray, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

    # Pad do max_width
    if new_w < max_width:
        pad = np.full((target_height, max_width), 255, dtype=np.uint8)
        pad[:, :new_w] = resized
        result = pad
    else:
        result = resized

    # Normalizacija
    result = result.astype(np.float32) / 255.0

    return result, new_w  # ◄ IZMENA: Vraćamo i novu širinu teksta bez paddinga


class OCRDataset(Dataset):
    def __init__(self, csv_path, base_dir, min_height=15, augment=False):
        """
        csv_path: putanja do dataset.csv
        base_dir: root folder projekta
        min_height: filtriraj slike ispod ove visine
        augment: da li da koristimo augmentaciju
        """
        self.base_dir = Path(base_dir)
        self.augment = augment

        df = pd.read_csv(csv_path)

        # Filtriraj prazne labele
        df = df[df['label'].notna()]
        df = df[df['label'].str.strip() != ""]

        # Filtriraj slike koje ne postoje ili su premale
        valid_rows = []
        for _, row in df.iterrows():
            img_path = self.base_dir / row['image_path']
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path)) #ucitaj sliku
            if img is None:
                continue
            if img.shape[0] < min_height:
                continue
            valid_rows.append(row)

        self.data = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"Dataset učitan: {len(self.data)} validnih uzoraka")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = self.base_dir / row['image_path']
        label_str = str(row['label'])

        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
            img_tensor = torch.tensor(img).unsqueeze(0)
            # Vraćamo: slika, labele, string, i broj vremenskih koraka (za celu širinu kao fallback)
            return img_tensor, torch.tensor([1], dtype=torch.long), label_str, (IMG_WIDTH // CNN_REDUCTION_FACTOR)

        if self.augment:
            img = self._augment(img)

        # ◄ IZMENA: Prihvatamo prerađenu sliku i njenu stvarnu širinu
        processed, valid_width = preprocess_image(img) 
        if processed is None:
            processed = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
            valid_width = IMG_WIDTH

        img_tensor = torch.tensor(processed).unsqueeze(0)  # [1, H, W]
        label_encoded = encode_label(label_str)

        if len(label_encoded) == 0:
            label_encoded = [1]  # fallback

        # Računamo koliko vremenskih koraka zauzima tekst nakon što prođe kroz CNN slojeve
        # max(1, ...) osigurava da nemamo nula koraka za ekstremno kratke slike
        valid_steps = max(1, valid_width // CNN_REDUCTION_FACTOR)

        return img_tensor, torch.tensor(label_encoded, dtype=torch.long), label_str, valid_steps

    def _augment(self, img):
        """Jednostavna augmentacija za trening."""
        if np.random.random() < 0.5:
            alpha = np.random.uniform(0.7, 1.3)
            beta = np.random.randint(-20, 20)
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

        if np.random.random() < 0.3:
            ksize = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, 1), 0)

        if np.random.random() < 0.2:
            noise = np.random.normal(0, 5, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return img


def collate_fn(batch):
    """
    Custom collate za CTC loss:
    Sada pakuje i dinamičke dužine ulaza (input_lengths) kako bi CTC ignorisao padding.
    """
    # ◄ IZMENA: Otpakujemo i valid_steps iz batch-a
    imgs, labels, label_strs, valid_steps = zip(*batch)

    imgs = torch.stack(imgs, 0)  # [B, 1, H, W]

    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_concat = torch.cat(labels)  # [Ukupan_broj_karaktera_u_batchu]
    
    # ◄ IZMENA: Pakujemo valid_steps svih slika u jedan tenzor
    input_lengths = torch.tensor(valid_steps, dtype=torch.long)

    # ◄ IZMENA: Sada vraćamo i input_lengths nazad u DataLoader!
    return imgs, labels_concat, label_lengths, label_strs, input_lengths