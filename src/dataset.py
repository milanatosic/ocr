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
    "АБВГДЂЕЖЗИЈКЛЉМНЊОПРСТЋУФХЦЧЏШ"  # ćirilica velika
    "абвгдђежзијклљмњопрстћуфхцчџш"    # ćirilica mala
)
BLANK = 0  # CTC blank token, ne znaci razmak, vec nema karaktera ovde
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 je blank
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # +1 za blank

IMG_HEIGHT = 48
IMG_WIDTH = 768  # max širina, manje slike se padduju


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
    1. Grayscale
    2. Resize na target_height, čuvajući proporcije
    3. Pad ili crop na max_width
    4. Normalizacija [0, 1]
    Sve slike moraju biti iste velicine za batch procesiranje. 
    Visina je uvek 48px. Sirina je proporcionalna originalu ali maksimalno 768px - 
    krace slike se dopunjuju belim pikselima sa desna(padding). Model uci da ignorise padding jer
    tamo nema teksta
    """
    # Grayscale
    if len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr

    h, w = gray.shape
    if h == 0 or w == 0:
        return None

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
    result = result.astype(np.float32) / 255.0 # pikseli slike su celi brojevi od 0-255

    return result


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
            # Vrati placeholder ako slika ne može da se učita
            img = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
            img = torch.tensor(img).unsqueeze(0)
            return img, torch.tensor([1], dtype=torch.long), label_str

        if self.augment:
            img = self._augment(img)

        processed = preprocess_image(img) # resize + pad + normalizacija
        if processed is None:
            processed = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)

        img_tensor = torch.tensor(processed).unsqueeze(0)  # dodaj kanal dimenziju [1, H, W]
        label_encoded = encode_label(label_str) # string -> indeksi

        if len(label_encoded) == 0:
            label_encoded = [1]  # fallback

        return img_tensor, torch.tensor(label_encoded, dtype=torch.long), label_str

    def _augment(self, img):
        """Jednostavna augmentacija za trening."""
        # Slučajni brightness/contrast
        if np.random.random() < 0.5:
            alpha = np.random.uniform(0.7, 1.3)  # kontrast
            beta = np.random.randint(-20, 20)      # brightness
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

        # Slučajni Gaussian blur
        if np.random.random() < 0.3:
            ksize = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, 1), 0)

        # Slučajni šum
        if np.random.random() < 0.2:
            noise = np.random.normal(0, 5, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return img


def collate_fn(batch):
    """
    Custom collate za CTC loss:
    - Slike su već iste veličine (paddovane)
    - Labele su različitih dužina -> čuvamo kao 1D tensor + lengths
    CTC loss ne prima listu labela razlicitih duzina nego jedan veliki 1D tensor svih labela
    zalepljenih jedna za drugu plus listu duzina da zna gde koja pocinje i zavrsava.
    Npr. labele ["AB", "CDE"] -> [2, 3, 4, 5, 6] + duzine [2, 3]
    """
    imgs, labels, label_strs = zip(*batch)

    imgs = torch.stack(imgs, 0)  # [B, 1, H, W]

    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_concat = torch.cat(labels)  # CTC očekuje 1D concatenated

    return imgs, labels_concat, label_lengths, label_strs