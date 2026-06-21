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
    "БВГДЂЖЗИЛЉНЊПРСЋУФХЦЧЏШ"   # ćirilica velika
    "бвгдђжзиклљмњпрстћуфхцчџш"  # ćirilica mala
)
BLANK = 0  # CTC blank token
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 je blank
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # +1 za blank

IMG_HEIGHT = 48
IMG_WIDTH = 768
CNN_REDUCTION_FACTOR = 4


def encode_label(text):
    """Pretvori string u listu indeksa karaktera."""
    indices = []
    for c in text:
        if c in CHAR_TO_IDX:
            indices.append(CHAR_TO_IDX[c])
    return indices


def decode_prediction(logits):
    """
    CTC greedy dekodiranje.
    logits: [T, num_classes] numpy array ili tensor
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()

    pred_indices = np.argmax(logits, axis=1)  # [T]

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
    Preprocess slike za CRNN.
    Vraća (processed_image, valid_width_before_padding).
    """
    if len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr

    h, w = gray.shape
    if h == 0 or w == 0:
        return None, 0

    new_w = int(w * target_height / h)
    new_w = max(1, min(new_w, max_width))
    resized = cv2.resize(gray, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

    if new_w < max_width:
        pad = np.full((target_height, max_width), 255, dtype=np.uint8)
        pad[:, :new_w] = resized
        result = pad
    else:
        result = resized

    result = result.astype(np.float32) / 255.0

    return result, new_w


class OCRDataset(Dataset):
    def __init__(self, csv_path, base_dir, min_height=15, augment=False):
        self.base_dir = Path(base_dir)
        self.augment = augment

        df = pd.read_csv(csv_path)
        df = df[df['label'].notna()]
        df = df[df['label'].str.strip() != ""]

        valid_rows = []
        for _, row in df.iterrows():
            img_path = self.base_dir / row['image_path']
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
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
            img_tensor = torch.zeros((1, IMG_HEIGHT, IMG_WIDTH), dtype=torch.float32)
            label_encoded = torch.tensor([1], dtype=torch.long)
            valid_steps = IMG_WIDTH // CNN_REDUCTION_FACTOR
            return img_tensor, label_encoded, label_str, valid_steps

        if self.augment:
            img = self._augment(img)

        processed, valid_width = preprocess_image(img)
        if processed is None:
            processed = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
            valid_width = IMG_WIDTH

        img_tensor = torch.tensor(processed).unsqueeze(0)  # [1, H, W]
        label_encoded = encode_label(label_str)

        if len(label_encoded) == 0:
            label_encoded = [1]

        valid_steps = max(1, valid_width // CNN_REDUCTION_FACTOR)

        return img_tensor, torch.tensor(label_encoded, dtype=torch.long), label_str, valid_steps

    def _augment(self, img):
        """
        Augmentacija realnih cropova. Radi na originalnoj slici (BGR/uint8)
        PRE preprocesiranja. Geometrijske transformacije koriste BORDER_REPLICATE
        kako se ne bi ubacivale crne ivice koje model ne vidi na realnim slikama.
        """
        h, w = img.shape[:2]

        # ── Geometrijske transformacije (uče invarijantnost oblika) ───────────

        # Rotacija — mali ugao da tekst ostane čitljiv
        if np.random.random() < 0.4:
            angle = np.random.uniform(-3, 3)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)

        # Perspektivno izobličenje — simulira kosi ugao fotografije
        if np.random.random() < 0.3:
            margin = int(min(h, w) * 0.05)
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst = np.float32([
                [np.random.randint(0, margin + 1), np.random.randint(0, margin + 1)],
                [w - np.random.randint(0, margin + 1), np.random.randint(0, margin + 1)],
                [w - np.random.randint(0, margin + 1), h - np.random.randint(0, margin + 1)],
                [np.random.randint(0, margin + 1), h - np.random.randint(0, margin + 1)],
            ])
            M = cv2.getPerspectiveTransform(src, dst)
            img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # ── Fotometrijske transformacije (uče invarijantnost na osvetljenje) ──

        # Kontrast i osvetljenje
        if np.random.random() < 0.5:
            alpha = np.random.uniform(0.7, 1.3)
            beta = np.random.randint(-20, 20)
            img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        # Senka skenera — jedna strana tamnija (čest artefakt na realnim slikama)
        if np.random.random() < 0.3:
            intensity = np.random.randint(15, 40)
            if np.random.random() < 0.5:
                grad = np.linspace(intensity, 0, w, dtype=np.float32)
                shadow = np.tile(grad, (h, 1))
            else:
                grad = np.linspace(intensity, 0, h, dtype=np.float32)
                shadow = np.tile(grad.reshape(-1, 1), (1, w))
            if img.ndim == 3:
                shadow = shadow[:, :, None]
            img = np.clip(img.astype(np.float32) - shadow, 0, 255).astype(np.uint8)

        # ── Degradacija kvaliteta (uče robusnost na loš snimak) ───────────────

        # Horizontalno Gausovo zamućenje
        if np.random.random() < 0.3:
            ksize = int(np.random.choice([3, 5]))
            img = cv2.GaussianBlur(img, (ksize, 1), 0)

        # Erozija / dilatacija — simulira deblje ili tanje mastilo / loš print
        if np.random.random() < 0.2:
            kernel = np.ones((2, 2), np.uint8)
            if np.random.random() < 0.5:
                img = cv2.erode(img, kernel, iterations=1)
            else:
                img = cv2.dilate(img, kernel, iterations=1)

        # Gausov šum
        if np.random.random() < 0.2:
            noise = np.random.normal(0, 5, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # JPEG kompresija — artefakti niskog kvaliteta
        if np.random.random() < 0.25:
            q = int(np.random.randint(50, 90))
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if ok:
                decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR if img.ndim == 3 else cv2.IMREAD_GRAYSCALE)
                if decoded is not None:
                    img = decoded

        return img


def collate_fn(batch):
    """
    Custom collate za CTC loss.
    Vraća: imgs, labels_concat, label_lengths, input_lengths, label_strs
    Redosled je takav da su svi tenzori grupisani zajedno pre stringova.
    """
    imgs, labels, label_strs, valid_steps = zip(*batch)

    imgs = torch.stack(imgs, 0)                                          # [B, 1, H, W]
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_concat = torch.cat(labels)                                    # [ukupno_karaktera]
    input_lengths = torch.tensor(valid_steps, dtype=torch.long)         # [B]

    # Redosled: imgs, labels_concat, label_lengths, input_lengths, label_strs
    return imgs, labels_concat, label_lengths, input_lengths, label_strs