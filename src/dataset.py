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
    "АБВГДЂЕЖЗИЈКЛЉМНЊОПРСЋТУФХЦЧЏШ"
    "абвгдђежзијклљмнњопрстћуфхцчџш"
)
BLANK = 0
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1


def encode_label(text):
    return [CHAR_TO_IDX[ch] for ch in text if ch in CHAR_TO_IDX]


def decode_prediction(log_probs, beam_width=1):
    """CTC decode. beam_width=1 je greedy; beam_width>1 je beam search."""
    if beam_width <= 1:
        indices = np.argmax(log_probs, axis=1)
        chars = []
        prev_idx = -1
        for idx in indices:
            if idx != 0 and idx != prev_idx:
                if idx in IDX_TO_CHAR:
                    chars.append(IDX_TO_CHAR[idx])
            prev_idx = idx
        return ''.join(chars)

    T, C = log_probs.shape
    NEG_INF = float('-inf')

    def log_add(a, b):
        if a == NEG_INF: return b
        if b == NEG_INF: return a
        m = max(a, b)
        return m + np.log1p(np.exp(min(a, b) - m))

    beams = {'': [0.0, NEG_INF]}

    for t in range(T):
        new_beams = {}
        lp = log_probs[t]

        for prefix, (pb, pnb) in beams.items():
            total = log_add(pb, pnb)
            last_char = prefix[-1] if prefix else None

            if prefix not in new_beams:
                new_beams[prefix] = [NEG_INF, NEG_INF]
            new_beams[prefix][0] = log_add(new_beams[prefix][0], total + lp[BLANK])

            for c in range(1, C):
                char = IDX_TO_CHAR.get(c)
                if char is None:
                    continue
                if char == last_char:
                    new_prefix = prefix + char
                    if new_prefix not in new_beams:
                        new_beams[new_prefix] = [NEG_INF, NEG_INF]
                    new_beams[new_prefix][1] = log_add(new_beams[new_prefix][1], pb + lp[c])
                    new_beams[prefix][1] = log_add(new_beams[prefix][1], pnb + lp[c])
                else:
                    new_prefix = prefix + char
                    if new_prefix not in new_beams:
                        new_beams[new_prefix] = [NEG_INF, NEG_INF]
                    new_beams[new_prefix][1] = log_add(new_beams[new_prefix][1], total + lp[c])

        beams = dict(
            sorted(new_beams.items(),
                   key=lambda x: log_add(x[1][0], x[1][1]),
                   reverse=True)[:beam_width]
        )

    if not beams:
        return ''
    return max(beams.items(), key=lambda x: log_add(x[1][0], x[1][1]))[0]


def preprocess_image(img, target_height=48):
    """Preprocess za predict.py — vraća (processed, valid_width)."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    if h == 0 or w == 0:
        return None, 0
    new_w = max(int(target_height * w / h), 1)
    img = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    return img, new_w


def augment_image(img):
    """Augmentacija tokom treninga na pravim slikama."""
    h, w = img.shape[:2]

    if random.random() < 0.4:
        angle = random.uniform(-3, 3)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if random.random() < 0.35:
        margin = int(min(h, w) * 0.05)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [random.randint(0, margin), random.randint(0, margin)],
            [w - random.randint(0, margin), random.randint(0, margin)],
            [w - random.randint(0, margin), h - random.randint(0, margin)],
            [random.randint(0, margin), h - random.randint(0, margin)],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if random.random() < 0.55:
        alpha = random.uniform(0.6, 1.4)
        beta = random.uniform(-40, 40)
        img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    if random.random() < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    if random.random() < 0.4:
        sigma = random.uniform(5, 18)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.25:
        kernel = np.ones((2, 2), np.uint8)
        if random.random() < 0.5:
            img = cv2.erode(img, kernel, iterations=1)
        else:
            img = cv2.dilate(img, kernel, iterations=1)

    if random.random() < 0.25:
        quality = random.randint(45, 85)
        _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        decoded = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if decoded is not None:
            img = decoded

    return img


class OCRDataset(Dataset):
    def __init__(self, csv_path, base_dir, img_height=48, min_height=15, augment=False):
        self.df = pd.read_csv(csv_path)
        self.base_dir = Path(base_dir)
        self.img_height = img_height
        self.min_height = min_height
        self.augment = augment

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
    batch = [(img, lbl, s) for img, lbl, s in batch
             if img is not None and lbl is not None and len(lbl) > 0]
    if len(batch) == 0:
        return None, None, None, None

    images, labels, label_strs = zip(*batch)

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
