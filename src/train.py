"""
Trening skripta za CRNN OCR model.
Pokretanje: python train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import zipfile
import os
import sys

# Provera okruženja
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    BASE_DIR = Path("/content/is_projekat")
else:
    BASE_DIR = Path(__file__).resolve().parent.parent

# Dodavanje src foldera u sistem putanja radi uvoza modula
sys.path.insert(0, str(BASE_DIR / "src"))

from model import CRNN
from dataset import OCRDataset, collate_fn, NUM_CLASSES, decode_prediction
from test_validation_train_split import analiziraj_i_podeli_po_korenu

# ── Konfiguracija ─────────────────────────────────────────────────────────────
CONFIG = {
    "csv_path":       str(BASE_DIR / "dataset.csv"),
    "base_dir":       str(BASE_DIR),
    "output_dir":     str(BASE_DIR / "checkpoints"),

    "img_height":     48,
    "img_width":      768,

    "hidden_size":    256,
    "num_lstm_layers": 2,

    "batch_size":     16,
    "num_epochs":     50,
    "learning_rate":  5e-4,   # Smanjen LR (bezbedniji i stabilniji za CTC)
    "weight_decay":   0,      # Isključen na početku da model lakše nauči strukturu fontova

    "min_height":     15,     # Filtriraj presitne isečke
    "patience":       8,      # Early stopping
    "save_every":     5,      # Periodični checkpoint na svakih N epoha
}

Path(CONFIG["output_dir"]).mkdir(exist_ok=True, parents=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Uređaj za trening: {device}")


def colab_save_zip(label="crnn"):
    """Zipuje checkpoints folder i automatski pokreće preuzimanje (samo na Colabu)."""
    if not IN_COLAB:
        return
        
    try:
        from google.colab import files as colab_files
        output_dir = CONFIG["output_dir"]
        zip_path = f"{label}_checkpoints.zip"
        
        with zipfile.ZipFile(zip_path, "w") as z:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    z.write(full_path, os.path.relpath(full_path, output_dir))
                    
        print(f"\nPakovanje završeno. Pokrećem preuzimanje fajla {zip_path}...")
        colab_files.download(zip_path)
    except Exception as e:
        print(f"Greška prilikom automatskog preuzimanja: {e}")


def cer(pred, target):
    """Character Error Rate (Levenshtein rastojanje)."""
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    d = np.zeros((len(pred) + 1, len(target) + 1))
    for i in range(len(pred) + 1):
        d[i][0] = i
    for j in range(len(target) + 1):
        d[0][j] = j
    for i in range(1, len(pred) + 1):
        for j in range(1, len(target) + 1):
            cost = 0 if pred[i-1] == target[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return d[len(pred)][len(target)] / len(target)


def evaluate(model, loader, ctc_loss):
    model.eval()
    total_loss = 0
    total_cer = 0
    n = 0

    with torch.no_grad():
        for imgs, labels, label_lengths, label_strs in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)

            logits = model(imgs)  # [T, B, C]
            T, B, C = logits.shape
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)

            loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)
            total_loss += loss.item()

            # Evaluacija kroz dekodiranje predikcija
            log_probs_np = log_probs.permute(1, 0, 2).cpu().numpy()
            for i in range(B):
                pred = decode_prediction(log_probs_np[i])
                total_cer += cer(pred, label_strs[i])
                n += 1

    return total_loss / len(loader), total_cer / max(n, 1)


def train():
    # ── 1. Podela podataka ────────────────────────────────────────────────────
    train_ds_df, val_ds_df, test_ds_df = analiziraj_i_podeli_po_korenu(CONFIG["csv_path"])

    # Snimanje u privremene CSV fajlove
    train_ds_df.to_csv("train_tmp.csv", index=False)
    val_ds_df.to_csv("val_tmp.csv", index=False)
    test_ds_df.to_csv("test_tmp.csv", index=False)

    # ── 2. PyTorch Dataset Objekti ────────────────────────────────────────────
    train_ds = OCRDataset("train_tmp.csv", CONFIG["base_dir"], min_height=CONFIG["min_height"], augment=True)
    val_ds = OCRDataset("val_tmp.csv", CONFIG["base_dir"], min_height=CONFIG["min_height"], augment=False)
    test_ds = OCRDataset("test_tmp.csv", CONFIG["base_dir"], min_height=CONFIG["min_height"], augment=False)

    # ── 3. DataLoader-i ───────────────────────────────────────────────────────
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=2)

    print(f"Podaci uspešno podeljeni -> Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── 4. Inicijalizacija Modela i Optimizatora ─────────────────────────────
    model = CRNN(
        num_classes=NUM_CLASSES,
        img_height=CONFIG["img_height"],
        hidden_size=CONFIG["hidden_size"],
        num_lstm_layers=CONFIG["num_lstm_layers"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Ukupno trenirajući parametri modela: {total_params:,}")

    # zero_infinity rešava beskonačne loss vrednosti zamenom sa 0
    ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # ── 5. Glavna Trening Petlja ──────────────────────────────────────────────
    best_val_loss = float('inf')
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_cer": []}

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        model.train()
        train_loss = 0

        for imgs, labels, label_lengths, _ in tqdm(train_loader, desc=f"Epoha {epoch}"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)

            optimizer.zero_grad()
            logits = model(imgs)  # [T, B, C]
            T, B, C = logits.shape
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)

            loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)

            # Ako loss uprkos svemu vrati grešku, preskoči eksploziju težina
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            
            # Stroži gradient clipping (maksimalna norma 1.0 stabilizuje LSTM slojeve)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, val_cer = evaluate(model, val_loader, ctc_loss)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_cer"].append(val_cer)

        print(f"Epoha {epoch:3d} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val CER: {val_cer:.4f}")

        # Periodično čuvanje checkpoint-a
        if epoch % CONFIG["save_every"] == 0:
            ckpt_path = Path(CONFIG["output_dir"]) / f"checkpoint_epoch{epoch}.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, ckpt_path)

        # Čuvanje najboljeg modela (Early Stopping provera)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Path(CONFIG["output_dir"]) / "best_model.pt")
            print(f"  → Novi najbolji model sačuvan (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"Rano zaustavljanje (Early stopping) aktivirano na epohi {epoch}.")
                break

    # ── 6. Finalna Test Evaluacija ────────────────────────────────────────────
    print("\n── Pokrećem Test Evaluaciju Sa Najboljim Modelom ──────────────────")
    model.load_state_dict(torch.load(Path(CONFIG["output_dir"]) / "best_model.pt"))
    test_loss, test_cer = evaluate(model, test_loader, ctc_loss)
    print(f"Finalni rezultati -> Test loss: {test_loss:.4f} | Test CER: {test_cer:.4f}")

    # Ispis uzoraka predikcija na samom kraju treninga
    print("\n── Nasumični primeri predikcija (Vizuelna provera) ────────────────")
    model.eval()
    with torch.no_grad():
        for imgs, labels, label_lengths, label_strs in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)
            log_probs_np = log_probs.permute(1, 0, 2).cpu().numpy()
            for i in range(min(5, len(label_strs))):
                pred = decode_prediction(log_probs_np[i])
                print(f"  Tačno tekst: {label_strs[i]}")
                print(f"  Predviđeno:  {pred}")
                print("-" * 30)
            break

    # Snimanje istorije treninga u JSON format
    with open(Path(CONFIG["output_dir"]) / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("Trening uspešno okončan!")
    colab_save_zip("ocr_model")


if __name__ == "__main__":
    train()