"""
Trening skripta za CRNN OCR model.
Pokretanje:
    python train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import zipfile
import os

from model import CRNN
from dataset import OCRDataset, collate_fn, NUM_CLASSES, decode_prediction
from test_validation_train_split import analiziraj_i_podeli_po_korenu
import sys

# Zamenite trenutnu liniju za BASE_DIR i CONFIG sa ovim:
import sys
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    # Ako smo na Colabu, znamo tačnu i fiksnu putanju do projekta
    BASE_DIR = Path("/content/is_projekat")
else:
    # Ako smo lokalno, uzimamo folder iznad src-a
    BASE_DIR = Path(__file__).resolve().parent.parent

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
    "num_epochs":     150,
    "learning_rate":  1e-3,
    "weight_decay":   1e-4,

    "train_ratio":    0.8,
    "val_ratio":      0.1,
    # ostatak je test (0.1)

    "min_height":     15,   # filtriraj slike ispod ove visine
    "patience":       8,    # early stopping
    "save_every":     5,    # čuvaj checkpoint svakih N epoha
}

Path(CONFIG["output_dir"]).mkdir(exist_ok=True, parents=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Uređaj: {device}")

def colab_save_zip(label="crnn"):
    """Zipuje checkpoints folder i preuzima ga na lokalni računar (samo na Colabu)."""
    # Proveravamo da li smo na Colabu (definišemo IN_COLAB preko sys.modules)
    import sys
    IN_COLAB = "google.colab" in sys.modules
    
    if not IN_COLAB:
        return
        
    try:
        from google.colab import files as colab_files
        output_dir = CONFIG["output_dir"]
        zip_path = f"{label}_checkpoints.zip"
        
        # Spakuj ceo checkpoints folder u zip
        with zipfile.ZipFile(zip_path, "w") as z:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    # Čuvamo fajl u zipu sa relativnom putanjom
                    z.write(full_path, os.path.relpath(full_path, output_dir))
                    
        print(f"\nPakovanje završeno. Pokrećem preuzimanje fajla {zip_path}...")
        colab_files.download(zip_path)
    except Exception as e:
        print(f"Greška prilikom automatskog preuzimanja: {e}")


def cer(pred, target):
    """Character Error Rate."""
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

            # CER
            log_probs_np = log_probs.permute(1, 0, 2).cpu().numpy()
            for i in range(B):
                pred = decode_prediction(log_probs_np[i])
                total_cer += cer(pred, label_strs[i])
                n += 1

    return total_loss / len(loader), total_cer / max(n, 1)


def train():
    # ── Dataset ───────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds = analiziraj_i_podeli_po_korenu(CONFIG["csv_path"])

    # Snimamo ih u privremene CSV fajlove kako bi tvoj OCRDataset mogao da ih učita
    train_ds.to_csv("train_tmp.csv", index=False)
    val_ds.to_csv("val_tmp.csv", index=False)
    test_ds.to_csv("test_tmp.csv", index=False)

    # ── 2. Kreiranje PyTorch Dataset objekata ─────────────────────────────────
    train_ds = OCRDataset(
        "train_tmp.csv",
        CONFIG["base_dir"],
        min_height=CONFIG["min_height"],
        augment=True  # Augmentacija je UPALJENA samo za trening set
    )
    
    val_ds = OCRDataset(
        "val_tmp.csv",
        CONFIG["base_dir"],
        min_height=CONFIG["min_height"],
        augment=False # Isključena za validaciju
    )
    
    test_ds = OCRDataset(
        "test_tmp.csv",
        CONFIG["base_dir"],
        min_height=CONFIG["min_height"],
        augment=False # Isključena za test
    )

    # ── 3. DataLoader-i ───────────────────────────────────────────────────────
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"],
                            shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"],
                             shuffle=False, collate_fn=collate_fn, num_workers=2)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CRNN(
        num_classes=NUM_CLASSES,
        img_height=CONFIG["img_height"],
        hidden_size=CONFIG["hidden_size"],
        num_lstm_layers=CONFIG["num_lstm_layers"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri modela: {total_params:,}")

    ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(),
                           lr=CONFIG["learning_rate"],
                           weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # ── Trening petlja ────────────────────────────────────────────────────────
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

            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, val_cer = evaluate(model, val_loader, ctc_loss)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_cer"].append(val_cer)

        print(f"Epoha {epoch:3d} | Train loss: {train_loss:.4f} | "
              f"Val loss: {val_loss:.4f} | Val CER: {val_cer:.4f}")

        # Sačuvaj checkpoint
        if epoch % CONFIG["save_every"] == 0:
            ckpt_path = Path(CONFIG["output_dir"]) / f"checkpoint_epoch{epoch}.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()}, ckpt_path)

        # Sačuvaj best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(),
                       Path(CONFIG["output_dir"]) / "best_model.pt")
            print(f"  → Novi best model sačuvan (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"Early stopping na epohi {epoch}.")
                break

    # ── Test evaluacija ───────────────────────────────────────────────────────
    print("\n── Test evaluacija ──────────────────────────────────────────")
    model.load_state_dict(torch.load(Path(CONFIG["output_dir"]) / "best_model.pt"))
    test_loss, test_cer = evaluate(model, test_loader, ctc_loss)
    print(f"Test loss: {test_loss:.4f} | Test CER: {test_cer:.4f}")

    # Primeri predikcija
    print("\n── Primeri predikcija ───────────────────────────────────────")
    model.eval()
    with torch.no_grad():
        for imgs, labels, label_lengths, label_strs in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)
            log_probs_np = log_probs.permute(1, 0, 2).cpu().numpy()
            for i in range(min(5, len(label_strs))):
                pred = decode_prediction(log_probs_np[i])
                print(f"  Tačno:    {label_strs[i]}")
                print(f"  Predviđeno:  {pred}")
                print()
            break

    # Sačuvaj istoriju
    with open(Path(CONFIG["output_dir"]) / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("Trening završen!")


if __name__ == "__main__":
    train()