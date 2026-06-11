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

sys.path.insert(0, str(BASE_DIR / "src"))

from model import CRNN
from dataset import OCRDataset, collate_fn, NUM_CLASSES, decode_prediction
from test_validation_train_split import analiziraj_i_podeli_po_korenu

# ── Konfiguracija ─────────────────────────────────────────────────────────────
CONFIG = {
    "csv_path":        str(BASE_DIR / "dataset.csv"),
    "base_dir":        str(BASE_DIR),
    "output_dir":      str(BASE_DIR / "checkpoints"),

    "img_height":      48,

    "hidden_size":     256,
    "num_lstm_layers": 2,

    "batch_size":      8,           # ← SMANJENO (zbog varijabilne širine, veći batch = više memorije)
    "num_epochs":      150,
    "learning_rate":   1e-4,
    "weight_decay":    1e-4,

    "min_height":      15,
    "patience":        15,
    "save_every":      5,
    "num_preview":     5,
    "warmup_epochs":   5,
    "grad_clip":       5.0,
}

Path(CONFIG["output_dir"]).mkdir(exist_ok=True, parents=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Uređaj za trening: {device}")


# ── Pomoćne funkcije ─────────────────────────────────────────────────────────

def colab_save_zip(label="crnn"):
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
            cost = 0 if pred[i - 1] == target[j - 1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return d[len(pred)][len(target)] / max(len(target), 1)


def get_actual_input_lengths(model, images):
    """
    Probuši slike kroz CNN deo modela da izmeriš TAČNU
    temporalnu dimenziju T posle konvolucija.
    Vraća input_lengths za svaki uzorak u batch-u.
    """
    with torch.no_grad():
        # Probuši kroz CNN slojeve
        x = images
        for layer in model.cnn:
            x = layer(x)
        # x shape: [B, C, H, W_after_cnn]
        # W_after_cnn je T (temporalna dimenzija)
        T = x.shape[3]  # širina posle CNN-a
        
        # Za svaki uzorak, input_length = T
        # (CTC koristi blank za padding regione)
        B = images.shape[0]
        input_lengths = torch.full((B,), T, dtype=torch.long)
    
    return input_lengths, T


def diagnostika(model, loader):
    """Ispisuje T i upoređuje sa dužinama teksta."""
    model.eval()
    max_target = 0
    all_targets = []

    with torch.no_grad():
        for batch_data in loader:
            if batch_data[0] is None:
                continue
            imgs, labels_concat, label_lengths, input_lengths, label_strs = batch_data
            imgs = imgs.to(device)

            # Izračunaj pravi T
            x = imgs
            for layer in model.cnn:
                x = layer(x)
            T = x.shape[3]

            for s in label_strs:
                all_targets.append(len(s))
            break

    if not all_targets:
        print("Nema podataka za dijagnostiku!")
        return 0, 0

    max_len = max(all_targets)
    avg_len = np.mean(all_targets)

    print("\n" + "=" * 60)
    print("  DIAGNOSTIKA: T vs dužina teksta")
    print("=" * 60)
    print(f"  T (posle CNN-a)               : {T}")
    print(f"  Max dužina ciljnog teksta      : {max_len}")
    print(f"  Prosečna dužina teksta         : {avg_len:.1f}")
    print(f"  T / max_len ratio              : {T / max_len:.2f}")

    if T < max_len:
        print(f"\n  ⚠️  PROBLEM: T ({T}) < max_target_len ({max_len})")
        print(f"  ⚠️  CTC ne može da proizvede dovoljno dug izlaz!")
    else:
        print(f"\n  ✓ OK: T ({T}) >= max_target_len ({max_len})")
    print("=" * 60 + "\n")
    return T, max_len


# ── Evaluacija ────────────────────────────────────────────────────────────────

def evaluate(model, loader, ctc_loss):
    model.eval()
    total_loss = 0.0
    total_cer = 0.0
    n = 0

    with torch.no_grad():
        for batch_data in loader:
            if batch_data[0] is None:
                continue
            imgs, labels_concat, label_lengths, _, label_strs = batch_data
            imgs          = imgs.to(device)
            labels_concat = labels_concat.to(device)
            label_lengths = label_lengths.to(device)

            logits = model(imgs)                                  # [T, B, C]
            T, B, C = logits.shape
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            # TAČNE input_lengths iz CNN-a
            input_lengths, _ = get_actual_input_lengths(model, imgs)
            input_lengths = input_lengths.to(device)
            input_lengths = torch.clamp(input_lengths, max=T)

            loss = ctc_loss(log_probs, labels_concat, input_lengths, label_lengths)
            total_loss += loss.item()

            log_probs_np = log_probs.permute(1, 0, 2).cpu().numpy()
            for i in range(B):
                pred = decode_prediction(log_probs_np[i])
                total_cer += cer(pred, label_strs[i])
                n += 1

    return total_loss / max(len(loader), 1), total_cer / max(n, 1)


def preview_predictions(model, loader, num_samples=5):
    model.eval()
    printed = 0

    with torch.no_grad():
        for batch_data in loader:
            if batch_data[0] is None:
                continue
            imgs, _, _, _, label_strs = batch_data
            imgs = imgs.to(device)
            logits = model(imgs)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)
            log_probs_np = log_probs.permute(1, 0, 2).cpu().numpy()

            for i in range(len(label_strs)):
                if printed >= num_samples:
                    return
                pred = decode_prediction(log_probs_np[i])
                tačno = label_strs[i]
                predviđeno = pred
                pogodak = "✓" if tačno == predviđeno else "✗"
                print(f"  [{pogodak}] Tačan tekst : {tačno}")
                print(f"      Predviđeno  : {predviđeno}")
                print(f"      CER         : {cer(predviđeno, tačno):.4f}")
                print(f"      {'-' * 40}")
                printed += 1


# ── Trening ───────────────────────────────────────────────────────────────────

def train():
    # ── 1. Podela podataka ────────────────────────────────────────────────────
    train_ds_df, val_ds_df, test_ds_df = analiziraj_i_podeli_po_korenu(CONFIG["csv_path"])

    train_ds_df.to_csv("train_tmp.csv", index=False)
    val_ds_df.to_csv("val_tmp.csv",   index=False)
    test_ds_df.to_csv("test_tmp.csv", index=False)

    # ── 2. Dataset objekti ────────────────────────────────────────────────────
    train_ds = OCRDataset("train_tmp.csv", CONFIG["base_dir"], img_height=CONFIG["img_height"],
                          min_height=CONFIG["min_height"], augment=True)
    val_ds   = OCRDataset("val_tmp.csv",   CONFIG["base_dir"], img_height=CONFIG["img_height"],
                          min_height=CONFIG["min_height"], augment=False)
    test_ds  = OCRDataset("test_tmp.csv",  CONFIG["base_dir"], img_height=CONFIG["img_height"],
                          min_height=CONFIG["min_height"], augment=False)

    # ── 3. DataLoaderi ────────────────────────────────────────────────────────
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False,
                              collate_fn=collate_fn, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"], shuffle=False,
                              collate_fn=collate_fn, num_workers=2)

    print(f"Podaci uspešno podeljeni -> Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── 4. Model ──────────────────────────────────────────────────────────────
    model = CRNN(
        num_classes=NUM_CLASSES,
        img_height=CONFIG["img_height"],
        hidden_size=CONFIG["hidden_size"],
        num_lstm_layers=CONFIG["num_lstm_layers"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Ukupno trenirajući parametri modela: {total_params:,}")

    # ── 4b. Dijagnostika ─────────────────────────────────────────────────────
    T_val, max_target = diagnostika(model, val_loader)

    if T_val < max_target:
        print("\n🚨 KRITIČAN PROBLEM: T < max_target_len!")
        print("   Proverite CNN arhitekturu u model.py")
        print("   Uverite se da NE koristite pooling po širini!\n")

    ctc_loss  = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"],
                           weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # ── 5. Trening petlja ─────────────────────────────────────────────────────
    best_val_loss    = float('inf')
    best_val_cer     = float('inf')
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_cer": [], "lr": []}

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        # Warmup
        if epoch <= CONFIG["warmup_epochs"]:
            warmup_lr = CONFIG["learning_rate"] * epoch / CONFIG["warmup_epochs"]
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr
            print(f"  [Warmup] LR = {warmup_lr:.6f}")

        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch_data in tqdm(train_loader, desc=f"Epoha {epoch}"):
            if batch_data[0] is None:
                continue
            imgs, labels_concat, label_lengths, _, _ = batch_data
            imgs          = imgs.to(device)
            labels_concat = labels_concat.to(device)
            label_lengths = label_lengths.to(device)

            optimizer.zero_grad()
            logits = model(imgs)                              # [T, B, C]
            T, B, C = logits.shape
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            # TAČNE input_lengths
            input_lengths, _ = get_actual_input_lengths(model, imgs)
            input_lengths = input_lengths.to(device)
            input_lengths = torch.clamp(input_lengths, max=T)

            loss = ctc_loss(log_probs, labels_concat, input_lengths, label_lengths)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  ⚠️ NaN/Inf loss preskočen!")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["grad_clip"])
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1

        train_loss /= max(num_batches, 1)
        val_loss, val_cer_score = evaluate(model, val_loader, ctc_loss)

        if epoch > CONFIG["warmup_epochs"]:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_cer"].append(val_cer_score)
        history["lr"].append(current_lr)

        print(f"Epoha {epoch:3d} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | "
              f"Val CER: {val_cer_score:.4f} | LR: {current_lr:.6f}")

        # Vizuelna provera
        print(f"\n── Primeri predikcija (epoha {epoch}) ────────────────────────")
        preview_predictions(model, val_loader, num_samples=CONFIG["num_preview"])
        print()

        # Čuvanje
        if epoch % CONFIG["save_every"] == 0:
            ckpt_path = Path(CONFIG["output_dir"]) / f"checkpoint_epoch{epoch}.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "val_loss": val_loss, "val_cer": val_cer_score}, ckpt_path)

        if val_cer_score < best_val_cer:
            best_val_cer     = val_cer_score
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "val_loss": val_loss, "val_cer": val_cer_score},
                       Path(CONFIG["output_dir"]) / "best_model.pt")
            print(f"  → Novi najbolji model sačuvan (CER: {best_val_cer:.4f})")
        else:
            patience_counter += 1
            print(f"  → Bez poboljšanja ({patience_counter}/{CONFIG['patience']})")
            if patience_counter >= CONFIG["patience"]:
                print(f"Early stopping na epohi {epoch}.")
                break

    # ── 6. Finalna evaluacija ─────────────────────────────────────────────────
    print("\n── Test Evaluacija ──────────────────────────────────────────────")
    checkpoint = torch.load(Path(CONFIG["output_dir"]) / "best_model.pt")
    model.load_state_dict(checkpoint["model"])
    print(f"Najbolji model iz epohe {checkpoint['epoch']}")

    test_loss, test_cer_score = evaluate(model, test_loader, ctc_loss)
    print(f"Test loss: {test_loss:.4f} | Test CER: {test_cer_score:.4f}")

    print("\n── Primeri na TEST setu ─────────────────────────────────────────")
    preview_predictions(model, test_loader, num_samples=10)

    with open(Path(CONFIG["output_dir"]) / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\nTrening završen!")
    colab_save_zip("ocr_model")


if __name__ == "__main__":
    train()