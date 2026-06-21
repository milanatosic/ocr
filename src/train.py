import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import json
import time
import shutil
from pathlib import Path
from tqdm import tqdm
import sys

# ── Automatska detekcija root foldera ─────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model import CRNN
from dataset import OCRDataset, collate_fn, NUM_CLASSES, decode_prediction

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ── Konfiguracija ─────────────────────────────────────────────────────────────
CONFIG = {
    "train_csv":    ROOT / "splits" / "train_combined.csv",
    "val_csv":      ROOT / "splits" / "val.csv",
    "test_csv":     ROOT / "splits" / "test.csv",
    "base_dir":     ROOT,
    "output_dir":   ROOT / "checkpoints",
    "drive_backup_dir": Path("/content/drive/MyDrive/ocr_checkpoints"),

    "img_height":    48,
    "hidden_size":   256,
    "num_lstm_layers": 2,

    "batch_size":    32,
    "num_epochs":    200,
    "learning_rate": 3e-4,
    "weight_decay":  5e-4,         
    "grad_clip":     5.0,
    "min_height":    15,
    "patience":      40,           
    "save_every":    10,
    "use_amp":       True,
}

CONFIG["output_dir"].mkdir(exist_ok=True, parents=True)

USE_DRIVE_BACKUP = Path("/content/drive/MyDrive").exists()
if USE_DRIVE_BACKUP:
    CONFIG["drive_backup_dir"].mkdir(exist_ok=True, parents=True)
    print(f"Drive backup AKTIVAN: {CONFIG['drive_backup_dir']}")
else:
    print(f"Drive backup nedostupan (radi se lokalno)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Uređaj: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Root projekta: {ROOT}\n")


def cer(pred, target):
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    d = np.zeros((len(pred) + 1, len(target) + 1), dtype=np.float32)
    for i in range(len(pred) + 1):
        d[i, 0] = i
    for j in range(len(target) + 1):
        d[0, j] = j
    for i in range(1, len(pred) + 1):
        for j in range(1, len(target) + 1):
            cost = 0 if pred[i - 1] == target[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return d[len(pred), len(target)] / len(target)


def evaluate(model, loader, ctc_loss, return_examples=False, n_examples=3):
    """Evaluacija. Opciono vraća i primere predikcija."""
    model.eval()
    total_loss = 0
    total_cer = 0
    n = 0
    examples = []

    with torch.no_grad():
        for batch in loader:
            if batch[0] is None:
                continue
            imgs, labels, label_lengths, label_strs = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)

            logits = model(imgs)
            T, B, _ = logits.shape
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)

            loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)
            total_loss += loss.item()

            log_probs_np = log_probs.permute(1, 0, 2).cpu().numpy()
            for i in range(len(label_strs)):
                pred = decode_prediction(log_probs_np[i])
                c = cer(pred, label_strs[i])
                total_cer += c
                n += 1
                if return_examples and len(examples) < n_examples:
                    examples.append((label_strs[i], pred, c))

    avg_loss = total_loss / max(len(loader), 1)
    avg_cer = total_cer / max(n, 1)

    if return_examples:
        return avg_loss, avg_cer, examples
    return avg_loss, avg_cer


def save_to_drive(local_path, name):
    if not USE_DRIVE_BACKUP:
        return
    try:
        drive_path = CONFIG["drive_backup_dir"] / name
        shutil.copy2(local_path, drive_path)
        print(f"Backup na Drive: {drive_path}")
    except Exception as e:
        print(f"Drive backup nije uspeo: {e}")


def train():
    train_ds = OCRDataset(CONFIG["train_csv"], CONFIG["base_dir"],
                          img_height=CONFIG["img_height"],
                          min_height=CONFIG["min_height"], augment=True)
    val_ds = OCRDataset(CONFIG["val_csv"], CONFIG["base_dir"],
                        img_height=CONFIG["img_height"],
                        min_height=CONFIG["min_height"], augment=False)
    test_ds = OCRDataset(CONFIG["test_csv"], CONFIG["base_dir"],
                         img_height=CONFIG["img_height"],
                         min_height=CONFIG["min_height"], augment=False)

    print(f"\nTrain: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                            collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                             collate_fn=collate_fn, num_workers=2, pin_memory=True)

    model = CRNN(num_classes=NUM_CLASSES, img_height=CONFIG["img_height"],
                 hidden_size=CONFIG["hidden_size"],
                 num_lstm_layers=CONFIG["num_lstm_layers"]).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri modela: {total_params:,}\n")

    ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                            weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["num_epochs"], eta_min=1e-6
    )
    scaler = GradScaler('cuda', enabled=CONFIG["use_amp"])

    best_val_cer = float('inf')
    best_val_loss = float('inf')
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_cer": []}

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        model.train()
        train_loss = 0
        t0 = time.time()

        for batch in tqdm(train_loader, desc=f"Epoha {epoch}", ncols=80):
            if batch[0] is None:
                continue
            imgs, labels, label_lengths, _ = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)

            optimizer.zero_grad()

            with autocast('cuda', enabled=CONFIG["use_amp"]):
                logits = model(imgs)
                T, B, _ = logits.shape
                log_probs = torch.nn.functional.log_softmax(logits, dim=2)
                input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
                loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)

            if not torch.isnan(loss) and not torch.isinf(loss):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)
        val_loss, val_cer, val_examples = evaluate(
            model, val_loader, ctc_loss,
            return_examples=True, n_examples=3
        )
        scheduler.step()

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_cer"].append(float(val_cer))

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoha {epoch:3d} | Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f} | CER: {val_cer:.4f} | "
              f"LR: {lr_now:.6f} | {time.time()-t0:.1f}s")

        # Primeri predikcija u svakoj epohi
        print(f"  ── Primeri (epoha {epoch}) ──")
        for tacno, pred, c in val_examples:
            print(f"    Tačno : {tacno}")
            print(f"    Predv.: {pred}")
            print(f"    CER   : {c:.3f}")
            print()

        if epoch % CONFIG["save_every"] == 0:
            ckpt_path = CONFIG["output_dir"] / f"checkpoint_epoch{epoch}.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_loss": float(val_loss), "val_cer": float(val_cer)}, ckpt_path)

        if val_cer < best_val_cer:
            best_val_cer = val_cer
            best_path = CONFIG["output_dir"] / "best_ocr_model.pt"
            torch.save(model.state_dict(), best_path)
            print(f"  → NOVI BEST MODEL (val_cer: {best_val_cer:.4f})")
            save_to_drive(best_path, "best_ocr_model.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"\nEarly stopping na epohi {epoch}.")
                break

    # ── Test ──
    print(f"\n{'═'*60}\n TEST EVALUACIJA\n{'═'*60}")
    model.load_state_dict(torch.load(
        CONFIG["output_dir"] / "best_ocr_model.pt",
        map_location=device, weights_only=True
    ))
    test_loss, test_cer = evaluate(model, test_loader, ctc_loss)
    print(f"Test loss: {test_loss:.4f} | Test CER: {test_cer:.4f} ({100*test_cer:.2f}%)")

    print(f"\n── PRIMERI PREDIKCIJA NA TEST SKUPU ──")
    model.eval()
    examples = []
    with torch.no_grad():
        for batch in test_loader:
            if batch[0] is None:
                continue
            imgs, labels, label_lengths, label_strs = batch
            imgs = imgs.to(device)
            logits = model(imgs)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)
            log_probs_np = log_probs.permute(1, 0, 2).cpu().numpy()
            for i in range(len(label_strs)):
                pred = decode_prediction(log_probs_np[i])
                examples.append((label_strs[i], pred, cer(pred, label_strs[i])))

    examples.sort(key=lambda x: x[2])
    print("\n── 5 NAJBOLJIH ──")
    for tacno, pred, c in examples[:5]:
        print(f"  Tačno: {tacno}\n  Predviđeno: {pred}\n  CER: {c:.3f}\n")
    print("\n── 5 NAJGORIH ──")
    for tacno, pred, c in examples[-5:]:
        print(f"  Tačno: {tacno}\n  Predviđeno: {pred}\n  CER: {c:.3f}\n")

    # POPRAVLJENO: history sa float konverzijom
    history_path = CONFIG["output_dir"] / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    save_to_drive(history_path, "history.json")

    print(f"\nTrening završen!")
    print(f"   Lokalno: {CONFIG['output_dir']}/best_ocr_model.pt")
    if USE_DRIVE_BACKUP:
        print(f"   Drive:   {CONFIG['drive_backup_dir']}/best_ocr_model.pt")


if __name__ == "__main__":
    train()