"""
Podela dataseta na train/val/test PO RAČUNU (racun_id).
"""

import pandas as pd
import random
from pathlib import Path

# ── Automatska detekcija root foldera ─────────────────────────────────────────
# Ova skripta je u src/, dakle root je folder iznad
ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = ROOT / "dataset.csv"
OUTPUT_DIR = ROOT / "splits"
SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10

random.seed(SEED)


def get_group(racun_id):
    """Izvuče naziv grupe iz racun_id (npr. 'grejanje10' -> 'grejanje')."""
    return ''.join(c for c in racun_id if not c.isdigit())


def main():
    print(f"Root projekta: {ROOT}")
    print(f"CSV: {CSV_PATH}")
    print(f"Output: {OUTPUT_DIR}\n")

    df = pd.read_csv(CSV_PATH)
    print(f"Ukupno cropova: {len(df)}")
    print(f"Jedinstvenih računa: {df['racun_id'].nunique()}")

    df['grupa'] = df['racun_id'].apply(get_group)

    print("\nBroj računa po grupi:")
    for grupa, group_df in df.groupby('grupa'):
        n_racuna = group_df['racun_id'].nunique()
        n_cropova = len(group_df)
        print(f"  {grupa:25s}: {n_racuna:3d} računa, {n_cropova:4d} cropova")

    train_ids, val_ids, test_ids = [], [], []

    print("\n═══ PODELA PO RAČUNIMA (unutar svake grupe) ═══")
    for grupa, group_df in df.groupby('grupa'):
        racuni = list(group_df['racun_id'].unique())
        random.shuffle(racuni)

        n = len(racuni)
        n_train = max(1, int(TRAIN_RATIO * n))
        n_val = max(1, int(VAL_RATIO * n)) if n >= 10 else 0
        n_test = n - n_train - n_val

        if n_test < 1 and n >= 3:
            n_train -= 1
            n_test = 1

        train_ids += racuni[:n_train]
        val_ids   += racuni[n_train:n_train + n_val]
        test_ids  += racuni[n_train + n_val:]

        print(f"  {grupa:25s}: {n_train:3d} train | {n_val:2d} val | {n_test:2d} test")

    train_df = df[df['racun_id'].isin(train_ids)].drop(columns=['grupa'])
    val_df   = df[df['racun_id'].isin(val_ids)].drop(columns=['grupa'])
    test_df  = df[df['racun_id'].isin(test_ids)].drop(columns=['grupa'])

    assert not (set(train_ids) & set(val_ids))
    assert not (set(train_ids) & set(test_ids))
    assert not (set(val_ids) & set(test_ids))

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print(f"\n═══ KONAČNA RASPODELA (cropovi) ═══")
    print(f"  TRAIN: {len(train_df):5d} ({100*len(train_df)/len(df):.1f}%)")
    print(f"  VAL:   {len(val_df):5d} ({100*len(val_df)/len(df):.1f}%)")
    print(f"  TEST:  {len(test_df):5d} ({100*len(test_df)/len(df):.1f}%)")
    print(f"\n✅ Sačuvano u {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()