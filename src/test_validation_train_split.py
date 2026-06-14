import pandas as pd
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "dataset.csv"
OUTPUT_DIR = ROOT / "splits"
SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10

random.seed(SEED)

# Grupe koje su UVEK ćirilične
CYRILLIC_GROUPS = {"struja", "jotel"}

# Koliko računa iz ćiriličnih grupa forsiraj u val/test
FORCED_CYR_VAL = {"struja": 8, "jotel": 1}
FORCED_CYR_TEST = {"struja": 3, "jotel": 1}


def get_group(racun_id):
    return ''.join(c for c in racun_id if not c.isdigit())


def main():
    df = pd.read_csv(CSV_PATH)
    df['grupa'] = df['racun_id'].apply(get_group)

    print(f"Ukupno cropova: {len(df)}")
    print(f"Jedinstvenih računa: {df['racun_id'].nunique()}\n")

    train_ids, val_ids, test_ids = [], [], []

    for grupa, group_df in df.groupby('grupa'):
        racuni = list(group_df['racun_id'].unique())
        random.shuffle(racuni)
        n = len(racuni)

        if grupa in CYRILLIC_GROUPS:
            # Forsiraj određen broj u val i test
            n_val_forced = FORCED_CYR_VAL.get(grupa, 0)
            n_test_forced = FORCED_CYR_TEST.get(grupa, 0)

            # Osiguraj da ima dovoljno računa
            n_val_forced = min(n_val_forced, max(0, n - 2))
            n_test_forced = min(n_test_forced, max(0, n - n_val_forced - 1))

            val_pick = racuni[:n_val_forced]
            test_pick = racuni[n_val_forced:n_val_forced + n_test_forced]
            train_pick = racuni[n_val_forced + n_test_forced:]

        else:
            # Standardna raspodela za ostale grupe
            n_train = max(1, int(TRAIN_RATIO * n))
            n_val = max(1, int(VAL_RATIO * n)) if n >= 10 else 0
            n_test = n - n_train - n_val

            if n_test < 1 and n >= 3:
                n_train -= 1
                n_test = 1

            train_pick = racuni[:n_train]
            val_pick = racuni[n_train:n_train + n_val]
            test_pick = racuni[n_train + n_val:]

        train_ids += train_pick
        val_ids += val_pick
        test_ids += test_pick

        print(f"  {grupa:25s}: {len(train_pick):3d} train | {len(val_pick):2d} val | {len(test_pick):2d} test")

    # Proveri preklapanja
    assert not (set(train_ids) & set(val_ids))
    assert not (set(train_ids) & set(test_ids))
    assert not (set(val_ids) & set(test_ids))

    train_df = df[df['racun_id'].isin(train_ids)].drop(columns=['grupa'])
    val_df = df[df['racun_id'].isin(val_ids)].drop(columns=['grupa'])
    test_df = df[df['racun_id'].isin(test_ids)].drop(columns=['grupa'])

    # Izračunaj % ćiriličnih cropova u val i test
    val_cyr = val_df[val_df['racun_id'].apply(get_group).isin(CYRILLIC_GROUPS)]
    test_cyr = test_df[test_df['racun_id'].apply(get_group).isin(CYRILLIC_GROUPS)]

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print(f"\nKONAČNA RASPODELA (cropovi)")
    print(f"  TRAIN: {len(train_df):5d} ({100*len(train_df)/len(df):.1f}%)")
    print(f"  VAL:   {len(val_df):5d} ({100*len(val_df)/len(df):.1f}%)")
    print(f"    → ćirilični: {len(val_cyr):3d} ({100*len(val_cyr)/len(val_df):.1f}%)")
    print(f"  TEST:  {len(test_df):5d} ({100*len(test_df)/len(df):.1f}%)")
    print(f"    → ćirilični: {len(test_cyr):3d} ({100*len(test_cyr)/len(test_df):.1f}%)")
    print(f"\nSačuvano u {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()