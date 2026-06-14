import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    "original_train": ROOT / "splits" / "train.csv",
    "synthetic_csv":  ROOT / "synthetic" / "synthetic_dataset.csv",
    "output_csv":     ROOT / "splits" / "train_combined.csv",
}


def main():
    df_orig = pd.read_csv(CONFIG["original_train"])
    df_synth = pd.read_csv(CONFIG["synthetic_csv"])

    # Spoji
    df_combined = pd.concat([df_orig, df_synth], ignore_index=True)
    df_combined.to_csv(CONFIG["output_csv"], index=False)
    
    print(f"Original train:  {len(df_orig)}")
    print(f"Synthetic:       {len(df_synth)}")
    print(f"Combined:        {len(df_combined)}")
    print(f"\nSačuvano: {CONFIG['output_csv']}")


if __name__ == "__main__":
    main()