import sys
import os
import shutil

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    GITHUB_TOKEN = "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"
    USERNAME = "milanatosic"
    REPO_NAME = "is_projekat"

    # ── BRISANJE STAROG KEDA (Ovo rešava problem sa keširanjem) ──
    if os.path.exists(REPO_NAME):
        print(f"Uklanjam stari folder {REPO_NAME} iz memorije...")
        shutil.rmtree(REPO_NAME)

    print("Kloniram najnoviju verziju sa GitHub-a...")
    os.system(f"git clone https://{GITHUB_TOKEN}@github.com/{USERNAME}/{REPO_NAME}.git > /dev/null 2>&1")

    # Umesto %cd {REPO_NAME} napiši:
    %cd /content/{REPO_NAME}

    # Čistimo sys.path i dodajemo src ponovo da budemo sigurni
    if os.path.abspath("src") not in sys.path:
        sys.path.append(os.path.abspath("src"))

    !pip install -q pandas scikit-learn tqdm opencv-python
    print("✅ Najnoviji kod je uspešno povučen i spreman!")


!python src/train.py
