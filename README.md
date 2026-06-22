# OCR za komunalne račune

Sistem za automatsko prepoznavanje teksta (OCR) sa računa za struju, grejanje, MTS, smeće i ostale komunalne usluge. Sastoji se iz dva dela YOLO za detekciju dela teksta odakle se čita i CRNN arhitekture i CTC loss funkcije.

## Arhitektura

```
Slika računa → YOLO detekcija → Crop regiona → CRNN OCR → Tekst
```

- **YOLO** — detektuje i klasifikuje regione (iznos, zaglavlje, uplatnica...) na slici računa
- **CRNN** — konvoluciona + rekurentna mreža sa CTC gubitkom za prepoznavanje teksta u isečcima

Podržava srpsku ćirilicu, latinicu i mešoviti tekst (156 klasa karaktera). 

## Struktura projekta

```
ocr/
├── src/
│   ├── model.py              # CRNN arhitektura
│   ├── dataset.py            # Dataset, preprocessing, augmentacija
│   ├── train.py              # Trening CRNN modela
│   ├── predict.py            # Predikcija na jednoj slici
│   ├── generate_synthetic.py # Generisanje sintetičkih podataka
│   ├── merge_datasets.py     # Spajanje realnih i sintetičkih podataka
│   ├── train_yolo.py         # Trening YOLO modela
│   └── predict_yolo.py       # Predikcija YOLO modelom
├── splits/                   # Train/val/test podela (CSV)
├── yolo_dataset/             # YOLO dataset (anotacije + data.yaml)
├── synthetic/                # Sintetički generisane slike i CSV
└── checkpoints/              # Čuvanje modela tokom treninga
```

## Instalacija

```bash
git clone https://github.com/milanatosic/is_projekat
cd is_projekat
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python pillow pandas tqdm ultralytics
```

## Pokretanje

### 1. Generisanje sintetičkih podataka

```bash
python src/generate_synthetic.py
```

Generiše slike u `synthetic/` i CSV fajl `synthetic/synthetic_dataset.csv`.

### 2. Spajanje dataseta

```bash
python src/merge_datasets.py
```

Spaja realne cropove i sintetičke slike u `splits/train_combined.csv`.

### 3. Trening CRNN modela

**Lokalno:**
```bash
python src/train.py
```

**Google Colab**:

```python
# Na početku Colab sveske:
git clone https://github.com/milanatosic/is_projekat /content/is_projekat
%cd /content/is_projekat
pip install opencv-python-headless pillow pandas tqdm

from google.colab import drive
drive.mount('/content/drive')

python src/train.py
```

Model se čuva u `checkpoints/best_ocr_model.pt`, a backup ide na Google Drive.

### 4. Predikcija na jednoj slici

```bash
python src/predict.py --img putanja/do/slike.jpg
```

### 5. Trening YOLO modela

```bash
python src/train_yolo.py
```

Dataset mora biti preuzet sa Roboflow-a i smešten u `yolo_dataset/` (slike u `yolo_dataset/train/images/`).

## Rezultati

| Metrika | Vrednost |
|---|---|
| CER (Character Error Rate) 
| WER (Word Error Rate) 
| Tačno prepoznatih 

Evaluacija na test skupu od 183 slike, grupisane po tipu računa (struja, MTS, grejanje, smeće, održavanje zgrade, jotel, uplatnica).
