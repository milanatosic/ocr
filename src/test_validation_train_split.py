import pandas as pd
from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG = str(BASE_DIR / "dataset.csv")

def analiziraj_i_podeli_po_korenu(csv_path):
    # 1. Učitavanje CSV-a
    df = pd.read_csv(csv_path)
    
    # Funkcija koja uklanja sve cifre sa kraja stringa
    def izvuci_koren(tekst):
        # r'\d+$' pronalazi sve spojeve brojeva na samom kraju stringa i brise ih
        return re.sub(r'\d+$', '', str(tekst))
    
    # Kreiramo novu privremenu kolonu koja nam služi kao stvarni ID grupe
    df['grupa_racuna'] = df['racun_id'].apply(izvuci_koren)
    
    # 2. Grupisanje po novoj koloni i brojanje slika
    statistika = df.groupby('grupa_racuna').size().reset_index(name='broj_slika')
    statistika = statistika.sort_values(by='broj_slika', ascending=False).reset_index(drop=True)
    
    print("\n" + "="*50)
    print(" PREGLED GRUPA RAČUNA (BEZ BROJEVA NA KRAJU)")
    print("="*50)
    for idx, row in statistika.iterrows():
        print(f"[{idx+1}] Grupa: {row['grupa_racuna']:<20} | Ukupno slika: {row['broj_slika']}")
    print("="*50)
    print(f"Ukupno jedinstvenih grupa: {len(statistika)}")
    print(f"Ukupno slika u celom datasetu: {len(df)}\n")
    
    # ── OVDE RUČNO UPISUJEŠ IMENA GRUPA (BEZ BROJEVA) KAKO ŽELIŠ DA IH PODELIŠ ──
    # Pogledaj ispis iznad i rasporedi korenove reči (npr. 'struja', 'grejanje')
    
    TRAIN_GRUPE = [
        'struja', 'mts', 'uplatnica'
        # ... dopuni listu grupama koje želiš u treningu
    ]
    
    VAL_GRUPE = [
        'jotel', 'grejanje',
        # ... dopuni listu za validaciju (npr. ovde staviš neku ćiriličnu grupu)
    ]
    
    TEST_GRUPE = [
        'odrzavanje_zgrade', 'smece'
        # ... dopuni listu za test
    ]
    
    # ── 3. Automatsko kreiranje podskupova na osnovu tvoje podele ───────────────
    df_train = df[df['grupa_racuna'].isin(TRAIN_GRUPE)].reset_index(drop=True)
    df_val = df[df['grupa_racuna'].isin(VAL_GRUPE)].reset_index(drop=True)
    df_test = df[df['grupa_racuna'].isin(TEST_GRUPE)].reset_index(drop=True)
    
    # Ispis finalne statistike
    print(" KONAČNA RASPODELA SLIKA:")
    print(f"  - TRAIN skup: {len(df_train)} slika ({len(df_train)/len(df)*100:.1f}%)")
    print(f"  - VAL skup:   {len(df_val)} slika ({len(df_val)/len(df)*100:.1f}%)")
    print(f"  - TEST skup:  {len(df_test)} slika ({len(df_test)/len(df)*100:.1f}%)")
    
    # Provera da li je nešto izostavljeno
    zaboravljeni = set(df['grupa_racuna'].unique()) - set(TRAIN_GRUPE + VAL_GRUPE + TEST_GRUPE)
    if zaboravljeni:
        print(f"\nNeraspodeljeni racuni: {zaboravljeni}")
        
    # Brišemo privremenu kolonu pre vraćanja da ti ne kvari originalnu strukturu CSV-a
    df_train = df_train.drop(columns=['grupa_racuna'])
    df_val = df_val.drop(columns=['grupa_racuna'])
    df_test = df_test.drop(columns=['grupa_racuna'])
    
    return df_train, df_val, df_test

df_train, df_val, df_test = analiziraj_i_podeli_po_korenu(CONFIG)