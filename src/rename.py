from pathlib import Path
import shutil
import hashlib

def hash_slike(putanja):
    with open(putanja, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

ulaz = Path("/home/milana/Desktop/ocr/originalne_slike")
izlaz = Path("/home/milana/Desktop/ocr/originalne_slike")
izlaz.mkdir(exist_ok=True)

videni_heshovi = set()

for folder in ulaz.iterdir():
    if folder.is_dir():
        naziv = folder.name
        slike = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg"))
        brojac = 1
        for slika in slike:
            hes = hash_slike(slika)
            if hes in videni_heshovi:
                print(f"⚠ Duplikat, preskačem: {slika.name}")
                continue
            videni_heshovi.add(hes)
            novo_ime = f"{naziv}{brojac}.jpg"
            shutil.copy(str(slika), str(izlaz / novo_ime))
            print(f"✓ {slika.name} → {novo_ime}")
            brojac += 1