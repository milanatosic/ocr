import hashlib
from pathlib import Path
import shutil


def hash_slike(putanja):
    with open(putanja, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


TEKUCI_DIR = Path(__file__).resolve().parent
BASE_DIR = TEKUCI_DIR.parent if TEKUCI_DIR.name == "src" else TEKUCI_DIR

ulaz = BASE_DIR / "originalne_slike"
izlaz = BASE_DIR / "preciscene_slike"
izlaz.mkdir(parents=True, exist_ok=True)

print(f"🔍 Provera foldera: {ulaz}")

if not ulaz.exists():
    print(f"❌ Ulazni folder uopšte ne postoji!")
else:
    # Brojimo stvari unutar originalne_slike
    sadrzaj_ulaza = list(ulaz.iterdir())
    print(f"📦 Unutar 'originalne_slike' pronađeno ukupno stavki: {len(sadrzaj_ulaza)}")
    
    for stavka in sadrzaj_ulaza:
        print(f"  -> Stavka: {stavka.name} (Da li je folder? {stavka.is_dir()})")
        
        if stavka.is_dir():
            # Ako je folder, da vidimo šta ima unutra
            sve_u_podfolderu = list(stavka.iterdir())
            print(f"     └─ Unutar podfoldera '{stavka.name}' ima ukupno {len(sve_u_podfolderu)} fajlova.")
            
            # Tražimo slike
            slike = (
                list(stavka.glob("*.jpg"))
                + list(stavka.glob("*.jpeg"))
                + list(stavka.glob("*.JPG"))
                + list(stavka.glob("*.JPEG"))
            )
            print(f"     └─ Od toga su slike (.jpg/.jpeg): {len(slike)}")

# --- Ostatak koda za kopiranje ---
videni_heshovi = set()
for folder in ulaz.iterdir():
    if folder.is_dir():
        naziv = folder.name
        slike = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.JPG")) + list(folder.glob("*.JPEG"))
        brojac = 1
        for slika in slike:
            hes = hash_slike(slika)
            if hes in videni_heshovi:
                continue
            videni_heshovi.add(hes)
            novo_ime = f"{naziv}{brojac}.jpg"
            shutil.copy(slika, izlaz / novo_ime)
            brojac += 1