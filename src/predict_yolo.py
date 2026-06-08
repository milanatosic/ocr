import cv2
from ultralytics import YOLO
from pathlib import Path

def main():
    # --- PAMETNO ODREĐIVANJE PUTANJE (ZA TEBE I KOLEGICU) ---
    TEKUCI_DIR = Path(__file__).resolve().parent
    BASE_DIR = TEKUCI_DIR.parent if TEKUCI_DIR.name == "src" else TEKUCI_DIR

    # 1. Putanja do istreniranog modela u korenu projekta
    model_path = BASE_DIR / "best.pt"
    
    # 2. Relativna putanja do test slike unutar 'originalne_slike'
    img_path = BASE_DIR / "originalne_slike" / "struja4.jpg"
    
    # Putanja do foldera gde se čuvaju isečeni delovi (output)
    output_dir = BASE_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    # --------------------------------------------------------

    # Provera da li model postoji
    if not model_path.exists():
        print(f"Model nije pronađen na lokaciji: {model_path}")
        print("Pobrinite se da 'best.pt' bude u korenu projekta.")
        return

    # Provera da li slika postoji
    if not img_path.exists():
        print(f"Slika nije pronađena na lokaciji: {img_path}")
        print("Proveri da li se slika 'struja4.jpg' nalazi unutar foldera 'originalne_slike'.")
        return

    # Učitavanje YOLO modela
    print(f"Učitavam model: {model_path.name}...")
    model = YOLO(str(model_path))
    
    # Pokretanje YOLO predikcije
    print(f"Analiziram sliku: {img_path.name}...")
    img = cv2.imread(str(img_path))
    results = model(img)
    
    # Prolazimo kroz sve detektovane kutije (bounding boxes)
    brojac_isecaka = 0
    for i, box in enumerate(results[0].boxes):
        # Uzimamo koordinate (x1, y1, x2, y2) kao celobrojne vrednosti
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Uzimamo indeks i ime klase koju je YOLO prepoznao
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        
        # OpenCV sečenje (crop)
        cropped_img = img[y1:y2, x1:x2]
        
        # Provera da li je isečak validan (da nema širinu ili visinu 0)
        if cropped_img.size > 0:
            crop_name = f"{class_name}_crop_{i}.jpg"
            cv2.imwrite(str(output_dir / crop_name), cropped_img)
            print(f"✓ Isečena klasa [{class_name}] → {crop_name}")
            brojac_isecaka += 1

    print("\n" + "=" * 40)
    print(f"Gotovo! Ukupno sačuvano isečaka u '{output_dir.name}': {brojac_isecaka}")

if __name__ == "__main__":
    main()