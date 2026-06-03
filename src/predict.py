import cv2
from ultralytics import YOLO
from pathlib import Path

def main():
    # 1. Putanja do tvog istreniranog modela
    model_path = "/home/milana/Desktop/ocr/best.pt"
    model = YOLO(model_path)
    
    # 2. Putanja do neke test slike računa na tvom laptopu
    # Stavi ovde putanju do bilo kog računa koji želiš da testiraš
    img_path = "/home/milana/Desktop/ocr/originalne_slike/struja4.jpg"
    
    if not Path(img_path).exists():
        print("Ne postoji putanjna do slike")
    # 1. Putanja do tvog istreniranog modela
    model_path = "/home/milana/Desktop/ocr/best.pt"
    model = YOLO(model_path)
    
    # 2. Putanja do neke test slike računa na tvom laptopu
    # Stavi ovde putanju do bilo kog računa koji želiš da testiraš
    img_path = "/home/milana/Desktop/ocr/originalne_slike/struja4.jpg"
    
    if not Path(img_path).exists():
        print(f"Slika nije pronađena na putanji: {img_path}")
        return

    # 3. Pokretanje YOLO predikcije
    img = cv2.imread(img_path)
    results = model(img)
    
    # Kreiramo folder gde ćemo čuvati isečke da ih vidiš očima
    output_dir = Path("/home/milana/Desktop/ocr/output")
    output_dir.mkdir(exist_ok=True)

    # 4. Prolazimo kroz sve detektovane kutije
    for i, box in enumerate(results[0].boxes):
        # Uzimamo koordinate (x1, y1, x2, y2) kao celobrojne vrednosti
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Uzimamo indeks i ime klase koju je YOLO prepoznao
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        
        # OpenCV sečenje (crop) - sečemo tačan pravougaonik sa slike
        cropped_img = img[y1:y2, x1:x2]
        
        # Čuvamo isečeni region na disk
        crop_name = f"{class_name}_crop_{i}.jpg"
        cv2.imwrite(str(output_dir / crop_name), cropped_img)
        print(f"Isečena klasa [{class_name}] i sačuvana kao {crop_name}")

if __name__ == "__main__":
    main()