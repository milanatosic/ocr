# src/train_yolo.py
from ultralytics import YOLO

def main():
    # 1. Učitavamo pre-trenirani YOLOv8 nano model (v8n) koji je super brz
    model = YOLO("yolov8n.pt")

    # 2. Pokrećemo treniranje modela
    results = model.train(
        data="./Inteligentni-sistemi-1/data.yaml",  # Pogledaj tačan naziv foldera koji ti je Roboflow napravio i izmeni ako treba!
        epochs=50,                                        # 50 epoha je sasvim dovoljno za stabilan rezultat
        imgsz=640,                                        # Standardna rezolucija na kojoj je YOLO treniran
        device="cpu"                                      # Treniramo lokalno na tvom procesoru
    )
    print("Trening je uspešno završen")

if __name__ == "__main__":
    main()