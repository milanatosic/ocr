import cv2
import numpy as np
import csv
from ultralytics import YOLO
from pathlib import Path
 
# Globalne promenljive
points = []          # Trenutni poligon koji se crta (lista tačaka)
polygons = []        # Lista svih završenih poligona: [[(x1,y1),(x2,y2),(x3,y3),(x4,y4)], ...]
current_display_img = None
 
def mouse_handler(event, x, y, flags, param):
    global points, polygons, current_display_img
 
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        redraw(current_display_img.copy())
 
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Desni klik = odustani od trenutnog poligona
        if points:
            points = []
            redraw(current_display_img.copy())
 
def redraw(img):
    # Crtaj završene poligone (plavo)
    for idx, poly in enumerate(polygons):
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.putText(img, f"Red_{idx}", (poly[0][0] + 5, poly[0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
 
    # Crtaj trenutne tačke (zeleno)
    for i, pt in enumerate(points):
        cv2.circle(img, pt, 5, (0, 255, 0), -1)
        if i > 0:
            cv2.line(img, points[i-1], pt, (0, 255, 0), 2)
 
    # Ako ima 4 tačke, zatvori poligon isprekidanom linijom
    if len(points) == 4:
        cv2.line(img, points[3], points[0], (0, 255, 0), 2)
 
    # Uputstvo u gornjem uglu
    cv2.putText(img, "Klikni 4 tacke -> Enter potvrdi | Z undo | R reset | S sacuvaj | Space preskoci",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f"Tacke: {len(points)}/4",
                (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
 
    cv2.imshow("CRNN Dataset Creator", img)
 
def perspektivni_crop(region, poly_points, scale=2.0):
    """
    Izvlači sadržaj poligona kao ispravljenu pravougaonu sliku.
    poly_points su koordinate na uvećanom prikazu (2x), pa ih delimo sa scale.
    """
    # Mapiramo koordinate nazad na originalni region
    orig_pts = np.array([(int(x / scale), int(y / scale)) for x, y in poly_points], dtype=np.float32)
 
    # Izračunaj širinu i visinu izlaznog cropa
    w1 = np.linalg.norm(orig_pts[1] - orig_pts[0])
    w2 = np.linalg.norm(orig_pts[2] - orig_pts[3])
    h1 = np.linalg.norm(orig_pts[3] - orig_pts[0])
    h2 = np.linalg.norm(orig_pts[2] - orig_pts[1])
    W = int(max(w1, w2))
    H = int(max(h1, h2))
 
    if W == 0 or H == 0:
        return None
 
    dst = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(orig_pts, dst)
    warped = cv2.warpPerspective(region, M, (W, H))
    return warped
 
def main():
    global points, polygons, current_display_img
 
    model = YOLO("/home/milana/Desktop/ocr/best.pt")
    images_dir = Path("/home/milana/Desktop/ocr/originalne_slike")
    output_rows_dir = Path("/home/milana/Desktop/ocr/output_rows")
    output_rows_dir.mkdir(exist_ok=True, parents=True)
 
    csv_path = Path("/home/milana/Desktop/ocr/dataset.csv")
    csv_exists = csv_path.exists()
    csv_file = open(csv_path, mode='a', encoding='utf-8', newline='')
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(['image_path', 'label'])
 
    available_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg"))
    if not available_images:
        print("❌ Nema pronađenih slika.")
        csv_file.close()
        return
 
    cv2.namedWindow("CRNN Dataset Creator")
    cv2.setMouseCallback("CRNN Dataset Creator", mouse_handler)
 
    print("\n=======================================================")
    print("   ANOTATOR - POLIGONI ZA KOSE REDOVE                 ")
    print("=======================================================")
    print(" 🖱️  Klikni 4 tačke oko reda (gore-levo, gore-desno,")
    print("      dole-desno, dole-levo) -> Enter potvrdi poligon")
    print(" ⌨️  Z  -> Obriši poslednju tačku")
    print(" ⌨️  R  -> Resetuj sve za ovaj region")
    print(" 💾 S  -> Sačuvaj i unesi labele u terminalu")
    print(" ⏭️  Space -> Preskoči region")
    print("=======================================================\n")
 
    for img_idx, path in enumerate(available_images):
        print(f"\n📸 [{img_idx+1}/{len(available_images)}] {path.name}")
        img = cv2.imread(str(path))
        if img is None:
            continue
        results = model(img, verbose=False)
 
        for i, box in enumerate(results[0].boxes):
            x1_yolo, y1_yolo, x2_yolo, y2_yolo = map(int, box.xyxy[0])
            class_name = model.names[int(box.cls[0])]
 
            cropped_region = img[y1_yolo:y2_yolo, x1_yolo:x2_yolo]
            if cropped_region.size == 0:
                continue
 
            scale = 2.0
            width = int(cropped_region.shape[1] * scale)
            height = int(cropped_region.shape[0] * scale)
            resized_display = cv2.resize(cropped_region, (width, height), interpolation=cv2.INTER_CUBIC)
 
            points = []
            polygons = []
            current_display_img = resized_display.copy()
            redraw(current_display_img.copy())
 
            print(f"   🔍 [{class_name}] Klikni 4 tačke oko svakog reda, Enter za potvrdu.")
 
            while True:
                key = cv2.waitKey(10) & 0xFF
 
                # Enter -> potvrdi poligon ako ima 4 tačke
                if key == 13:
                    if len(points) == 4:
                        polygons.append(points.copy())
                        points = []
                        print(f"   ✓ Poligon {len(polygons)} potvrđen.")
                        redraw(current_display_img.copy())
                    else:
                        print(f"   ⚠ Potrebno je tačno 4 tačke (trenutno {len(points)}).")
 
                # Z -> obriši poslednju tačku
                elif key == ord('z') or key == ord('Z'):
                    if points:
                        points.pop()
                        print("   ↩️  Obrisana poslednja tačka.")
                    elif polygons:
                        points = polygons.pop()
                        print("   ↩️  Vraćen poslednji poligon za izmenu.")
                    redraw(current_display_img.copy())
 
                # R -> resetuj sve
                elif key == ord('r') or key == ord('R'):
                    points = []
                    polygons = []
                    redraw(current_display_img.copy())
                    print("   🔄 Resetovano.")
 
                # S -> sačuvaj
                elif key == ord('s') or key == ord('S'):
                    if not polygons:
                        print("   ⚠ Nema poligona. Nacrtaj bar jedan ili stisni Space.")
                        continue
 
                    print(f"\n--- 📝 UNOS TEKSTA ZA [{class_name}] ---")
                    for idx, poly in enumerate(polygons):
                        crop = perspektivni_crop(cropped_region, poly, scale=scale)
                        if crop is None:
                            continue
 
                        row_name = f"{path.stem}_{class_name}_r{idx}.jpg"
                        cv2.imwrite(str(output_rows_dir / row_name), crop)
 
                        text_label = input(f" Tekst za Red {idx} ({row_name}): ").strip()
                        csv_writer.writerow([f"output_rows/{row_name}", text_label])
                        csv_file.flush()
 
                    print("✅ Sačuvano.")
                    break
 
                # Space -> preskoči
                elif key == 32:
                    print(f"   ⏭️  Preskočen [{class_name}].")
                    break
 
        print("\n-------------------------------------------------------------------")
        izlaz = input("Enter za sledeći račun, 'q' za prekid: ").strip().lower()
        if izlaz == 'q':
            print("👋 Prekidam...")
            break
 
    cv2.destroyAllWindows()
    csv_file.close()
    print(f"\n🎉 Gotovo!")
    print(f"📁 Slike: {output_rows_dir}")
    print(f"📊 Dataset: {csv_path}")
 
if __name__ == "__main__":
    main()
