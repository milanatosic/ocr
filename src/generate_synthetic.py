"""
Generiše sintetičke slike teksta za trening OCR-a.
Posebno fokusirano na ĆIRILICU jer je nedovoljno zastupljena.

Pokretanje:
    python src/generate_synthetic.py
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    "output_dir":      ROOT / "synthetic",
    "output_csv":      ROOT / "synthetic" / "synthetic_dataset.csv",
    "num_samples_cyr": 2000,    # ćirilica (glavno)
    "num_samples_lat": 500,     # latinica (za balans)
    "img_height":      48,
}

# ── Fontovi (probaj različite, ako neki ne postoji preskočiće) ────────────────
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
]

# ── Rečnik tekstova karakterističnih za RAČUNE ────────────────────────────────
TEMPLATES_CYR = [
    # Naslovi i opisi usluga
    "Назив и врста услуге", "Датум", "Количина", "Цена", "Износ", "Период",
    "Накнада", "Услуга", "Попуст", "Камата", "Укупна накнада", "ПДВ",
    "Утрошено једнотарифно", "Утрошено двотарифно", "Шифра валуте",
    "Извештај о уплати", "Црвена зона", "Зелена зона", "Период обрачуна",
    "Број рачуна", "Шифра плаћања", "Позив на број", "Прималац",
    "Уплатилац", "Сврха уплате", "Износ за уплату", "Рок плаћања",
    "Контролни број", "Идентификациони број", "ПИБ", "Матични број",
    "Адреса", "Место", "Поштански број", "Република Србија",
    # Tekst za stavke
    "Грејање стана", "Утрошак воде", "Електрична енергија",
    "Одржавање зграде", "Одношење смећа", "Кабловска телевизија",
    "Интернет услуга", "Телефонска услуга", "Накнада за гас",
    "Порез на имовину", "Накнада за уређење", "Комунална такса",
    # Brojevi sa labelama
    "Укупно за рачун", "Износ без ПДВ", "Стопа ПДВ", "Основица",
    "Фиксни део", "Варијабилни део", "Тарифа",
]

TEMPLATES_LAT = [
    "Naziv i vrsta usluge", "Datum", "Količina", "Cena", "Iznos",
    "Naknada", "Usluga", "Popust", "Kamata", "Ukupna naknada", "PDV",
    "Šifra valute", "Period obračuna", "Broj računa", "Šifra plaćanja",
    "Poziv na broj", "Primalac", "Uplatilac", "Svrha uplate",
    "Iznos za uplatu", "Rok plaćanja", "Kontrolni broj", "PIB",
    "Adresa", "Mesto", "Republika Srbija", "Grejanje stana",
    "Utrošak vode", "Električna energija", "Održavanje zgrade",
    "Odnošenje smeća", "Kablovska televizija", "Internet usluga",
    "Telefonska usluga", "Porez na imovinu", "Komunalna taksa",
    "Ukupno za račun", "Iznos bez PDV", "Stopa PDV", "Osnovica",
    "Fiksni deo", "Varijabilni deo", "Tarifa",
]


def random_number(template_type=None):
    """Generiše realističan broj kakav se javlja na računima."""
    t = template_type or random.choice(['price', 'date', 'percent', 'id', 'amount'])
    
    if t == 'price':
        # 123.45 ili 1,234.56 ili 12,345.67
        n = random.choice([
            f"{random.uniform(1, 9999):.2f}",
            f"{random.randint(100, 999999):,.2f}",
            f"{random.uniform(0.01, 99.99):.2f} din",
            f"={random.uniform(100, 9999):.2f} Din.",
        ])
        return n
    elif t == 'date':
        d = random.randint(1, 28)
        m = random.randint(1, 12)
        y = random.randint(2020, 2026)
        return random.choice([
            f"{d:02d}.{m:02d}.{y}.",
            f"{d:02d}.{m:02d}.{y}.-{d:02d}.{m:02d}.{y}.",
            f"{d:02d}/{m:02d}/{y}",
        ])
    elif t == 'percent':
        return f"{random.choice([5, 10, 15, 20, 25, 50, 90, 100])}.00%"
    elif t == 'id':
        return ''.join([str(random.randint(0, 9)) for _ in range(random.randint(8, 16))])
    elif t == 'amount':
        return f"{random.uniform(1, 999):.2f} kWh"


def random_text(script='cyr'):
    """Generiše random tekst kombinujući templates i brojeve."""
    templates = TEMPLATES_CYR if script == 'cyr' else TEMPLATES_LAT
    
    choice = random.random()
    if choice < 0.3:
        # Samo template
        return random.choice(templates)
    elif choice < 0.6:
        # Template + broj
        return f"{random.choice(templates)} {random_number()}"
    elif choice < 0.8:
        # Više brojeva
        return f"{random.choice(templates)} {random_number()} {random_number()}"
    elif choice < 0.9:
        # Samo broj
        return random_number()
    else:
        # Kombinacija sa procentom
        return f"{random.choice(templates)} {random_number('percent')} {random_number('price')}"


def render_text_image(text, font_path, font_size=32):
    """Renderuje tekst u sliku."""
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        return None
    
    # Izračunaj veličinu teksta
    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    # Padding oko teksta
    pad = 10
    img_w = text_w + 2 * pad
    img_h = text_h + 2 * pad
    
    # Boje (pozadina svetla, tekst taman — ali sa varijacijama)
    bg_color = random.randint(220, 255)
    text_color = random.randint(0, 80)
    
    img = Image.new('L', (img_w, img_h), bg_color)
    draw = ImageDraw.Draw(img)
    draw.text((pad - bbox[0], pad - bbox[1]), text, fill=text_color, font=font)
    
    return np.array(img)


def add_realistic_noise(img):
    """Doda realistične distorzije slici."""

    # Osiguraj da je slika uint8 i 2D
    img = img.astype(np.uint8)

    # Blaga rotacija
    if random.random() < 0.3:
        angle = random.uniform(-2, 2)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

        img = cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

    # Gaussian noise
    if random.random() < 0.4:
        sigma = random.uniform(3, 10)
        noise = np.random.normal(0, sigma, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Blur
    if random.random() < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # Erozija / dilatacija
    if random.random() < 0.2:
        kernel = np.ones((2, 2), np.uint8)
        if random.random() < 0.5:
            img = cv2.erode(img, kernel, iterations=1)
        else:
            img = cv2.dilate(img, kernel, iterations=1)

    # JPEG kompresija
    if random.random() < 0.3:
        q = random.randint(60, 95)
        _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        decoded = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

        if decoded is not None:
            img = decoded

    return img


def main():
    output_dir = CONFIG["output_dir"]
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Proveri koji fontovi postoje
    valid_fonts = [f for f in FONT_PATHS if Path(f).exists()]
    if len(valid_fonts) == 0:
        print("Nijedan font nije pronađen!")
        print("   Instaliraj: sudo apt-get install fonts-dejavu fonts-liberation fonts-freefont-ttf")
        return
    print(f"Pronađeno fontova: {len(valid_fonts)}")
    for f in valid_fonts:
        print(f"   - {Path(f).name}")
    
    records = []
    
    # ── Generiši ćirilične ────────────────────────────────────────────────────
    print(f"\nGenerisanje {CONFIG['num_samples_cyr']} ćiriličnih slika...")
    for i in tqdm(range(CONFIG["num_samples_cyr"]), desc="Ćirilica"):
        text = random_text('cyr')
        font_path = random.choice(valid_fonts)
        font_size = random.randint(28, 40)
        
        img = render_text_image(text, font_path, font_size)
        if img is None:
            continue
        img = add_realistic_noise(img)
        
        filename = f"synth_cyr_{i:05d}.png"
        filepath = output_dir / filename
        cv2.imwrite(str(filepath), img)
        
        records.append({
            "image_path": f"synthetic/{filename}",
            "label": text,
            "racun_id": f"synth_cyr_{i:05d}",
        })
    
    # ── Generiši latinične ───────────────────────────────────────────────────
    print(f"\nGenerisanje {CONFIG['num_samples_lat']} latiničnih slika...")
    for i in tqdm(range(CONFIG["num_samples_lat"]), desc="Latinica"):
        text = random_text('lat')
        font_path = random.choice(valid_fonts)
        font_size = random.randint(28, 40)
        
        img = render_text_image(text, font_path, font_size)
        if img is None:
            continue
        img = add_realistic_noise(img)
        
        filename = f"synth_lat_{i:05d}.png"
        filepath = output_dir / filename
        cv2.imwrite(str(filepath), img)
        
        records.append({
            "image_path": f"synthetic/{filename}",
            "label": text,
            "racun_id": f"synth_lat_{i:05d}",
        })
    
    # ── Sačuvaj CSV ──────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df.to_csv(CONFIG["output_csv"], index=False)
    
    print(f"\n{'═'*60}")
    print(f"  GENERISANO {len(df)} SINTETIČKIH SLIKA")
    print(f"{'═'*60}")
    print(f"  Folder: {output_dir}")
    print(f"  CSV:    {CONFIG['output_csv']}")
    print(f"  Ćirilica: {sum(1 for r in records if 'cyr' in r['image_path'])}")
    print(f"  Latinica: {sum(1 for r in records if 'lat' in r['image_path'])}")


if __name__ == "__main__":
    main()