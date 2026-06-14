import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    "output_dir":           ROOT / "synthetic",
    "output_csv":           ROOT / "synthetic" / "synthetic_dataset.csv",
    "num_samples_cyr":      500,
    "num_samples_cyr_short": 1000,  # kratke ćirilične reči iz tabela
    "num_samples_lat":      300,
    "img_height":           48,
}

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

BOLD_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

# ── Templates ─────────────────────────────────────────────────────────────────

# Duge ćirilične rečenice (kao pre)
TEMPLATES_CYR = [
    "Назив и врста услуге", "Датум", "Количина", "Цена", "Износ", "Период",
    "Накнада", "Услуга", "Попуст", "Камата", "Укупна накнада", "ПДВ",
    "Утрошеноједнотарифно", "Утрошено двотарифно", "Шифра валуте",
    "Извештај о уплати", "Црвена зона", "Зелена зона", "Период обрачуна",
    "Број рачуна", "Шифра плаћања", "Позив на број", "Прималац",
    "Уплатилац", "Сврха уплате", "Износ за уплату", "Рок плаћања",
    "Контролни број", "Идентификациони број", "ПИБ", "Матични број",
    "Адреса", "Место", "Поштански број", "Република Србија",
    "Грејање стана", "Утрошак воде", "Електрична енергија",
    "Одржавање зграде", "Одношење смећа", "Кабловска телевизија",
    "Интернет услуга", "Телефонска услуга", "Накнада за гас",
    "Порез на имовину", "Накнада за уређење", "Комунална такса",
    "Укупно за рачун", "Износ без ПДВ", "Стопа ПДВ", "Основица",
    "Фиксни део", "Варијабилни део", "Тарифа",
    # Specifično za struja račune
    "Потрошња у обрачунском периоду",
    "Укупно задужење за обрачунски период",
    "За уплату за електричну енергију",
    "Преплата за претходни обрачунски период",
    "Гарантовано снабдевање",
    "Број бројила/начин очитавања",
    "Датум издавања рачуна",
    "Место издавања рачуна",
    "Датум промета и акцизе",
    "Врста снабдевања",
    "Адреса мерног места",
    "Шифра мерног места",
    "Рок за плаћање",
    "Контакт центар",
    "Идентификациона ознака одговорног лица",
]

# Kratke ćirilične reči — zaglavlja tabela, oznake polja
TEMPLATES_CYR_SHORT = [
    # Zaglavlja tabela struja računa
    "Назив", "Цена", "Износ", "Датум", "Број",
    "Валута", "Модел", "Шифра", "Период", "Тарифа",
    "Накнада", "Износ", "Услуга", "Попуст", "Камата",
    # Caps verzije (kao u zaglavljima EPS računa)
    "НАЗИВ", "ЦЕНА", "ИЗНОС", "ДАТУМ", "БРОЈ",
    "ВАЛУТА", "ТАРИФА", "ПЕРИОД", "ШИФРА", "НАКНАДА",
    "РАЧУН", "ЗБИР", "УКУПНО", "ОСНОВИЦА", "ПОПУСТ",
    # Kratke kombinacije
    "Цена по", "Број дана", "Датум од", "Датум до",
    "Плава зона", "Црвена зона", "Зелена зона",
    "Утрошено", "јединици", "динара", "(динара)",
    "Рачун", "број", "шифра", "валута", "износ", "модела",
    # Specifični termini sa struja računa
    "ЈТ/ДУТ", "ЈТ", "ДУТ", "kWh", "дин",
    "ТАРИФА", "Тарифа", "тарифа",
    # ALL CAPS naslovi kao na EPS računu
    "РАЧУН ЗА ЕЛЕКТРИЧНУ ЕНЕРГИЈУ",
    "ПОТРОШЊА У ОБРАЧУНСКОМ ПЕРИОДУ",
    "УКУПНО ЗАДУЖЕЊЕ ЗА ОБРАЧУНСКИ ПЕРИОД",
    "ЗА УПЛАТУ ЗА ЕЛЕКТРИЧНУ ЕНЕРГИЈУ",
    "ПРЕПЛАТА ЗА ПРЕТХОДНИ ОБРАЧУНСКИ ПЕРИОД",
    "РОК ЗА ПЛАЋАЊЕ",
    "ОКТОБАР", "НОВЕМБАР", "ДЕЦЕМБАР",
    "ЈАНУАР", "ФЕБРУАР", "МАРТ",
    "АПРИЛ", "МАЈ", "ЈУН",
    "ЈУЛ", "АВГУСТ", "СЕПТЕМБАР",
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
    t = template_type or random.choice(['price', 'date', 'percent', 'id', 'amount'])
    if t == 'price':
        return random.choice([
            f"{random.uniform(1, 9999):.2f}",
            f"{random.randint(100, 999999):,.2f}",
            f"{random.uniform(0.01, 99.99):.2f} din",
            f"={random.uniform(100, 9999):.2f} Din.",
        ])
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
    templates = TEMPLATES_CYR if script == 'cyr' else TEMPLATES_LAT
    choice = random.random()
    if choice < 0.3:
        return random.choice(templates)
    elif choice < 0.6:
        return f"{random.choice(templates)} {random_number()}"
    elif choice < 0.8:
        return f"{random.choice(templates)} {random_number()} {random_number()}"
    elif choice < 0.9:
        return random_number()
    else:
        return f"{random.choice(templates)} {random_number('percent')} {random_number('price')}"


def random_text_short():
    """Kratke ćirilične reči — zaglavlja tabela."""
    choice = random.random()
    if choice < 0.5:
        # Samo kratka reč
        return random.choice(TEMPLATES_CYR_SHORT)
    elif choice < 0.75:
        # Kratka reč + broj (kao u tabeli)
        return f"{random.choice(TEMPLATES_CYR_SHORT)} {random_number('price')}"
    else:
        # Kratka reč + kWh ili dин
        val = random.uniform(1, 999)
        unit = random.choice(["kWh", "дин", "din", "m2", "%"])
        return f"{random.choice(TEMPLATES_CYR_SHORT)} {val:.2f} {unit}"


def render_text_image(text, font_path, font_size=32, style='normal'):
    """Renderuje tekst u sliku sa različitim vizuelnim stilovima."""
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        return None

    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    pad = 10
    img_w = text_w + 2 * pad
    img_h = text_h + 2 * pad

    if style == 'normal':
        # Svetla pozadina, taman tekst
        bg_color = random.randint(230, 255)
        text_color = random.randint(0, 60)
        img = Image.new('L', (img_w, img_h), bg_color)

    elif style == 'table_header':
        # Siva pozadina kao zaglavlje tabele na struja računu
        bg_color = random.randint(190, 225)
        text_color = random.randint(0, 50)
        img = Image.new('L', (img_w, img_h), bg_color)

    elif style == 'table_row':
        # Svetlo siva pozadina — red u tabeli
        bg_color = random.randint(235, 250)
        text_color = random.randint(10, 60)
        img = Image.new('L', (img_w, img_h), bg_color)

    elif style == 'dark_header':
        # Tamna pozadina, svetli tekst (kao plavi header EPS računa)
        bg_color = random.randint(30, 80)
        text_color = random.randint(200, 255)
        img = Image.new('L', (img_w, img_h), bg_color)

    else:
        bg_color = random.randint(220, 255)
        text_color = random.randint(0, 80)
        img = Image.new('L', (img_w, img_h), bg_color)

    draw = ImageDraw.Draw(img)
    draw.text((pad - bbox[0], pad - bbox[1]), text, fill=text_color, font=font)
    return np.array(img)


def add_realistic_noise(img):
    img = img.astype(np.uint8)

    if random.random() < 0.3:
        angle = random.uniform(-2, 2)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

    if random.random() < 0.4:
        sigma = random.uniform(3, 10)
        noise = np.random.normal(0, sigma, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    if random.random() < 0.2:
        kernel = np.ones((2, 2), np.uint8)
        if random.random() < 0.5:
            img = cv2.erode(img, kernel, iterations=1)
        else:
            img = cv2.dilate(img, kernel, iterations=1)

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

    valid_fonts = [f for f in FONT_PATHS if Path(f).exists()]
    valid_bold_fonts = [f for f in BOLD_FONT_PATHS if Path(f).exists()]
    if not valid_bold_fonts:
        valid_bold_fonts = valid_fonts  # fallback

    if len(valid_fonts) == 0:
        print("Nijedan font nije pronađen!")
        print("   Instaliraj: sudo apt-get install fonts-dejavu fonts-liberation fonts-freefont-ttf")
        return
    print(f"Pronađeno fontova: {len(valid_fonts)}")
    for f in valid_fonts:
        print(f"   - {Path(f).name}")

    records = []

    # ── 1. Duge ćirilične rečenice ────────────────────────────────────────────
    print(f"\nGenerisanje {CONFIG['num_samples_cyr']} dugih ćiriličnih slika...")
    for i in tqdm(range(CONFIG["num_samples_cyr"]), desc="Ćirilica (duge)"):
        text = random_text('cyr')

        # Stil i font zavise jedan od drugog
        style = random.choices(
            ['normal', 'table_header', 'table_row', 'dark_header'],
            weights=[0.5, 0.25, 0.2, 0.05]
        )[0]
        font_path = random.choice(valid_fonts)
        font_size = random.randint(26, 40)

        img = render_text_image(text, font_path, font_size, style)
        if img is None:
            continue
        img = add_realistic_noise(img)

        filename = f"synth_cyr_{i:05d}.png"
        cv2.imwrite(str(output_dir / filename), img)
        records.append({
            "image_path": f"synthetic/{filename}",
            "label": text,
            "racun_id": f"synth_cyr_{i:05d}",
        })

    # ── 2. Kratke ćirilične reči (zaglavlja tabela) ───────────────────────────
    print(f"\nGenerisanje {CONFIG['num_samples_cyr_short']} kratkih ćiriličnih slika...")
    for i in tqdm(range(CONFIG["num_samples_cyr_short"]), desc="Ćirilica (kratke)"):
        text = random_text_short()

        # Kratke reči u zaglavljima su češće bold i na sivoj pozadini
        style = random.choices(
            ['normal', 'table_header', 'table_row', 'dark_header'],
            weights=[0.2, 0.45, 0.25, 0.1]
        )[0]

        # Bold font češći za kratke reči (kao na EPS računu)
        font_path = random.choices(
            [random.choice(valid_bold_fonts), random.choice(valid_fonts)],
            weights=[0.6, 0.4]
        )[0]
        font_size = random.randint(20, 36)

        img = render_text_image(text, font_path, font_size, style)
        if img is None:
            continue
        img = add_realistic_noise(img)

        filename = f"synth_cyr_short_{i:05d}.png"
        cv2.imwrite(str(output_dir / filename), img)
        records.append({
            "image_path": f"synthetic/{filename}",
            "label": text,
            "racun_id": f"synth_cyr_short_{i:05d}",
        })

    # ── 3. Latinične ─────────────────────────────────────────────────────────
    print(f"\nGenerisanje {CONFIG['num_samples_lat']} latiničnih slika...")
    for i in tqdm(range(CONFIG["num_samples_lat"]), desc="Latinica"):
        text = random_text('lat')
        style = random.choices(
            ['normal', 'table_header', 'table_row'],
            weights=[0.6, 0.25, 0.15]
        )[0]
        font_path = random.choice(valid_fonts)
        font_size = random.randint(28, 40)

        img = render_text_image(text, font_path, font_size, style)
        if img is None:
            continue
        img = add_realistic_noise(img)

        filename = f"synth_lat_{i:05d}.png"
        cv2.imwrite(str(output_dir / filename), img)
        records.append({
            "image_path": f"synthetic/{filename}",
            "label": text,
            "racun_id": f"synth_lat_{i:05d}",
        })

    # ── Sačuvaj CSV ───────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df.to_csv(CONFIG["output_csv"], index=False)

    n_cyr = sum(1 for r in records if 'synth_cyr_' in r['image_path'] and 'short' not in r['image_path'])
    n_short = sum(1 for r in records if 'short' in r['image_path'])
    n_lat = sum(1 for r in records if 'lat' in r['image_path'])

    print(f"\n{'═'*60}")
    print(f"  GENERISANO {len(df)} SINTETIČKIH SLIKA")
    print(f"{'═'*60}")
    print(f"  Folder:          {output_dir}")
    print(f"  CSV:             {CONFIG['output_csv']}")
    print(f"  Ćirilica (duge): {n_cyr}")
    print(f"  Ćirilica (kratke zaglavlja): {n_short}")
    print(f"  Latinica:        {n_lat}")


if __name__ == "__main__":
    main()