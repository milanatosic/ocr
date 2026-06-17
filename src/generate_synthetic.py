"""
Generiše sintetičke slike teksta za trening OCR-a.
Posebno fokusirano na ĆIRILICU jer je nedovoljno zastupljena.
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
    "output_dir":            ROOT / "synthetic",
    "output_csv":            ROOT / "synthetic" / "synthetic_dataset.csv",
    "num_samples_cyr":       200,   # duge ćirilične rečenice (smanjeno)
    "num_samples_cyr_short": 400,  # kratke ćirilične reči (povećano)
    "num_samples_numbers":   300,   # samo brojevi i kratke numeričke kombinacije
    "num_samples_lat":       200,   # latinica (smanjeno)
    "img_height":            48,
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
    "Потрошња у обрачунском периоду",
    "Укупно задужење за обрачунски период",
    "За уплату за електричну енергију",
    "Преплата за претходни обрачунски период",
    "Гарантовано снабдевање",
    "Датум издавања рачуна",
    "Место издавања рачуна",
    "Датум промета и акцизе",
    "Адреса мерног места",
    "Рок за плаћање",
]

# Kratke ćirilične reči — zaglavlja tabela
TEMPLATES_CYR_SHORT = [
    # Jednorečne oznake
    "Назив", "Цена", "Износ", "Датум", "Број",
    "Валута", "Модел", "Шифра", "Период", "Тарифа",
    "Накнада", "Услуга", "Попуст", "Камата", "Рачун",
    "Збир", "Укупно", "Основица", "Зона", "Врста",
    # Caps verzije
    "НАЗИВ", "ЦЕНА", "ИЗНОС", "ДАТУМ", "БРОЈ",
    "ВАЛУТА", "ТАРИФА", "ПЕРИОД", "ШИФРА", "НАКНАДА",
    "РАЧУН", "ЗБИР", "УКУПНО", "ОСНОВИЦА", "ПОПУСТ",
    # Kratke kombinacije (2-3 reči)
    "Цена по", "Број дана", "Датум од", "Датум до",
    "Плава зона", "Црвена зона", "Зелена зона",
    "Зелена зона", "Плава зона",
    # Termini sa struja računa
    "Утрошено", "јединици", "динара", "(динара)",
    "ТАРИФА", "Тарифа",
    # Specifični termini
    "ЈТ/ДУТ", "ЈТ", "ДУТ",
    "(kW/kWh)", "kWh", "kW", "MWh", "MW", "дин",
    # Više primera sa словом Ц
    "Цена по јединици", "Цена без ПДВ", "Малопродајна цена",
    "Цене", "ЦЕНА", "Цена по", "Укупна цена",
    "Цена са ПДВ", "Јединична цена",
    # ALL CAPS naslovi
    "РАЧУН ЗА ЕЛЕКТРИЧНУ ЕНЕРГИЈУ",
    "ПОТРОШЊА У ОБРАЧУНСКОМ ПЕРИОДУ",
    "УКУПНО ЗАДУЖЕЊЕ ЗА ОБРАЧУНСКИ ПЕРИОД",
    "ЗА УПЛАТУ ЗА ЕЛЕКТРИЧНУ ЕНЕРГИЈУ",
    "РОК ЗА ПЛАЋАЊЕ",
    # Meseci
    "ОКТОБАР", "НОВЕМБАР", "ДЕЦЕМБАР",
    "ЈАНУАР", "ФЕБРУАР", "МАРТ",
    "АПРИЛ", "МАЈ", "ЈУН",
    "ЈУЛ", "АВГУСТ", "СЕПТЕМБАР",
]

TEMPLATES_RARE_CYR = [
    # Слова Љ (nikad viđena u realnom datasetu)
    "Поље", "Ваљево", "Краљево", "Недеља", "Земља", "Кљун",
    "Пољопривреда", "Насеље", "Пољска", "Вољна",
    # Слова Џ
    "Наруџба", "Наруџбина", "Џак",
    # Слова Ђ (samo 2 pojavljivanja u realnom datasetu)
    "Ђорђе", "Ђурађ", "Ђак", "Ђубриво",
    # Слова Ћ (samo 1 pojavljivanje)
    "Ноћна тарифа", "Ноћни период", "Кућа", "Вреће",
    # Velika latinična slova sa dijkritičima — Č i Đ nikad viđena
    "Čačak", "Čovek", "Čist", "Čačanska banka",
    "Šabac", "Šifra usluge",
    "Žitište", "Žiro račun",
    "Đorđević", "Đakovica", "Đurđevdan",
    # Kombinacije koje pokrivaju više retkih slova
    "Поље Ваљево", "Земља и насеље", "Краљево насеље",
    "Čačak Šabac", "Žiro Đorđević",
    "Ноћна потрошња", "Ноћни тарифни период",
]

TEMPLATES_NUMBERS_BALANCED = [
    # Garantuju cifre 4, 6, 7, 8, 9 koje su najređe u realnom datasetu
    "47.96", "74.83", "679.44", "87.54", "496.78",
    "4.76", "97.84", "647.99", "746", "849",
    "7/7", "48%", "96%", "74%", "67.89",
    "487", "694", "879", "746", "968",
    "14.07.2024.", "29.04.2023.", "17.06.2025.",
    "49.99", "87", "64", "96", "74",
    "467,89", "987,44", "746,00", "849,17",
    "4.796,87", "6.748,99", "9.467,00",
    # kW/kWh formati
    "174 kWh", "500 kW", "1.500 kWh", "3,5 kWh",
    "(kW/kWh)", "(kWh)", "(kW)", "kW/kWh",
    "247 kWh", "89 kW", "1.200 kWh", "450 kW",
    # Sa zagradama
    "(100,00)", "(bez PDV)", "(sa PDV)", "(din)",
    "(dinara)", "(RSD)", "(ukupno)", "(neto)",
    # Konfuzni parovi cifara: 1/7, 3/8, 4/9, 6/9
    "17", "71", "37", "73", "38", "83", "47", "74", "69", "96",
    "117", "171", "711", "337", "373", "733",
    "138", "183", "318", "381", "813", "831",
    "147", "174", "417", "471", "714", "741",
    "169", "196", "619", "691", "916", "961",
    "1.738", "3.847", "6.914", "7.431", "9.638",
    "17,38", "38,47", "69,14", "74,31", "96,38",
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
    # kW/kWh u kontekstu
    "Snaga (kW)", "Energija (kWh)", "Potrošnja (kWh)",
    "Obračunska snaga (kW)", "Aktivna energija (kWh)",
    "Reaktivna energija (kWh)", "Vršna snaga (kW)",
    # Ciljano za F, S, L, E, G, O, Z (najgori karakteri)
    "Fiksna stopa", "Fond za osiguranje", "Faktura za gas",
    "Fiksni i varijabilni deo", "Fond solidarnosti",
    "Servisna lista", "Stopa PDV", "Saldo obaveze",
    "Snaga i energija", "Stopa osiguranja", "Servis gasa",
    "Lista stavki", "Lizing usluge", "Lokacija objekta",
    "Električna energija i gas", "Energetska efikasnost",
    "Evidencija potrošnje", "Efikasnost sistema",
    "Godišnja stopa", "Grejanje i gas", "Godišnji obračun",
    "Gas i struja", "Gubitak energije",
    "Osnova za obračun", "Opis stavke", "Obračun gasa",
    "Zbir stavki", "Zona isporuke", "Zarada po osnovu",
    "Godišnji fond sati", "Osnov za fakturisanje",
    "FIKSNI DEO", "SNAGA", "LISTA", "ENERGIJA", "GAS", "OSNOVA", "ZONA",
    "FOND", "STOPA", "SERVIS", "GODIŠNJE", "EVIDENCIJA",
]


def random_number(template_type=None):
    t = template_type or random.choice(['price', 'date', 'percent', 'id', 'amount', 'plain'])
    if t == 'price':
        return random.choice([
            f"{random.uniform(1, 9999):.2f}",
            f"{random.randint(100, 999999):,}".replace(',', '.'),
            f"{random.uniform(0.01, 99.99):.2f} din",
            f"{random.uniform(100, 9999):.2f} Din.",
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
    elif t == 'plain':
        # Kratki brojevi kakvi se javljaju u tabelama struja računa
        return random.choice([
            str(random.randint(1, 9999)),
            f"{random.uniform(1, 999):.2f}",
            f"{random.randint(1, 500)},{random.randint(0, 9999):04d}",
            f"{random.uniform(0.1, 99.9):.4f}",
            str(random.randint(1, 365)),
        ])


def random_number_only():
    """Generiše samo numeričke labele — kratke kombinacije brojeva."""
    # 20% vremena koristi balansiranu listu koja garantuje cifre 4,6,7,8,9
    if random.random() < 0.20:
        return random.choice(TEMPLATES_NUMBERS_BALANCED)

    choice = random.random()
    if choice < 0.3:
        # Kratki broj — izbegavamo previše nula: raspon počinje od 4
        return str(random.randint(4, 9999))
    elif choice < 0.5:
        # Decimalni broj
        return f"{random.uniform(0.1, 9999.9):.2f}"
    elif choice < 0.65:
        # Srpski format sa zarezom
        return f"{random.randint(4, 9999)},{random.randint(0, 99):02d}"
    elif choice < 0.75:
        # kWh vrednost
        return f"{random.uniform(1, 999):.2f} kWh"
    elif choice < 0.83:
        # Datum — biramo mesece 4–9 češće da pokrijemo te cifre
        d = random.randint(1, 28)
        m = random.choices(range(1, 13), weights=[1,1,1,2,2,2,2,2,2,1,1,1])[0]
        y = random.randint(2020, 2026)
        return f"{d:02d}.{m:02d}.{y}."
    elif choice < 0.90:
        # Procenat
        return f"{random.uniform(0.1, 100):.2f}%"
    elif choice < 0.95:
        # Kombinacija sa дин
        return f"{random.uniform(1, 9999):.2f} дин"
    else:
        # ID broj
        return ''.join([str(random.randint(0, 9)) for _ in range(random.randint(3, 8))])


def random_text(script='cyr'):
    templates = TEMPLATES_CYR if script == 'cyr' else TEMPLATES_LAT
    choice = random.random()
    if choice < 0.35:
        return random.choice(templates)
    elif choice < 0.6:
        return f"{random.choice(templates)} {random_number()}"
    elif choice < 0.75:
        return f"{random.choice(templates)} {random_number()} {random_number()}"
    elif choice < 0.88:
        return random_number()
    else:
        return f"{random.choice(templates)} {random_number('percent')} {random_number('price')}"


def random_text_short():
    """Kratke ćirilične reči — zaglavlja tabela."""
    choice = random.random()
    if choice < 0.15:
        # Retka slova Љ Џ Ђ Ћ i latinska dijkritika Č Š Ž Đ — eksplicitno
        return random.choice(TEMPLATES_RARE_CYR)
    elif choice < 0.52:
        return random.choice(TEMPLATES_CYR_SHORT)
    elif choice < 0.72:
        return f"{random.choice(TEMPLATES_CYR_SHORT)} {random_number('price')}"
    elif choice < 0.86:
        val = random.uniform(1, 999)
        unit = random.choice(["kWh", "дин", "din", "m2", "%", "дин."])
        return f"{random.choice(TEMPLATES_CYR_SHORT)} {val:.2f} {unit}"
    else:
        return f"{random.choice(TEMPLATES_CYR_SHORT)} {random_number('plain')}"


def render_text_image(text, font_path, font_size=32, style='normal'):
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
        bg_color = random.randint(230, 255)
        text_color = random.randint(0, 60)
    elif style == 'table_header':
        bg_color = random.randint(190, 225)
        text_color = random.randint(0, 50)
    elif style == 'table_row':
        bg_color = random.randint(235, 250)
        text_color = random.randint(10, 60)
    elif style == 'dark_header':
        bg_color = random.randint(30, 80)
        text_color = random.randint(200, 255)
    else:
        bg_color = random.randint(220, 255)
        text_color = random.randint(0, 80)

    img = Image.new('L', (img_w, img_h), bg_color)

    # Papirna tekstura — šum na pozadini pre crtanja teksta
    if style != 'dark_header' and random.random() < 0.7:
        arr = np.array(img).astype(np.float32)
        grain = np.random.normal(0, random.uniform(2, 7), arr.shape)
        arr = np.clip(arr + grain, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    draw = ImageDraw.Draw(img)

    # Blago variranje boje mastila (printer neravnomerno nanosi mastilo)
    if random.random() < 0.3:
        text_color = max(0, min(255, text_color + random.randint(-15, 15)))

    draw.text((pad - bbox[0], pad - bbox[1]), text, fill=text_color, font=font)
    return np.array(img)


def add_realistic_noise(img):
    img = img.astype(np.uint8)
    h, w = img.shape[:2]

    # Rotacija
    if random.random() < 0.4:
        angle = random.uniform(-3, 3)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

    # Perspective transform — simulira kosi ugao skenera
    if random.random() < 0.4:
        margin = int(min(h, w) * 0.06)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [random.randint(0, margin), random.randint(0, margin)],
            [w - random.randint(0, margin), random.randint(0, margin)],
            [w - random.randint(0, margin), h - random.randint(0, margin)],
            [random.randint(0, margin), h - random.randint(0, margin)],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Scanner shadow — jedna strana tamnija (čest artefakt skenera)
    if random.random() < 0.35:
        intensity = random.randint(15, 45)
        if random.random() < 0.5:
            grad = np.linspace(intensity, 0, w, dtype=np.float32)
            shadow = np.tile(grad, (h, 1))
        else:
            grad = np.linspace(intensity, 0, h, dtype=np.float32)
            shadow = np.tile(grad.reshape(-1, 1), (1, w))
        img = np.clip(img.astype(np.float32) - shadow, 0, 255).astype(np.uint8)

    # Gaussov šum — širi opseg
    if random.random() < 0.55:
        sigma = random.uniform(3, 20)
        noise = np.random.normal(0, sigma, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Salt and pepper šum — izolovani pikseli (artefakti skenera)
    if random.random() < 0.35:
        amount = random.uniform(0.001, 0.012)
        sp = np.random.random(img.shape)
        img[sp < amount / 2] = 0
        img[sp > 1 - amount / 2] = 255

    # Neravnomerno osvetljenje — eliptična svetla/tamna mesta
    if random.random() < 0.35:
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        rx = random.randint(w // 3, w * 2)
        ry = random.randint(h, h * 4)
        intensity = random.randint(-30, 30)
        Y, X = np.ogrid[:h, :w]
        mask = ((X - cx) ** 2 / rx ** 2 + (Y - cy) ** 2 / ry ** 2) <= 1
        img = np.clip(img.astype(np.float32) + mask * intensity, 0, 255).astype(np.uint8)

    # Horizontalne pruge skenera
    if random.random() < 0.2:
        for _ in range(random.randint(1, 3)):
            y = random.randint(0, h - 1)
            streak = random.randint(-40, -10)
            img[y] = np.clip(img[y].astype(np.int32) + streak, 0, 255).astype(np.uint8)

    # Blur
    if random.random() < 0.4:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # Erozija/dilatacija — simulira loš printer ili razlivanje mastila
    if random.random() < 0.3:
        kernel = np.ones((2, 2), np.uint8)
        if random.random() < 0.5:
            img = cv2.erode(img, kernel, iterations=1)
        else:
            img = cv2.dilate(img, kernel, iterations=1)

    # JPEG kompresija — simulira loše skenove
    if random.random() < 0.45:
        q = random.randint(30, 85)
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
        valid_bold_fonts = valid_fonts

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

    # ── 2. Kratke ćirilične reči ──────────────────────────────────────────────
    print(f"\nGenerisanje {CONFIG['num_samples_cyr_short']} kratkih ćiriličnih slika...")
    for i in tqdm(range(CONFIG["num_samples_cyr_short"]), desc="Ćirilica (kratke)"):
        text = random_text_short()
        # Kratke reči češće bold i na sivoj pozadini
        style = random.choices(
            ['normal', 'table_header', 'table_row', 'dark_header'],
            weights=[0.2, 0.45, 0.25, 0.1]
        )[0]
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

    # ── 3. Samo brojevi i numeričke kombinacije ───────────────────────────────
    print(f"\nGenerisanje {CONFIG['num_samples_numbers']} numeričkih slika...")
    for i in tqdm(range(CONFIG["num_samples_numbers"]), desc="Brojevi"):
        text = random_number_only()
        # Brojevi se javljaju i na svetloj i na sivoj pozadini
        style = random.choices(
            ['normal', 'table_header', 'table_row'],
            weights=[0.4, 0.35, 0.25]
        )[0]
        # Brojevi su često bold ili regularni, različite veličine
        font_path = random.choices(
            [random.choice(valid_bold_fonts), random.choice(valid_fonts)],
            weights=[0.4, 0.6]
        )[0]
        font_size = random.randint(20, 38)

        img = render_text_image(text, font_path, font_size, style)
        if img is None:
            continue
        img = add_realistic_noise(img)

        filename = f"synth_num_{i:05d}.png"
        cv2.imwrite(str(output_dir / filename), img)
        records.append({
            "image_path": f"synthetic/{filename}",
            "label": text,
            "racun_id": f"synth_num_{i:05d}",
        })

    # ── 4. Latinične ──────────────────────────────────────────────────────────
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

    # ── 5. Character floor — dopuni retke karaktere ───────────────────────────
    # Za svaki karakter koji je ispod praga, generiši ciljane uzorke
    CHAR_FLOOR = 60   # minimalni broj pojavljivanja po karakteru u sintetici
    RARE_CHAR_WORDS = {
        'Љ': ["Поље", "Недеља", "Ваљево", "Краљево", "Земља", "Насеље",
              "Пољска", "Вољна", "Пољопривреда", "Биљка", "Гољ", "Кљун"],
        'љ': ["поље", "недеља", "земља", "вољна", "биљка", "насеље",
              "пољска", "пољопривреда", "кљун", "обиље", "дивљина"],
        'Џ': ["Наруџба", "Наруџбина", "Џак", "Џеп", "Џиновски"],
        'џ': ["наруџба", "наруџбина", "џак", "џеп", "наруџбеница"],
        'Ђ': ["Ђорђе", "Ђурађ", "Ђак", "Ђубриво", "Ђаво", "Ђорђевић"],
        'ђ': ["ђорђе", "ђак", "ђубриво", "ђаво", "ђорђевић"],
        'Ћ': ["Ноћна тарифа", "Ноћни период", "Кућа", "Ћирилица", "Ћошак", "Ноћ"],
        'ћ': ["ноћна", "ноћни", "кућа", "вреће", "ћирилица", "ноћ", "ћошак"],
        'Č': ["Čačak", "Čovek", "Čačanska banka", "Čist", "Član", "Čelik"],
        'č': ["čačak", "čovek", "čist", "član", "čelik", "račun", "računa"],
        'Š': ["Šabac", "Šifra usluge", "Šabačka", "Šuma", "Šinska"],
        'š': ["šifra", "šabac", "šalje", "šuma", "šinska", "utrošak", "trošak"],
        'Ž': ["Žitište", "Žiro račun", "Žuta", "Živković", "Ženska"],
        'ž': ["žiro", "žitište", "žuta", "živković", "uslužna", "već", "između"],
        'Đ': ["Đorđević", "Đakovica", "Đurđevdan", "Đak", "Đačka"],
        'đ': ["đorđević", "đakovica", "đak", "između", "priključak"],
    }

    from collections import Counter
    current_counts = Counter()
    for r in records:
        current_counts.update(r['label'])

    print(f"\nGenerisanje floor uzoraka (cilj: {CHAR_FLOOR} po karakteru)...")
    floor_count = 0
    for char, words in RARE_CHAR_WORDS.items():
        deficit = CHAR_FLOOR - current_counts.get(char, 0)
        if deficit <= 0:
            continue
        avg_char_per_word = sum(w.count(char) for w in words) / len(words)
        samples_needed = max(1, int(deficit / max(avg_char_per_word, 1)))
        for i in range(samples_needed):
            text = random.choice(words)
            font_path = random.choice(valid_bold_fonts)
            font_size = random.randint(22, 36)
            style = random.choices(
                ['normal', 'table_header', 'table_row'],
                weights=[0.5, 0.3, 0.2]
            )[0]
            img = render_text_image(text, font_path, font_size, style)
            if img is None:
                continue
            img = add_realistic_noise(img)
            filename = f"synth_floor_{floor_count:05d}.png"
            cv2.imwrite(str(output_dir / filename), img)
            records.append({
                "image_path": f"synthetic/{filename}",
                "label": text,
                "racun_id": f"synth_floor_{floor_count:05d}",
            })
            current_counts.update(text)
            floor_count += 1
    print(f"  Floor uzorci generisani: {floor_count}")

    # ── Sačuvaj CSV ───────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df.to_csv(CONFIG["output_csv"], index=False)

    n_cyr = sum(1 for r in records if r['racun_id'].startswith('synth_cyr_') and 'short' not in r['racun_id'])
    n_short = sum(1 for r in records if 'short' in r['racun_id'])
    n_num = sum(1 for r in records if r['racun_id'].startswith('synth_num_'))
    n_lat = sum(1 for r in records if r['racun_id'].startswith('synth_lat_'))
    n_floor = sum(1 for r in records if r['racun_id'].startswith('synth_floor_'))

    print(f"\n{'═'*60}")
    print(f"  GENERISANO {len(df)} SINTETIČKIH SLIKA")
    print(f"{'═'*60}")
    print(f"  Ćirilica (duge):    {n_cyr}")
    print(f"  Ćirilica (kratke):  {n_short}")
    print(f"  Brojevi:            {n_num}")
    print(f"  Latinica:           {n_lat}")
    print(f"  Floor (retki karak.): {n_floor}")
    print(f"  Ukupno:             {len(df)}")
    print(f"  CSV: {CONFIG['output_csv']}")


if __name__ == "__main__":
    main()