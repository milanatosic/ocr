import streamlit as st
import cv2
import numpy as np
import csv
import base64
import io
import json
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ─── PODEŠAVANJA ─────────────────────────────────────────────────────────────
MODEL_PATH = "/home/milana/Desktop/ocr/best.pt"
IMAGES_DIR = Path("/home/milana/Desktop/ocr/originalne_slike")
OUTPUT_DIR = Path("/home/milana/Desktop/ocr/output_rows")
CSV_PATH = Path("/home/milana/Desktop/ocr/dataset.csv")

# Novi folder u kom čuvamo tačne koordinate nacrtanih oblika na disku
STATES_DIR = Path("/home/milana/Desktop/ocr/canvas_states")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
STATES_DIR.mkdir(exist_ok=True, parents=True)

st.set_page_config(page_title="OCR Dataset Creator", layout="wide", page_icon="✂️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    .stApp { background: #0f0f0f; color: #e8e4dc; }
    .main-title {
        font-size: 2.5rem; font-weight: 800; letter-spacing: -0.03em;
        background: linear-gradient(135deg, #f0e6c8, #c8a96e);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle { font-family: 'JetBrains Mono', monospace; color: #666; font-size: 0.8rem; margin-bottom: 2rem; }
    .card { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; }
    .badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace; background: #2a2a2a; color: #c8a96e; border: 1px solid #c8a96e33; }
    .stat { font-family: 'JetBrains Mono', monospace; color: #666; font-size: 0.8rem; }
    .crop-row { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px; padding: 1rem; margin-bottom: 0.8rem; }
    .stButton > button { background: #c8a96e; color: #0f0f0f; border: none;
        font-family: 'Syne', sans-serif; font-weight: 700; border-radius: 8px; padding: 0.5rem 1.5rem; }
    .stButton > button:hover { background: #f0e6c8; }
    .stTextInput > div > div > input { background: #1a1a1a; border: 1px solid #2a2a2a;
        color: #e8e4dc; border-radius: 8px; font-family: 'JetBrains Mono', monospace; }
    .progress-bar { background: #2a2a2a; border-radius: 4px; height: 6px; margin: 0.5rem 0; }
    .progress-fill { background: linear-gradient(90deg, #c8a96e, #f0e6c8); height: 100%; border-radius: 4px; }
    hr { border-color: #2a2a2a; }
    label { color: #999 !important; font-size: 0.85rem !important; }

    [data-testid="stVerticalBlock"] > div {
        overflow-x: auto !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────────────────────
for key, default in {
    "current_image_idx": 0,
    "crops": [],
    "saved_count": 0,
    "yolo_regions": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Brojanje već sačuvanih redova u CSV-u prilikom pokretanja aplikacije
if st.session_state.saved_count == 0 and CSV_PATH.exists():
    try:
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            st.session_state.saved_count = sum(1 for _ in f) - 1
    except:
        pass

# ─── HELPER FUNKCIJE ZASNOVANE NA DISKU ────────────────────────────────────────
@st.cache_resource
def load_yolo():
    return YOLO(MODEL_PATH)

def bgr_to_pil(img_bgr):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def perspektivni_crop(region, pts):
    pts = np.array(pts, dtype=np.float32)
    w1 = np.linalg.norm(pts[1] - pts[0])
    w2 = np.linalg.norm(pts[2] - pts[3])
    h1 = np.linalg.norm(pts[3] - pts[0])
    h2 = np.linalg.norm(pts[2] - pts[1])
    W = int(max(w1, w2))
    H = int(max(h1, h2))
    if W == 0 or H == 0:
        return None
    dst = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(region, M, (W, H))

def sacuvaj_csv(rows_data):
    csv_exists = CSV_PATH.exists()
    with open(CSV_PATH, mode='a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(['image_path', 'label', 'racun_id'])
        for row in rows_data:
            writer.writerow([row['path'], row['label'], row['racun_id']])

# Funkcije za trajno pamćenje geometrije sa platna
def get_state_file_path(racun_id, klasa_ime):
    return STATES_DIR / f"{racun_id}_{klasa_ime}_state.json"

def ucitaj_geometriju_platna(racun_id, klasa_ime):
    putanja = get_state_file_path(racun_id, klasa_ime)
    if putanja.exists():
        try:
            with open(putanja, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return None
    return None

def sacuvaj_geometriju_platna(racun_id, klasa_ime, json_data):
    putanja = get_state_file_path(racun_id, klasa_ime)
    if json_data and "objects" in json_data and json_data["objects"]:
        with open(putanja, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
    elif putanja.exists():
        # Ako je korisnik obrisao sve oblike, brišemo i fajl stanja sa diska
        putanja.unlink()

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">OCR Dataset Creator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Sve nacrtano se automatski čuva i učitava pri sledećem ulasku! 💾</div>', unsafe_allow_html=True)

yolo = load_yolo()
images = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.jpeg")) + list(IMAGES_DIR.glob("*.png"))
total = len(images)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Status")
    st.markdown("✅ Pamćenje napretka: **UKLJUČENO**")
    st.divider()
    st.markdown("### 📊 Statistike")
    st.markdown(f'<div class="stat">Slika: {st.session_state.current_image_idx + 1}/{total}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="stat">Ukupno u CSV-u: {st.session_state.saved_count} redova</div>', unsafe_allow_html=True)
    if total > 0:
        pct = int((st.session_state.current_image_idx + 1) / total * 100)
        st.markdown(f'<div class="progress-bar"><div class="progress-fill" style="width:{pct}%"></div></div>', unsafe_allow_html=True)
    st.divider()
    st.markdown("### 💾 Kako radi auto-save?")
    st.markdown("""
    - Čim nacrtaš ili izmeniš oblik, aplikacija kreira `.json` fajl za tu regiju.
    - Možeš slobodno ugasiti terminal, restartovati računar ili promeniti sliku.
    - Kada se vratiš na istu sliku, **sve te čeka onako kako si ostavila**.
    """)
    st.divider()
    if CSV_PATH.exists():
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            st.download_button("⬇️ Preuzmi dataset.csv", f.read(), "dataset.csv", "text/csv", use_container_width=True)

if not images:
    st.error(f"Nema slika u `{IMAGES_DIR}`")
    st.stop()

# ─── NAVIGACIJA ──────────────────────────────────────────────────────────────
col_prev, col_sel, col_next = st.columns([1, 4, 1])
with col_prev:
    if st.button("⬅️", use_container_width=True):
        if st.session_state.current_image_idx > 0:
            st.session_state.current_image_idx -= 1
            st.session_state.yolo_regions = []
            st.session_state.crops = []
            st.rerun()
with col_sel:
    sel = st.selectbox("Izaberi sliku", range(total),
        index=st.session_state.current_image_idx,
        format_func=lambda i: f"{i+1}. {images[i].name}")
    if sel != st.session_state.current_image_idx:
        st.session_state.current_image_idx = sel
        st.session_state.yolo_regions = []
        st.session_state.crops = []
        st.rerun()
with col_next:
    if st.button("➡️", use_container_width=True):
        if st.session_state.current_image_idx < total - 1:
            st.session_state.current_image_idx += 1
            st.session_state.yolo_regions = []
            st.session_state.crops = []
            st.rerun()

current_path = images[st.session_state.current_image_idx]
racun_id = current_path.stem
slika_bgr = cv2.imread(str(current_path))
if slika_bgr is None:
    st.error(f"Ne mogu da učitam: {current_path}")
    st.stop()

st.markdown(f'<div class="badge">📄 racun_id: {racun_id}</div>', unsafe_allow_html=True)
st.markdown("")

# ─── YOLO DETEKCIJA ──────────────────────────────────────────────────────────
if not st.session_state.yolo_regions:
    with st.spinner("YOLO detektuje regione..."):
        rez = yolo(slika_bgr, verbose=False)
        regions = []
        for box in rez[0].boxes:
            klasa_id = int(box.cls[0])
            klasa_ime = yolo.names[klasa_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            region_img = slika_bgr[y1:y2, x1:x2]
            if region_img.size > 0:
                regions.append({"img": region_img, "klasa": klasa_ime, "conf": conf})
        st.session_state.yolo_regions = regions

regions = st.session_state.yolo_regions
if not regions:
    st.warning("YOLO nije detektovao nijedan region.")
    st.stop()

# ─── IZBOR REGIONA ───────────────────────────────────────────────────────────
st.divider()
region_options = [f"{i+1}. {r['klasa']} (conf: {r['conf']:.2f})" for i, r in enumerate(regions)]
selected_idx = st.selectbox("Izaberi region za anotaciju", range(len(regions)),
    format_func=lambda i: region_options[i])
region = regions[selected_idx]
region_img = region["img"]
rh, rw = region_img.shape[:2]

# Učitavanje starog stanja (ako postoji na disku) za ovu konkretnu kombinaciju slike i regije
prethodno_stanje = ucitaj_geometriju_platna(racun_id, region["klasa"])

# ─── CANVAS I KONTROLE ───────────────────────────────────────────────────────
col_left, col_right = st.columns([7, 5])

with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"**Region:** `{region['klasa']}` — {rw}×{rh}px")

    c_tool, c_zoom = st.columns([2, 2])
    with c_tool:
        drawing_mode = st.radio("Izaberi mod rada:", ["polygon", "rect", "transform"],
            format_func=lambda x: "🔷 Novi Poligon" if x == "polygon" else ("⬜ Novi Pravougaonik" if x == "rect" else "📐 Pomeri / Izmeni oblik"),
            horizontal=False)
    with c_zoom:
        zoom_factor = st.slider("🔍 Stepen uvećanja (Zoom):", min_value=1.0, max_value=5.0, value=2.0, step=0.1)

    MAX_W = 1000
    base_scale = min(MAX_W / rw, 4.0)
    final_scale = base_scale * zoom_factor
    
    canvas_w = int(rw * final_scale)
    canvas_h = int(rh * final_scale)

    region_resized = cv2.resize(region_img, (canvas_w, canvas_h), interpolation=cv2.INTER_CUBIC)
    region_pil = bgr_to_pil(region_resized)

    # Otvaramo scroll-box kontejner fiksne visine
    with st.container(height=600):
        canvas_result = st_canvas(
            fill_color="rgba(200, 169, 110, 0.15)",
            stroke_width=2,
            stroke_color="#c8a96e",
            background_image=region_pil,
            update_streamlit=True,
            height=canvas_h,
            width=canvas_w,
            drawing_mode=drawing_mode,
            display_toolbar=True,
            point_display_radius=5,
            # Prosleđujemo prethodno_stanje koje smo pročitali iz JSON fajla!
            initial_drawing=prethodno_stanje,
            key=f"canvas_{selected_idx}_{st.session_state.current_image_idx}_z_{zoom_factor}",
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# AUTOMATSKO ČUVANJE NA DISK: Čim se registruje bilo kakva promena geometrije na ekranu
if canvas_result.json_data is not None:
    sacuvaj_geometriju_platna(racun_id, region["klasa"], canvas_result.json_data)

with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Preview i unos podataka**")

    crop_preview = None
    crop_pts = None

    if canvas_result.json_data is not None:
        objects = canvas_result.json_data.get("objects", [])
        if objects:
            last_obj = objects[-1]
            obj_type = last_obj.get("type", "")

            if obj_type == "polygon":
                raw_pts = last_obj.get("path", [])
                pts = []
                for cmd in raw_pts:
                    if cmd[0] in ("M", "L") and len(cmd) >= 3:
                        pts.append([cmd[1] / final_scale, cmd[2] / final_scale])
                if len(pts) >= 3:
                    if len(pts) == 4:
                        crop_pts = pts
                        crop_preview = perspektivni_crop(region_img, pts)
                    else:
                        pts_arr = np.array(pts)
                        x1c, y1c = pts_arr.min(axis=0).astype(int)
                        x2c, y2c = pts_arr.max(axis=0).astype(int)
                        x1c, y1c = max(0, x1c), max(0, y1c)
                        x2c, y2c = min(rw, x2c), min(rh, y2c)
                        crop_preview = region_img[y1c:y2c, x1c:x2c]

            elif obj_type == "rect":
                scale_x = last_obj.get("scaleX", 1.0)
                scale_y = last_obj.get("scaleY", 1.0)
                left = (last_obj.get("left", 0)) / final_scale
                top = (last_obj.get("top", 0)) / final_scale
                w_r = (last_obj.get("width", 0) * scale_x) / final_scale
                h_r = (last_obj.get("height", 0) * scale_y) / final_scale
                
                x1c, y1c = int(left), int(top)
                x2c, y2c = int(left + w_r), int(top + h_r)
                x1c, y1c = max(0, x1c), max(0, y1c)
                x2c, y2c = min(rw, x2c), min(rh, y2c)
                if x2c > x1c and y2c > y1c:
                    crop_preview = region_img[y1c:y2c, x1c:x2c]

    if crop_preview is not None and crop_preview.size > 0:
        st.markdown("**Trenutni izrez (Sirov kvalitet):**")
        st.image(bgr_to_pil(crop_preview), use_container_width=True)
    else:
        st.markdown('<div class="stat">Nacrtaj oblik ili ga izmeni da generišeš preview.</div>', unsafe_allow_html=True)

    label_input = st.text_input("Tekst koji piše u ovom redu", placeholder="Unesi tekst...", key="label_inp")

    if st.button("➕ Dodaj red", use_container_width=True):
        if crop_preview is not None and crop_preview.size > 0 and label_input.strip():
            st.session_state.crops.append({
                "img": crop_preview.copy(),
                "klasa": region["klasa"],
                "label": label_input.strip(),
                "racun_id": racun_id,
            })
            st.success(f"✅ Dodat red: '{label_input.strip()}'")
            st.rerun()
        elif not label_input.strip():
            st.warning("⚠️ Unesi tekst pre dodavanja.")
        else:
            st.warning("⚠️ Obeleži tekst na slici pre dodavanja.")

    st.markdown('</div>', unsafe_allow_html=True)

    # ─── LISTA REDOVA ZA ČUVANJE ──────────────────────────────────────────────
    if st.session_state.crops:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"**Redovi spremni za snimanje** ({len(st.session_state.crops)})")

        for i, crop in enumerate(st.session_state.crops):
            st.markdown('<div class="crop-row">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([2, 3, 1])
            with c1:
                st.image(bgr_to_pil(crop["img"]), use_container_width=True)
            with c2:
                st.markdown(f'<div class="badge">{crop["klasa"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="stat" style="margin-top:4px">"{crop["label"]}"</div>', unsafe_allow_html=True)
            with c3:
                if st.button("🗑️", key=f"del_{i}"):
                    st.session_state.crops.pop(i)
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.divider()
        if st.button("💾 Sačuvaj sve u CSV", use_container_width=True):
            rows_to_save = []
            for i, crop in enumerate(st.session_state.crops):
                row_name = f"{crop['racun_id']}_{crop['klasa']}_r{i}.jpg"
                cv2.imwrite(str(OUTPUT_DIR / row_name), crop["img"])
                rows_to_save.append({
                    "path": f"output_rows/{row_name}",
                    "label": crop["label"],
                    "racun_id": crop["racun_id"],
                })
            sacuvaj_csv(rows_to_save)
            st.session_state.saved_count += len(rows_to_save)
            st.session_state.crops = []
            st.success(f"✅ Uspešno upisano u CSV!")
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)