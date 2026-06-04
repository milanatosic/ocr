import streamlit as st
import cv2
import numpy as np
import csv
import base64
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
STATES_DIR = Path("/home/milana/Desktop/ocr/canvas_states")
CROPS_FILE = Path("/home/milana/Desktop/ocr/canvas_states/pending_crops.json")

for d in [OUTPUT_DIR, STATES_DIR]:
    d.mkdir(exist_ok=True, parents=True)

st.set_page_config(page_title="OCR Dataset Creator", layout="wide", page_icon="✂️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    .stApp { background: #0f0f0f; color: #e8e4dc; }
    .main-title {
        font-size: 2.2rem; font-weight: 800; letter-spacing: -0.03em;
        background: linear-gradient(135deg, #f0e6c8, #c8a96e);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle { font-family: 'JetBrains Mono', monospace; color: #666; font-size: 0.75rem; margin-bottom: 1rem; }
    .card { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 12px; padding: 1.2rem; margin-bottom: 0.8rem; }
    .badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace; background: #2a2a2a; color: #c8a96e; border: 1px solid #c8a96e33; }
    .stat { font-family: 'JetBrains Mono', monospace; color: #666; font-size: 0.78rem; }
    .crop-row { background: #111; border: 1px solid #2a2a2a; border-radius: 8px; padding: 0.8rem; margin-bottom: 0.6rem; }
    .stButton > button { background: #c8a96e; color: #0f0f0f; border: none;
        font-family: 'Syne', sans-serif; font-weight: 700; border-radius: 8px; padding: 0.4rem 1rem; }
    .stButton > button:hover { background: #f0e6c8; }
    .stTextInput > div > div > input { background: #1a1a1a; border: 1px solid #2a2a2a;
        color: #e8e4dc; border-radius: 8px; font-family: 'JetBrains Mono', monospace; }
    .preview-box { width: 100%; display: flex; align-items: center; justify-content: center;
        border: 1px dashed #333; border-radius: 6px; background: #111; margin: 8px 0; overflow: hidden;
        min-height: 60px; }
    .preview-box img { max-width: 100%; object-fit: contain; }
    hr { border-color: #2a2a2a; }
    label { color: #999 !important; font-size: 0.82rem !important; }
    [data-testid="stVBlockInsideContainer"] { overflow: auto !important; }
    [data-testid="stVBlockInsideContainer"]::-webkit-scrollbar { width: 8px; height: 8px; }
    [data-testid="stVBlockInsideContainer"]::-webkit-scrollbar-track { background: #111; }
    [data-testid="stVBlockInsideContainer"]::-webkit-scrollbar-thumb { background: #c8a96e; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────────────────────
for k, v in {"current_image_idx": 0, "yolo_regions": [], "saved_count": 0}.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.saved_count == 0 and CSV_PATH.exists():
    try:
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            st.session_state.saved_count = max(0, sum(1 for _ in f) - 1)
    except:
        pass

# ─── HELPER FUNKCIJE ─────────────────────────────────────────────────────────
@st.cache_resource
def load_yolo():
    return YOLO(MODEL_PATH)

def bgr_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def img_to_b64(img):
    if img is None or img.size == 0:
        return ""
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf).decode()

def state_path(racun_id, klasa):
    return STATES_DIR / f"{racun_id}__{klasa}.json"

def load_canvas_state(racun_id, klasa):
    p = state_path(racun_id, klasa)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return None
    return None

def save_canvas_state(racun_id, klasa, canvas_json):
    if canvas_json and canvas_json.get("objects"):
        with open(state_path(racun_id, klasa), "w", encoding="utf-8") as f:
            json.dump(canvas_json, f, ensure_ascii=False)

def is_done(racun_id):
    return any(STATES_DIR.glob(f"{racun_id}__*.json"))

# Crops se čuvaju NA DISKU da ne nestaju pri rerunu
def load_pending_crops():
    if CROPS_FILE.exists():
        try:
            with open(CROPS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_pending_crops(crops):
    # Čuvamo sve osim numpy array-a (img) — img čuvamo kao b64
    serializable = []
    for c in crops:
        serializable.append({
            "img_b64": c["img_b64"],
            "klasa": c["klasa"],
            "label": c["label"],
            "racun_id": c["racun_id"],
            "w": c["w"],
            "h": c["h"],
        })
    with open(CROPS_FILE, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False)

def add_pending_crop(img_bgr, klasa, label, racun_id):
    crops = load_pending_crops()
    h, w = img_bgr.shape[:2]
    crops.append({
        "img_b64": img_to_b64(img_bgr),
        "klasa": klasa,
        "label": label,
        "racun_id": racun_id,
        "w": w,
        "h": h,
    })
    save_pending_crops(crops)

def delete_pending_crop(idx):
    crops = load_pending_crops()
    if 0 <= idx < len(crops):
        crops.pop(idx)
        save_pending_crops(crops)

def clear_pending_crops():
    if CROPS_FILE.exists():
        CROPS_FILE.unlink()

def crop_from_rect(img, obj, scale):
    try:
        sx = obj.get("scaleX") or 1.0
        sy = obj.get("scaleY") or 1.0
        w = (obj.get("width", 0) * sx) / scale
        h = (obj.get("height", 0) * sy) / scale
        if w < 3 or h < 3:
            return None
        left = obj.get("left", 0) / scale
        top = obj.get("top", 0) / scale
        angle = obj.get("angle", 0)
        rad = np.radians(angle)
        ca, sa = np.cos(rad), np.sin(rad)
        p1 = [left, top]
        p2 = [left + w*ca, top + w*sa]
        p3 = [left + w*ca - h*sa, top + w*sa + h*ca]
        p4 = [left - h*sa, top + h*ca]
        src = np.array([p1, p2, p3, p4], dtype=np.float32)
        dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, M, (int(w), int(h)))
    except:
        return None

def save_csv(rows):
    exists = CSV_PATH.exists()
    with open(CSV_PATH, 'a', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(['image_path', 'label', 'racun_id'])
        for r in rows:
            w.writerow([r['path'], r['label'], r['racun_id']])

# ─── UI ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">OCR Dataset Creator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Stabilna verzija · Crops se čuvaju na disku · Nema nestajanja</div>', unsafe_allow_html=True)

yolo = load_yolo()
images = sorted(list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.jpeg")) + list(IMAGES_DIR.glob("*.png")))
total = len(images)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Statistike")
    pending = load_pending_crops()
    st.markdown(f'<div class="stat">Slika: {st.session_state.current_image_idx + 1} / {total}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="stat">Sačuvano u CSV: {st.session_state.saved_count}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="stat">Čeka na čuvanje: {len(pending)}</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown("### 🖱️ Uputstvo")
    st.markdown("""
    **Crtaj:** vuci miš da napraviš okvir  
    **Pomeri:** transform mod → klikni i vuci  
    **Rotiraj:** transform mod → kružna ručica  
    **Obriši okvir:** dugme ispod canvasa  
    **Zoom:** slider  
    """)
    st.divider()
    if CSV_PATH.exists():
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            st.download_button("⬇️ dataset.csv", f.read(), "dataset.csv", "text/csv", use_container_width=True)

if not images:
    st.error(f"Nema slika u `{IMAGES_DIR}`")
    st.stop()

# ─── NAVIGACIJA ──────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([1, 4, 1])
with c1:
    if st.button("⬅️", use_container_width=True):
        if st.session_state.current_image_idx > 0:
            st.session_state.current_image_idx -= 1
            st.session_state.yolo_regions = []
            st.rerun()
with c2:
    sel = st.selectbox("Izaberi sliku", range(total), index=st.session_state.current_image_idx,
        format_func=lambda i: f"{'✅' if is_done(images[i].stem) else '📄'} {i+1}. {images[i].name}")
    if sel != st.session_state.current_image_idx:
        st.session_state.current_image_idx = sel
        st.session_state.yolo_regions = []
        st.rerun()
with c3:
    if st.button("➡️", use_container_width=True):
        if st.session_state.current_image_idx < total - 1:
            st.session_state.current_image_idx += 1
            st.session_state.yolo_regions = []
            st.rerun()

current_path = images[st.session_state.current_image_idx]
racun_id = current_path.stem
slika_bgr = cv2.imread(str(current_path))
if slika_bgr is None:
    st.error(f"Ne mogu da učitam: {current_path.name}")
    st.stop()

st.markdown(f'<div class="badge">📄 {racun_id}</div>', unsafe_allow_html=True)
st.markdown("")

# ─── YOLO ────────────────────────────────────────────────────────────────────
if not st.session_state.yolo_regions:
    with st.spinner("YOLO detektuje regione..."):
        rez = yolo(slika_bgr, verbose=False)
        regs = []
        for box in rez[0].boxes:
            kid = int(box.cls[0])
            kime = yolo.names[kid]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            rimg = slika_bgr[y1:y2, x1:x2]
            if rimg.size > 0:
                regs.append({"img": rimg, "klasa": kime, "conf": conf})
        st.session_state.yolo_regions = regs

regions = st.session_state.yolo_regions
if not regions:
    st.warning("YOLO nije detektovao nijedan region.")
    st.stop()

# ─── IZBOR REGIONA ───────────────────────────────────────────────────────────
st.divider()
sel_idx = st.selectbox("Izaberi region", range(len(regions)),
    format_func=lambda i: f"{i+1}. {regions[i]['klasa']} ({regions[i]['conf']:.2f})")

region = regions[sel_idx]
rimg = region["img"]
rh, rw = rimg.shape[:2]
saved_canvas = load_canvas_state(racun_id, region["klasa"])

# ─── GLAVNI LAYOUT ────────────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 2])

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"**{region['klasa']}** — {rw}×{rh}px")

    # Mod i zoom
    m1, m2 = st.columns(2)
    with m1:
        mod = st.radio("Alat", ["rect", "transform"],
            format_func=lambda x: "🟥 Crtaj okvir" if x == "rect" else "🎯 Pomeri/Rotiraj",
            horizontal=False, key=f"mod_{sel_idx}")
    with m2:
        zoom = st.slider("🔍 Zoom", 0.3, 4.0, 1.0, 0.05,
                         key=f"zoom_{sel_idx}_{st.session_state.current_image_idx}")

    if mod == "rect":
        st.markdown('<div class="stat">Vuci miš da napraviš okvir oko reda teksta.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="stat">Klikni okvir → pomeri ili rotiraj kružnom ručicom.</div>', unsafe_allow_html=True)

    cw = int(rw * zoom)
    ch = int(rh * zoom)
    rimg_pil = bgr_to_pil(cv2.resize(rimg, (cw, ch), interpolation=cv2.INTER_CUBIC))

    with st.container(height=600):
        canvas_result = st_canvas(
            fill_color="rgba(200, 169, 110, 0.15)",
            stroke_width=2,
            stroke_color="#c8a96e",
            background_image=rimg_pil,
            update_streamlit=True,
            height=ch,
            width=cw,
            drawing_mode=mod,
            display_toolbar=False,
            point_display_radius=5,
            initial_drawing=saved_canvas,
            key=f"canvas_{racun_id}_{sel_idx}_{zoom}_{mod}",
        )

    # Dugme za brisanje poslednjeg okvira
    if st.button("🗑️ Obriši poslednji okvir", use_container_width=True):
        p = state_path(racun_id, region["klasa"])
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("objects"):
                    data["objects"].pop()
                    with open(p, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False)
            except:
                pass
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    # ─── PREVIEW ─────────────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Preview isečenog reda**")

    crop_preview = None
    if canvas_result.json_data:
        objs = canvas_result.json_data.get("objects", [])
        if objs:
            save_canvas_state(racun_id, region["klasa"], canvas_result.json_data)
            crop_preview = crop_from_rect(rimg, objs[-1], zoom)

    if crop_preview is not None and crop_preview.size > 0:
        ph, pw = crop_preview.shape[:2]
        b64 = img_to_b64(crop_preview)
        st.markdown(f"""
        <div class="preview-box">
            <img src="data:image/jpeg;base64,{b64}" />
        </div>
        <div class="stat">{pw}×{ph}px</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="preview-box" style="min-height:80px">
            <div class="stat" style="text-align:center;padding:1rem">
                Nacrtaj okvir na slici levo<br>da vidiš preview ovde
            </div>
        </div>
        """, unsafe_allow_html=True)

    label = st.text_input("Tekst koji piše u redu", placeholder="Unesi tekst...", key="label_inp")

    if st.button("➕ Dodaj red u listu", use_container_width=True):
        if crop_preview is not None and crop_preview.size > 0 and label.strip():
            add_pending_crop(crop_preview, region["klasa"], label.strip(), racun_id)
            st.success("✅ Dodat red!")
            st.rerun()
        elif not label.strip():
            st.warning("⚠️ Unesi tekst.")
        else:
            st.warning("⚠️ Nacrtaj okvir na slici.")

    st.markdown('</div>', unsafe_allow_html=True)

    # ─── LISTA REDOVA (čita se sa diska, nikad ne nestaje) ───────────────────
    pending_crops = load_pending_crops()

    if pending_crops:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"**Redovi koji čekaju na čuvanje** ({len(pending_crops)})")

        for i, crop in enumerate(pending_crops):
            st.markdown('<div class="crop-row">', unsafe_allow_html=True)
            c_img, c_info, c_del = st.columns([2, 3, 1])
            with c_img:
                st.markdown(f"""
                <div class="preview-box" style="min-height:50px">
                    <img src="data:image/jpeg;base64,{crop['img_b64']}" />
                </div>
                """, unsafe_allow_html=True)
            with c_info:
                st.markdown(f'<div class="badge">{crop["klasa"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="stat" style="margin-top:4px">"{crop["label"]}"</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="stat">{crop["w"]}×{crop["h"]}px · {crop["racun_id"]}</div>', unsafe_allow_html=True)
            with c_del:
                if st.button("🗑️", key=f"del_{i}"):
                    delete_pending_crop(i)
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        if st.button("💾 Sačuvaj sve u CSV", use_container_width=True):
            to_save = []
            for i, crop in enumerate(pending_crops):
                name = f"{crop['racun_id']}_{crop['klasa']}_r{i}.jpg"
                img_data = base64.b64decode(crop["img_b64"])
                img_arr = np.frombuffer(img_data, dtype=np.uint8)
                img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    cv2.imwrite(str(OUTPUT_DIR / name), img_bgr)
                    to_save.append({
                        "path": f"output_rows/{name}",
                        "label": crop["label"],
                        "racun_id": crop["racun_id"],
                    })
            save_csv(to_save)
            st.session_state.saved_count += len(to_save)
            clear_pending_crops()
            st.success(f"✅ Sačuvano {len(to_save)} redova u CSV!")
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)