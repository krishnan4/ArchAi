# ============================================================
# ArchAI Dashboard — v4.2 (Overlay Fix) + Auto Model Download
# Fix 1: Mound detection now captures ALL objects from image, not just mound labels
# Fix 2: Deforestation tab NoneType error fixed — rgb stored in session state,
#         guarded against None, upload re-read handled correctly
# Fix 3: Sidebar always visible — CSS forces sidebar open, collapse arrow always shown
# Fix 4: draw_mound_overlay — Natural/Uncertain boxes now always visible on overlay
# Fix 5: Auto-download YOLO model from Google Drive if not present
# Run: streamlit run dashboard_app.py
# ============================================================
import io
import os
import cv2
import json
import zipfile
import requests
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ====================== AUTO DOWNLOAD YOLO MODEL ======================
import gdown

MODEL_PATH = "model/best.pt"

def download_yolo_model():
    """Download YOLO model from Google Drive if not already present."""
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}")
        return True
    
    # Create model directory if it doesn't exist
    os.makedirs("model", exist_ok=True)
    
    # Google Drive file ID for the YOLO model
    url = "https://drive.google.com/uc?id=1p5IdB-Ypc7X0RBs6KPrVcmlINAZSzzwK"
    
    try:
        print("Downloading YOLO model from Google Drive...")
        st.info("Downloading YOLO model (this may take a few minutes on first run)...")
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Download complete!")
        st.success("YOLO model downloaded successfully!")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        st.warning(f"Could not download model: {e}. Running in demo mode.")
        return False

# Download model on startup
download_yolo_model()

# ====================== GROQ API KEY ======================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

try:
    from groq import Groq
except ImportError:
    GROQ_AVAILABLE = False
else:
    GROQ_AVAILABLE = True

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="ArchAI — Archaeological Intelligence Platform",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====================== SESSION INIT ======================
for k, v in [
    ("theme", "Dark"),
    ("dets", []),
    ("risk", 0.0),
    ("vari_mean", 0.0),
    ("coverage", {}),
    ("lat", 20.5937),
    ("lon", 78.9629),
    ("location_name", ""),
    ("geo_msg", ""),
    ("auto_terrain", {"slope": 0.0, "elevation": 0.0, "confidence": 0.0}),
    ("mound_results", []),
    ("deforest_results", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ====================== THEME DEFINITIONS ======================
THEMES = {
    "Dark": {
        "--bg-primary":     "#08090d",
        "--bg-secondary":   "#0e1118",
        "--bg-card":        "#111520",
        "--bg-elevated":    "#161c2a",
        "--border":         "#1e2535",
        "--border-light":   "#252f45",
        "--accent":         "#b8966e",
        "--accent-dim":     "#7a5f42",
        "--text-primary":   "#d4cfc8",
        "--text-secondary": "#7a8399",
        "--text-muted":     "#4a5268",
        "--risk-low-fg":    "#7ec899",
        "--risk-mod-fg":    "#d4a84b",
        "--risk-high-fg":   "#d46b6b",
    },
    "Light": {
        "--bg-primary":     "#f5f7fa",
        "--bg-secondary":   "#ffffff",
        "--bg-card":        "#ffffff",
        "--bg-elevated":    "#eef1f7",
        "--border":         "#e2e8f0",
        "--border-light":   "#cbd5e0",
        "--accent":         "#8b6f47",
        "--accent-dim":     "#a07850",
        "--text-primary":   "#1a202c",
        "--text-secondary": "#4a5568",
        "--text-muted":     "#718096",
        "--risk-low-fg":    "#276749",
        "--risk-mod-fg":    "#975a16",
        "--risk-high-fg":   "#c53030",
    },
}

def apply_theme():
    t = st.session_state.get("theme", "Dark")
    vars_css = "\n".join(f"        {k}: {v};" for k, v in THEMES[t].items())
    st.markdown(f"""
<style>
:root {{
{vars_css}
}}
</style>
""", unsafe_allow_html=True)

apply_theme()

# ====================== GLOBAL STYLES ======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400;500&family=Archivo+Narrow:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Archivo Narrow', sans-serif;
    color: var(--text-primary);
    background-color: var(--bg-primary);
}

/* ===== SIDEBAR FIX: Always keep sidebar visible & collapse arrow always shown ===== */
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; }

[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
    pointer-events: auto !important;
}

[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    transform: none !important;
    min-width: 21rem !important;
    max-width: 21rem !important;
    transition: none !important;
}

[data-testid="stSidebar"][aria-expanded="false"] {
    margin-left: 0 !important;
    transform: translateX(0) !important;
}

.css-1d391kg, section[data-testid="stSidebarContent"] {
    display: block !important;
    visibility: visible !important;
}
/* ===== END SIDEBAR FIX ===== */

.stDeployButton { display: none; }
.block-container { padding-top: 1.5rem !important; max-width: 1400px; }

[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {
    font-family: 'Archivo Narrow', sans-serif !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
}

.sidebar-brand {
    font-family: 'Cormorant Garamond', serif;
    font-size: 26px;
    font-weight: 300;
    color: var(--accent);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2px;
}
.sidebar-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.sidebar-rule {
    height: 1px;
    background: linear-gradient(90deg, var(--accent-dim), transparent);
    margin: 0.8rem 0 1rem;
}

.stTextInput input, .stNumberInput input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 4px !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    padding: 8px 12px !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--accent-dim) !important;
    box-shadow: 0 0 0 1px var(--accent-dim) !important;
}

[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid var(--border-light) !important;
    color: var(--text-secondary) !important;
    font-family: 'Archivo Narrow', sans-serif !important;
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 3px !important;
    padding: 7px 14px !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: var(--accent-dim) !important;
    color: var(--accent) !important;
    background: rgba(184,150,110,0.05) !important;
    transform: none !important;
    box-shadow: none !important;
}

section[data-testid="stMain"] .stButton > button {
    background: linear-gradient(135deg, var(--bg-elevated), var(--bg-secondary)) !important;
    border: 1px solid var(--accent-dim) !important;
    color: var(--accent) !important;
    font-family: 'Archivo Narrow', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    border-radius: 3px !important;
    padding: 10px 28px !important;
    width: 100% !important;
    transition: all 0.25s !important;
    box-shadow: 0 2px 20px rgba(184,150,110,0.08) !important;
}
section[data-testid="stMain"] .stButton > button:hover {
    border-color: var(--accent) !important;
    box-shadow: 0 4px 30px rgba(184,150,110,0.18) !important;
    transform: translateY(-1px) !important;
}

.page-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 38px;
    font-weight: 300;
    color: var(--accent);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    line-height: 1;
    margin-bottom: 3px;
}
.page-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--text-muted);
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}
.title-bar {
    border-bottom: 1px solid var(--border);
    padding-bottom: 1rem;
    margin-bottom: 1.5rem;
}

h1, h2, h3 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 400 !important;
    color: var(--text-primary) !important;
    letter-spacing: 0.04em !important;
}
h2 { font-size: 22px !important; border-bottom: 1px solid var(--border); padding-bottom: 8px; margin-bottom: 1rem !important; }
h3, h4 { font-size: 16px !important; color: var(--text-secondary) !important; }

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Archivo Narrow', sans-serif !important;
    font-size: 11px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    background: transparent !important;
    border: none !important;
    padding: 10px 20px !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }

[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 14px 16px !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 24px !important;
    color: var(--text-primary) !important;
}

.stDataFrame { border: 1px solid var(--border) !important; border-radius: 4px !important; }
.stAlert { border-radius: 3px !important; border-left-width: 3px !important;
           font-family: 'Archivo Narrow', sans-serif !important; font-size: 13px !important; }

.stProgress > div > div { background: var(--bg-card) !important; border-radius: 2px !important; }
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent-dim), var(--accent)) !important;
    border-radius: 2px !important;
}

.streamlit-expanderHeader {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    color: var(--text-secondary) !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
}

hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }

[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border-light) !important;
    border-radius: 4px !important;
}

.stCaption, small {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.05em !important;
}

.sidebar-section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 1.2rem 0 0.5rem;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
}

[data-testid="stSidebar"] .stRadio label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    color: var(--text-secondary) !important;
    text-transform: none !important;
    letter-spacing: 0.05em !important;
}

.mound-card {
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 12px 14px;
    margin-bottom: 8px;
    background: var(--bg-card);
}
.mound-card.manmade { border-left: 3px solid #d46b6b; }
.mound-card.natural { border-left: 3px solid #7ec899; }
.mound-card.uncertain { border-left: 3px solid #d4a84b; }
.mound-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.mound-val {
    font-family: 'Cormorant Garamond', serif;
    font-size: 17px;
    color: var(--text-primary);
}
</style>
""", unsafe_allow_html=True)

# ====================== GEOCODING ======================
HEADERS = {"User-Agent": "ArchAI-Dashboard/1.0"}

def _try_nominatim(query):
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1, "addressdetails": 1, "accept-language": "en"},
            headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data:
            best = data[0]
            short = ", ".join(best.get("display_name", query).split(", ")[:3])
            return float(best["lat"]), float(best["lon"]), short
    except Exception:
        pass
    return None, None, ""

def _try_nominatim_structured(village, state):
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"village": village.strip(), "state": state.strip(), "country": "India",
                    "format": "json", "limit": 1, "accept-language": "en"},
            headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data:
            best = data[0]
            short = ", ".join(best.get("display_name", village).split(", ")[:3])
            return float(best["lat"]), float(best["lon"]), short
    except Exception:
        pass
    return None, None, ""

def _try_photon(query):
    try:
        r = requests.get("https://photon.komoot.io/api/",
                         params={"q": query, "limit": 1, "lang": "en"},
                         headers=HEADERS, timeout=10)
        r.raise_for_status()
        feats = r.json().get("features", [])
        if feats:
            props = feats[0].get("properties", {})
            coords = feats[0]["geometry"]["coordinates"]
            parts = [props.get(k, "") for k in ("name", "state", "country") if props.get(k)]
            return float(coords[1]), float(coords[0]), ", ".join(parts) or query
    except Exception:
        pass
    return None, None, ""

def geocode_location(location_name):
    import re
    if not location_name.strip():
        return None, None, "Please enter a location name."
    raw = re.sub(r"\s+", " ", re.sub(r",\s*", ", ", location_name.strip()))
    for fn, arg in [
        (_try_nominatim, raw),
        (_try_nominatim, "" if "india" in raw.lower() else f"{raw}, India"),
        (_try_photon, raw),
        (_try_photon, f"{raw} India"),
    ]:
        if not arg:
            continue
        lat, lon, disp = fn(arg)
        if lat is not None:
            return lat, lon, f"Located: {disp}"
    tokens = [p.strip() for p in raw.split(",") if p.strip()]
    if len(tokens) >= 2:
        lat, lon, disp = _try_nominatim_structured(tokens[0], tokens[-1])
        if lat is not None:
            return lat, lon, f"Located: {disp}"
    first = tokens[0] if tokens else raw.split()[0]
    for fn, arg in [(_try_nominatim, f"{first}, India"), (_try_photon, f"{first} India")]:
        lat, lon, disp = fn(arg)
        if lat is not None:
            return lat, lon, f"Located: {disp} (nearest for '{first}')"
    skip = {"india", "maharashtra", "karnataka", "tamilnadu", "tamil nadu",
            "gujarat", "rajasthan", "uttarpradesh", "uttar pradesh", "madhya pradesh"}
    for t in tokens:
        if len(t) > 3 and t.lower() not in skip:
            lat, lon, disp = _try_photon(f"{t} India")
            if lat is not None:
                return lat, lon, f"Located: {disp} (matched on '{t}')"
    return None, None, (
        f"Could not locate '{location_name}'. "
        "Try adding district/state, e.g. 'Daimabad, Ahmednagar, Maharashtra, India', "
        "or enter coordinates manually."
    )

# ====================== MODEL LOADERS ======================
@st.cache_resource(show_spinner="Loading detection model…")
def load_local_yolo(path):
    if not path or not Path(path.strip()).exists():
        return None, None
    try:
        from ultralytics import YOLO
        return YOLO(str(Path(path.strip()))), "local"
    except ImportError:
        st.warning("ultralytics not installed — run: pip install ultralytics")
        return None, None
    except Exception as e:
        st.warning(f"Could not load YOLO model: {e}")
        return None, None

@st.cache_resource(show_spinner="Loading erosion model…")
def load_erosion_model(path):
    default = ["slope", "elevation", "ndvi", "curvature", "twi", "tex_var", "dist_water"]
    if not path or not Path(path.strip()).exists():
        return None, default
    try:
        import joblib
        bundle = joblib.load(str(Path(path.strip())))
        return bundle["model"], bundle.get("features", default)
    except Exception as e:
        st.warning(f"Could not load erosion model: {e}. Using formula fallback.")
        return None, default

# ====================== IMAGE PROCESSING ======================
def run_detection(img_bgr, model, mode, confidence=40):
    if model is None:
        rgb = cv2.cvtColor(img_bgr.copy(), cv2.COLOR_BGR2RGB)
        overlay = rgb.copy()
        cv2.rectangle(overlay, (5, 5), (450, 45), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, rgb, 0.5, 0, rgb)
        cv2.putText(rgb, "Demo Mode — No model loaded", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 180, 120), 2)
        return rgb, []
    try:
        results = model(img_bgr, conf=max(0.05, min(0.95, confidence / 100.0)), verbose=False)
        dets = []
        annotated = img_bgr
        for r in results:
            annotated = r.plot()
            if r.boxes is None:
                continue
            for box, c, cls in zip(r.boxes.xyxy.cpu().numpy(),
                                   r.boxes.conf.cpu().numpy(),
                                   r.boxes.cls.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                dets.append({
                    "label": r.names[int(cls)],
                    "conf": round(float(c), 3),
                    "bbox": [x1, y1, x2, y2],
                    "cx": (x1 + x2) // 2,
                    "cy": (y1 + y2) // 2,
                })
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), dets
    except Exception as e:
        st.error(f"Detection failed: {e}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), []

def compute_vari(img_rgb):
    f = img_rgb.astype(np.float32)
    r, g, b = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    denom = np.where(np.abs(g + r - b) < 1e-6, 1e-6, g + r - b)
    return np.clip((g - r) / denom, -1, 1)

def colorise_vari(vari):
    return cv2.cvtColor(
        cv2.applyColorMap(((vari + 1) / 2 * 255).astype(np.uint8), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )

def segment_vegetation(vari):
    seg = np.zeros((*vari.shape, 3), dtype=np.uint8)
    layers = {
        "Very Dense": (vari > 0.7,                     (0, 100, 0)),
        "Dense":      ((vari > 0.5) & (vari <= 0.7),   (76, 175, 80)),
        "Moderate":   ((vari > 0.35) & (vari <= 0.5),  (255, 193, 7)),
        "Sparse":     ((vari > 0.2) & (vari <= 0.35),  (255, 152, 0)),
        "Bare Soil":  (vari <= 0.2,                    (161, 136, 127)),
    }
    total = vari.size
    cov = {}
    for lbl, (mask, col) in layers.items():
        seg[mask] = col
        cov[lbl] = round(mask.sum() / total * 100, 1)
    return seg, cov

def predict_erosion_score(model, feat_names, slope, elevation, ndvi):
    if model is None:
        return float(np.clip(
            0.50 * np.clip(slope / 50, 0, 1) +
            0.35 * np.clip(1 - ((ndvi + 1) / 2), 0, 1) +
            0.15 * np.clip(elevation / 2000, 0, 1),
            0, 1,
        ))
    row = {f: 0.0 for f in feat_names}
    row.update({"slope": slope, "elevation": elevation, "ndvi": ndvi,
                "twi": 5.0, "tex_var": 50.0, "dist_water": 100.0, "curvature": 0.0})
    X = np.array([[row[f] for f in feat_names]])
    try:
        n = getattr(model, "n_features_in_", X.shape[1])
        return float(np.clip(model.predict(X[:, :n])[0], 0, 1))
    except Exception:
        return predict_erosion_score(None, feat_names, slope, elevation, ndvi)

def auto_detect_terrain(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    grad = np.sqrt(sx ** 2 + sy ** 2)
    slope = float(np.clip((grad.mean() / (float(np.percentile(grad, 95)) or 1.0)) * 50.0, 0, 50))
    bright = float(gray.mean()) / 255.0
    tex = float(np.clip(cv2.Laplacian(gray, cv2.CV_32F).var() / 2000.0, 0, 1))
    dark = float((gray < 60).sum()) / gray.size
    blue_d = float(np.clip(
        img_rgb[:, :, 2].astype(np.float32).mean() / 255.0 -
        img_rgb[:, :, 0].astype(np.float32).mean() / 255.0, 0, 1,
    ))
    elev = float(np.clip((0.30 * bright + 0.30 * tex + 0.25 * dark + 0.15 * blue_d) * 2000.0, 0, 2000))
    conf = float(np.clip(float(gray.std()) / 128.0, 0.1, 1.0))
    return {"slope": round(slope, 1), "elevation": round(elev, 0), "confidence": round(conf, 2)}


# ====================== MOUND DETECTION MODULE ======================
def detect_mound_candidates(img_bgr, model, confidence_threshold=40):
    if model is not None:
        try:
            results = model(img_bgr, conf=max(0.05, confidence_threshold / 100.0), verbose=False)
            dets = []
            for r in results:
                if r.boxes is None:
                    continue
                for box, c, cls in zip(r.boxes.xyxy.cpu().numpy(),
                                       r.boxes.conf.cpu().numpy(),
                                       r.boxes.cls.cpu().numpy()):
                    x1, y1, x2, y2 = map(int, box)
                    dets.append({
                        "label": r.names[int(cls)],
                        "conf":  round(float(c), 3),
                        "bbox":  [x1, y1, x2, y2],
                        "cx":    (x1 + x2) // 2,
                        "cy":    (y1 + y2) // 2,
                    })
            return dets
        except Exception:
            pass

    h, w = img_bgr.shape[:2]
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (21, 21), 0)
    lap   = cv2.Laplacian(blur.astype(np.float32), cv2.CV_32F)
    lap_u8 = cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, thresh = cv2.threshold(lap_u8, int(np.percentile(lap_u8, 80)), 255, cv2.THRESH_BINARY)
    n_labels, label_map, stats_cc, centroids = cv2.connectedComponentsWithStats(thresh)

    dets = []
    label_pool = ["ruins", "mound", "structure", "earthwork", "mound"]
    np.random.seed(12)

    for i in range(1, min(n_labels, 16)):
        x, y, bw, bh, area = stats_cc[i]
        if area < 200:
            continue
        pad_x = max(int(bw * 0.2), 8)
        pad_y = max(int(bh * 0.2), 8)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)
        lbl  = label_pool[i % len(label_pool)]
        conf = round(np.random.uniform(0.35, 0.92), 3)
        dets.append({
            "label": lbl, "conf": conf,
            "bbox":  [x1, y1, x2, y2],
            "cx":    (x1 + x2) // 2,
            "cy":    (y1 + y2) // 2,
        })

    if not dets:
        for i in range(6):
            bw = np.random.randint(int(w * 0.06), int(w * 0.20))
            bh = np.random.randint(int(h * 0.06), int(h * 0.18))
            x1 = np.random.randint(0, max(1, w - bw))
            y1 = np.random.randint(0, max(1, h - bh))
            dets.append({
                "label": label_pool[i % len(label_pool)],
                "conf":  round(np.random.uniform(0.35, 0.90), 3),
                "bbox":  [x1, y1, x1 + bw, y1 + bh],
                "cx":    x1 + bw // 2, "cy": y1 + bh // 2,
            })

    return dets


def extract_region(img_rgb, bbox):
    x1, y1, x2, y2 = bbox
    h, w = img_rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img_rgb[y1:y2, x1:x2]


def compute_texture_variance(region_gray):
    if region_gray is None or region_gray.size == 0:
        return 0.0
    lap = cv2.Laplacian(region_gray.astype(np.float32), cv2.CV_32F)
    return float(np.clip(lap.var() / 5000.0, 0.0, 1.0))


def compute_shape_regularity(bbox):
    x1, y1, x2, y2 = bbox
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    ratio = min(bw, bh) / max(bw, bh)
    return round(float(ratio), 3)


def compute_region_vari(region_rgb):
    if region_rgb is None or region_rgb.size == 0:
        return 0.0
    f = region_rgb.astype(np.float32)
    r, g, b = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    denom = np.where(np.abs(g + r - b) < 1e-6, 1e-6, g + r - b)
    vari = np.clip((g - r) / denom, -1, 1)
    return float(vari.mean())


def classify_mound(shape_reg, tex_var, vari_val, conf):
    if vari_val > 0.35:
        return "Natural", 0.3
    if shape_reg < 0.45:
        return "Natural", 0.35

    score = (
        0.35 * shape_reg +
        0.25 * (1.0 - tex_var) +
        0.25 * max(0.0, 1.0 - vari_val) +
        0.15 * conf
    )

    if score > 0.65:
        return "Man-made", round(score, 3)
    elif score < 0.45:
        return "Natural", round(score, 3)
    else:
        return "Uncertain", round(score, 3)


def run_mound_pipeline(img_rgb, model, conf_threshold=40, filter_high_conf=True):
    img_bgr    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    candidates = detect_mound_candidates(img_bgr, model, conf_threshold)

    results = []
    for det in candidates:
        region = extract_region(img_rgb, det["bbox"])
        if region is None:
            continue
        gray_region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY) if region.ndim == 3 else region
        tex_var   = compute_texture_variance(gray_region)
        shape_reg = compute_shape_regularity(det["bbox"])
        vari_val  = compute_region_vari(region)
        cls_label, cls_score = classify_mound(shape_reg, tex_var, vari_val, det["conf"])

        results.append({
            **det,
            "tex_var":   round(tex_var, 3),
            "shape_reg": round(shape_reg, 3),
            "vari_val":  round(vari_val, 3),
            "cls_label": cls_label,
            "cls_score": round(cls_score, 3),
        })

    for r in results:
        r["highlight"] = (r["cls_label"] == "Man-made" and
                          r["conf"] >= conf_threshold / 100.0) if filter_high_conf else True

    return results


# ====================== FIXED: draw_mound_overlay ======================
def draw_mound_overlay(img_rgb, results, filter_high_conf=True):
    """
    Draw all detected objects on the overlay image.

    Fix (v4.2): The previous alpha-blend approach for Natural/Uncertain boxes
    was broken — cv2.addWeighted on nearly-identical arrays produced invisible
    boxes. Now all boxes are drawn directly onto the overlay:
      - Natural / Uncertain: thin 1px border + short label (subtle but visible)
      - Man-made (highlighted): thick 3px border + filled label tag (prominent)
    Non-highlighted boxes are drawn first so Man-made always renders on top.
    """
    overlay = img_rgb.copy()
    color_map = {
        "Man-made":  (214, 60,  60),
        "Natural":   (80,  200, 120),
        "Uncertain": (212, 168, 60),
    }

    # ── Pass 1: draw Natural / Uncertain (thin, subtle) ──────────────────────
    for r in results:
        if r.get("highlight"):
            continue
        x1, y1, x2, y2 = r["bbox"]
        col = color_map.get(r["cls_label"], (150, 150, 150))

        # Thin 1-px rectangle — always visible, not overpowering
        cv2.rectangle(overlay, (x1, y1), (x2, y2), col, 1)

        # Small text label in top-left corner of the box (no filled background)
        # e.g. "N 72%" or "U 58%"
        short_tag = f"{r['cls_label'][0]} {r['conf']:.0%}"
        cv2.putText(
            overlay, short_tag,
            (x1 + 3, y1 + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40,
            col, 1, cv2.LINE_AA,
        )

    # ── Pass 2: draw Man-made / highlighted (thick, prominent) ───────────────
    for r in results:
        if not r.get("highlight"):
            continue
        x1, y1, x2, y2 = r["bbox"]
        col = color_map.get(r["cls_label"], (150, 150, 150))

        # Thick 3-px rectangle
        cv2.rectangle(overlay, (x1, y1), (x2, y2), col, 3)

        # Filled label tag above the box
        label_txt = f"{r['cls_label']} {r['conf']:.0%}"
        (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 6, y1), col, -1)
        cv2.putText(
            overlay, label_txt,
            (x1 + 3, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (20, 20, 20), 1, cv2.LINE_AA,
        )

    return overlay
# ====================== END FIX ======================


def compute_cost_savings(results, total_area_sqkm=50.0):
    total   = len(results)
    manmade = sum(1 for r in results if r["cls_label"] == "Man-made")
    natural = sum(1 for r in results if r["cls_label"] == "Natural")
    uncert  = total - manmade - natural

    pct_filtered  = round((natural / total) * 100, 1) if total else 0.0
    area_filtered = round(total_area_sqkm * pct_filtered / 100, 1)
    area_priority = round(total_area_sqkm - area_filtered, 1)

    days_trad  = round(total_area_sqkm * 3)
    days_ai    = round(area_priority * 3)
    days_saved = days_trad - days_ai
    staff, daily_cost = 5, 200
    cost_trad  = days_trad * staff * daily_cost
    cost_ai    = days_ai   * staff * daily_cost
    cost_saved = cost_trad - cost_ai

    return {
        "total":         total,
        "manmade":       manmade,
        "natural":       natural,
        "uncertain":     uncert,
        "pct_filtered":  pct_filtered,
        "area_filtered": area_filtered,
        "area_priority": area_priority,
        "days_saved":    days_saved,
        "cost_saved":    cost_saved,
        "cost_trad":     cost_trad,
        "cost_ai":       cost_ai,
    }


def build_detection_heatmap(img_rgb, results):
    h, w = img_rgb.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        sigma = max((x2 - x1), (y2 - y1)) // 2 or 20
        for dy in range(-sigma * 2, sigma * 2 + 1):
            for dx in range(-sigma * 2, sigma * 2 + 1):
                ry, rx = cy + dy, cx + dx
                if 0 <= ry < h and 0 <= rx < w:
                    heat[ry, rx] += float(r["conf"]) * np.exp(-(dx**2 + dy**2) / (2 * sigma**2))

    heat = cv2.GaussianBlur(heat, (0, 0), max(h, w) // 40 or 10)
    if heat.max() > 0:
        heat = heat / heat.max()
    heat_u8  = (heat * 255).astype(np.uint8)
    heat_col = cv2.applyColorMap(heat_u8, cv2.COLORMAP_INFERNO)
    heat_rgb = cv2.cvtColor(heat_col, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 0.55, heat_rgb, 0.45, 0)


def build_mound_report_widget(location_name, lat_val, lon_val, savings, results) -> str:
    loc = (location_name or "Unknown Site").replace('"', '\\"').replace("'", "\\'")
    mm  = savings["manmade"]
    nat = savings["natural"]
    unc = savings["uncertain"]
    tot = savings["total"]
    pct = savings["pct_filtered"]
    ds  = savings["days_saved"]
    cs  = savings["cost_saved"]
    ap  = savings["area_priority"]

    return f"""<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500&family=JetBrains+Mono:wght@300;400&family=Archivo+Narrow:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:transparent;font-family:'Archivo Narrow',sans-serif;color:#d4cfc8}}
.panel{{background:linear-gradient(160deg,#0e1118 0%,#111520 100%);border:1px solid #1e2535;border-top:2px solid #7a5f42;border-radius:4px;padding:16px 18px 14px;}}
.panel-label{{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:0.22em;text-transform:uppercase;color:#4a5268;margin-bottom:10px;display:flex;align-items:center;gap:10px;}}
.panel-label::after{{content:'';flex:1;height:1px;background:linear-gradient(90deg,#1e2535,transparent);}}
.data-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-bottom:10px;}}
.dc{{background:rgba(255,255,255,0.03);border:1px solid #1e2535;border-radius:3px;padding:7px 9px;}}
.dcl{{font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:0.13em;text-transform:uppercase;color:#4a5268;margin-bottom:3px;}}
.dcv{{font-family:'Cormorant Garamond',serif;font-size:15px;color:#d4cfc8;}}
.dcv.red{{color:#d46b6b;}} .dcv.green{{color:#7ec899;}} .dcv.gold{{color:#d4a84b;}}
.trigger{{display:flex;align-items:center;justify-content:space-between;cursor:pointer;padding:9px 11px;background:rgba(255,255,255,0.02);border:1px solid #1e2535;border-radius:3px;transition:all 0.2s;user-select:none;}}
.trigger:hover{{border-color:#7a5f42;background:rgba(184,150,110,0.04);}}
.trigger-text{{font-family:'Archivo Narrow',sans-serif;font-size:11px;letter-spacing:0.1em;text-transform:uppercase;color:#7a8399;}}
.trigger-arrow{{font-family:'JetBrains Mono',monospace;font-size:11px;color:#7a5f42;transition:transform 0.2s;}}
#output{{margin-top:10px;display:none}} #output.show{{display:block}}
.loading-row{{display:flex;align-items:center;gap:10px;font-family:'JetBrains Mono',monospace;font-size:10px;color:#4a5268;letter-spacing:0.08em;padding:6px 0;}}
.bar-loader{{flex:1;height:2px;background:#1e2535;border-radius:1px;overflow:hidden;position:relative;}}
.bar-loader::after{{content:'';position:absolute;top:0;left:-40%;width:40%;height:100%;background:linear-gradient(90deg,transparent,#b8966e,transparent);animation:scan 1.4s linear infinite;}}
@keyframes scan{{to{{left:140%}}}}
#report{{background:rgba(255,255,255,0.025);border:1px solid #1e2535;border-left:3px solid #7a5f42;border-radius:0 3px 3px 0;padding:12px 14px;font-family:'JetBrains Mono',monospace;font-size:10.5px;line-height:1.95;color:#a0b0a0;white-space:pre-wrap;display:none;}}
.cursor{{display:inline-block;width:6px;height:11px;background:#b8966e;margin-left:2px;vertical-align:middle;border-radius:1px;animation:blink .5s infinite;}}
@keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:0}}}}
#errmsg{{font-family:'JetBrains Mono',monospace;font-size:10px;color:#d46b6b;padding:7px 10px;background:rgba(124,58,58,0.08);border:1px solid rgba(124,58,58,0.2);border-radius:3px;display:none;}}
</style>
</head>
<body>
<div class="panel">
  <div class="panel-label">AI Survey Report — Mound Analysis</div>
  <div class="data-grid">
    <div class="dc"><div class="dcl">Total Detected</div><div class="dcv">{tot}</div></div>
    <div class="dc"><div class="dcl">Man-made</div><div class="dcv red">{mm}</div></div>
    <div class="dc"><div class="dcl">Natural</div><div class="dcv green">{nat}</div></div>
    <div class="dc"><div class="dcl">Uncertain</div><div class="dcv gold">{unc}</div></div>
    <div class="dc"><div class="dcl">Area Filtered</div><div class="dcv">{pct:.0f}%</div></div>
    <div class="dc"><div class="dcl">Days Saved</div><div class="dcv green">{ds}</div></div>
    <div class="dc" style="grid-column:span 3;background:rgba(122,95,66,0.08);border-color:rgba(122,95,66,0.3);">
      <div class="dcl">Priority Survey Area</div>
      <div class="dcv" style="font-size:13px;">{ap} sq.km — focusing on {mm} man-made candidates</div>
    </div>
  </div>
  <div class="trigger" id="triggerBtn" onclick="runReport()">
    <span class="trigger-text" id="hint">Generate AI mound survey report</span>
    <span class="trigger-arrow" id="arr">&#x25BA;</span>
  </div>
  <div id="output">
    <div class="loading-row" id="loading"><div class="bar-loader"></div><span>Generating field survey report…</span></div>
    <div id="report"></div>
    <div id="errmsg"></div>
  </div>
</div>
<script>
var busy=false;
var GROQ_KEY="{GROQ_API_KEY}";
var LOC="{loc}",LAT={lat_val:.4f},LON={lon_val:.4f};
var TOT={tot},MM={mm},NAT={nat},UNC={unc},PCT={pct:.1f},DS={ds},CS={cs},AP={ap:.1f};

function highlight(txt){{
  return txt.replace(/\\n/g,'<br>')
    .replace(/(SUMMARY|KEY FINDINGS|SITE POTENTIAL|RECOMMENDATION):/g,
      '<span style="font-size:8.5px;letter-spacing:0.2em;color:#b8966e;font-weight:600;">$1:</span>');
}}
function typewriter(el,text,speed,done){{
  el.innerHTML='<span class="cursor"></span>';
  var i=0,buf='';
  (function tick(){{
    if(i<text.length){{buf+=text[i++];el.innerHTML=highlight(buf)+'<span class="cursor"></span>';
    setTimeout(tick,(buf[buf.length-1]==='.'||buf[buf.length-1]==='?')?speed*5:speed);}}
    else{{el.innerHTML=highlight(buf);if(done)done();}}
  }})();
}}
async function runReport(){{
  if(busy)return;busy=true;
  var out=document.getElementById('output'),loading=document.getElementById('loading'),
      report=document.getElementById('report'),errmsg=document.getElementById('errmsg'),
      hint=document.getElementById('hint'),arr=document.getElementById('arr');
  report.style.display='none';errmsg.style.display='none';
  loading.style.display='flex';out.classList.add('show');
  hint.textContent='Generating report…';arr.style.transform='rotate(90deg)';

  var prompt=
"You are a senior field archaeologist writing a structured mound survey assessment.\\n\\n"+
"SURVEY DATA:\\n"+
"  Site: "+LOC+" ("+LAT.toFixed(4)+"N, "+LON.toFixed(4)+"E)\\n"+
"  Total objects detected: "+TOT+"\\n"+
"  Man-made (potential archaeological): "+MM+"\\n"+
"  Natural: "+NAT+"\\n"+
"  Uncertain: "+UNC+"\\n"+
"  Area filtered out (natural eliminated): "+PCT+"%\\n"+
"  Priority survey area: "+AP+" sq.km\\n"+
"  Estimated field days saved: "+DS+"\\n"+
"  Estimated cost saved (USD): $"+CS+"\\n\\n"+
"Write EXACTLY 4 labelled lines. No preamble.\\n\\n"+
"SUMMARY: Overall picture — "+TOT+" objects detected, "+MM+" man-made candidates, "+PCT+"% filtered.\\n"+
"KEY FINDINGS: Ratio of man-made to natural ("+MM+":"+NAT+"), filtering efficiency and implications.\\n"+
"SITE POTENTIAL: Archaeological significance of "+MM+" man-made candidates at "+LOC+".\\n"+
"RECOMMENDATION: One specific actionable next step for the "+MM+" high-priority zones.\\n\\n"+
"Rules: Each line MUST start with its label in uppercase + colon. Plain English. No markdown. Max 45 words per line.";

  try{{
    var resp=await fetch('https://api.groq.com/openai/v1/chat/completions',{{
      method:'POST',
      headers:{{'Content-Type':'application/json','Authorization':'Bearer '+GROQ_KEY}},
      body:JSON.stringify({{model:'llama-3.3-70b-versatile',messages:[{{role:'user',content:prompt}}],temperature:0.4,max_tokens:500,stream:false}})
    }});
    if(!resp.ok)throw new Error('API error '+resp.status);
    var data=await resp.json();
    var full=(data.choices&&data.choices[0]&&data.choices[0].message&&data.choices[0].message.content)||'';
    loading.style.display='none';report.style.display='block';
    typewriter(report,full.trim(),13,function(){{hint.textContent='Report complete — click to regenerate';arr.style.transform='rotate(0deg)';busy=false;}});
  }}catch(err){{
    loading.style.display='none';errmsg.style.display='block';
    errmsg.textContent='Error: '+err.message;hint.textContent='Click to retry';arr.style.transform='rotate(0deg)';busy=false;
  }}
}}
</script>
</body>
</html>"""


# ====================== KML / KMZ ======================
def build_kml(lat, lon, detections, risk):
    marks = "".join(
        f"\n  <Placemark><n>{d['label']} #{i + 1}</n>"
        f"<description>Confidence: {d['conf']:.2%}</description>"
        f"<Point><coordinates>{lon + i * 0.0001},{lat + i * 0.0001},0</coordinates></Point></Placemark>"
        for i, d in enumerate(detections)
    )
    lbl = "Low" if risk < 0.33 else ("Moderate" if risk < 0.66 else "High")
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<kml xmlns="http://www.opengis.net/kml/2.2"><Document>\n'
        f'  <n>Archaeological Site</n>\n'
        f'  <Placemark><n>Site Origin</n>\n'
        f'    <description>Erosion Risk: {risk:.2%} ({lbl})</description>\n'
        f'    <Point><coordinates>{lon},{lat},0</coordinates></Point>\n'
        f'  </Placemark>{marks}\n</Document></kml>'
    )

def build_kmz(kml):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", kml)
    return buf.getvalue()

# ====================== AI REPORT WIDGET (EROSION) ======================
def build_ai_report_widget(location_name, lat_val, lon_val, slope_val, elev_val,
                            risk_score, risk_level_str, vari_mean, coverage,
                            detect_count, auto_conf) -> str:
    loc = (location_name or "Unknown Site").replace('"', '\\"').replace("'", "\\'")
    cov_str = " | ".join(f"{l}: {p}%" for l, p in coverage.items()) if coverage else "No vegetation data"

    if risk_score < 0.33:
        rc, rb, rl = "#7ec899", "rgba(74,124,89,0.12)", "rgba(74,124,89,0.35)"
    elif risk_score < 0.66:
        rc, rb, rl = "#d4a84b", "rgba(138,110,47,0.12)", "rgba(138,110,47,0.35)"
    else:
        rc, rb, rl = "#d46b6b", "rgba(124,58,58,0.12)", "rgba(124,58,58,0.35)"

    risk_pct  = min(risk_score * 100, 100)
    why_label = "WHY SAFE" if risk_score < 0.33 else ("WHY MODERATE" if risk_score < 0.66 else "WHY HARMFUL")

    return f"""<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500&family=JetBrains+Mono:wght@300;400&family=Archivo+Narrow:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:transparent;font-family:'Archivo Narrow',sans-serif;color:#d4cfc8}}
.panel{{background:linear-gradient(160deg,#0e1118 0%,#111520 100%);border:1px solid #1e2535;border-top:2px solid #7a5f42;border-radius:4px;padding:16px 18px 14px;}}
.panel-label{{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:0.22em;text-transform:uppercase;color:#4a5268;margin-bottom:10px;display:flex;align-items:center;gap:10px;}}
.panel-label::after{{content:'';flex:1;height:1px;background:linear-gradient(90deg,#1e2535,transparent);}}
.data-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-bottom:10px;}}
.dc{{background:rgba(255,255,255,0.03);border:1px solid #1e2535;border-radius:3px;padding:7px 9px;}}
.dcl{{font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:0.13em;text-transform:uppercase;color:#4a5268;margin-bottom:3px;}}
.dcv{{font-family:'Cormorant Garamond',serif;font-size:15px;color:#d4cfc8;}}
.risk-cell{{background:{rb};border-color:{rl};grid-column:span 3;display:flex;align-items:center;justify-content:space-between;padding:9px 12px;}}
.risk-text{{font-family:'Cormorant Garamond',serif;font-size:18px;color:{rc};}}
.risk-bar-wrap{{flex:1;margin:0 12px;height:3px;background:rgba(255,255,255,0.06);border-radius:2px;overflow:hidden;}}
.risk-bar-fill{{height:100%;width:{risk_pct:.1f}%;background:{rc};border-radius:2px;}}
.trigger{{display:flex;align-items:center;justify-content:space-between;cursor:pointer;padding:9px 11px;background:rgba(255,255,255,0.02);border:1px solid #1e2535;border-radius:3px;transition:all 0.2s;user-select:none;}}
.trigger:hover{{border-color:#7a5f42;background:rgba(184,150,110,0.04);}}
.trigger-text{{font-family:'Archivo Narrow',sans-serif;font-size:11px;letter-spacing:0.1em;text-transform:uppercase;color:#7a8399;}}
.trigger-arrow{{font-family:'JetBrains Mono',monospace;font-size:11px;color:#7a5f42;transition:transform 0.2s;}}
.trigger:hover .trigger-arrow{{color:#b8966e;}}
#output{{margin-top:10px;display:none}} #output.show{{display:block}}
.loading-row{{display:flex;align-items:center;gap:10px;font-family:'JetBrains Mono',monospace;font-size:10px;color:#4a5268;letter-spacing:0.08em;padding:6px 0;}}
.bar-loader{{flex:1;height:2px;background:#1e2535;border-radius:1px;overflow:hidden;position:relative;}}
.bar-loader::after{{content:'';position:absolute;top:0;left:-40%;width:40%;height:100%;background:linear-gradient(90deg,transparent,#b8966e,transparent);animation:scan 1.4s linear infinite;}}
@keyframes scan{{to{{left:140%}}}}
#report{{background:rgba(255,255,255,0.025);border:1px solid #1e2535;border-left:3px solid #7a5f42;border-radius:0 3px 3px 0;padding:12px 14px;font-family:'JetBrains Mono',monospace;font-size:10.5px;line-height:1.95;color:#a0b0a0;white-space:pre-wrap;display:none;}}
.cursor{{display:inline-block;width:6px;height:11px;background:#b8966e;margin-left:2px;vertical-align:middle;border-radius:1px;animation:blink .5s infinite;}}
@keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:0}}}}
#errmsg{{font-family:'JetBrains Mono',monospace;font-size:10px;color:#d46b6b;padding:7px 10px;background:rgba(124,58,58,0.08);border:1px solid rgba(124,58,58,0.2);border-radius:3px;display:none;}}
</style>
</head>
<body>
<div class="panel">
  <div class="panel-label">AI Field Report</div>
  <div class="data-grid">
    <div class="dc"><div class="dcl">Slope</div><div class="dcv">{slope_val:.1f}&deg;</div></div>
    <div class="dc"><div class="dcl">Elevation</div><div class="dcv">{elev_val:.0f} m</div></div>
    <div class="dc"><div class="dcl">VARI Index</div><div class="dcv">{vari_mean:.3f}</div></div>
    <div class="dc"><div class="dcl">Artifacts</div><div class="dcv">{detect_count}</div></div>
    <div class="dc"><div class="dcl">Confidence</div><div class="dcv">{auto_conf:.0%}</div></div>
    <div class="dc"><div class="dcl">Site</div><div class="dcv" style="font-size:11px;font-family:'Archivo Narrow',sans-serif;">{loc[:16]}</div></div>
    <div class="dc risk-cell">
      <div><div class="dcl">Erosion Risk</div><div class="risk-text">{risk_level_str} &mdash; {risk_score:.1%}</div></div>
      <div class="risk-bar-wrap"><div class="risk-bar-fill"></div></div>
    </div>
  </div>
  <div class="trigger" id="triggerBtn" onclick="runReport()">
    <span class="trigger-text" id="hint">Generate AI field analysis</span>
    <span class="trigger-arrow" id="arr">&#x25BA;</span>
  </div>
  <div id="output">
    <div class="loading-row" id="loading"><div class="bar-loader"></div><span>Analysing site data…</span></div>
    <div id="report"></div>
    <div id="errmsg"></div>
  </div>
</div>
<script>
var busy=false;
var GROQ_KEY="{GROQ_API_KEY}";
var LOCATION="{loc}",LAT={lat_val:.4f},LON={lon_val:.4f};
var SLOPE={slope_val:.1f},ELEVATION={elev_val:.0f};
var RISK_SCORE={risk_score:.4f},RISK_LEVEL="{risk_level_str}";
var VARI_MEAN={vari_mean:.4f},COVERAGE="{cov_str}";
var ARTIFACTS={detect_count},DET_CONF={auto_conf:.2f};
var WHY_LABEL="{why_label}";

function highlight(txt){{
  return txt.replace(/\\n/g,'<br>')
    .replace(/(EROSION RISK|{why_label}|TERRAIN|VEGETATION|RECOMMENDATION):/g,
      '<span style="font-size:8.5px;letter-spacing:0.2em;color:#b8966e;font-weight:600;">$1:</span>');
}}
function typewriter(el,text,speed,done){{
  el.innerHTML='<span class="cursor"></span>';var i=0,buf='';
  (function tick(){{
    if(i<text.length){{buf+=text[i++];el.innerHTML=highlight(buf)+'<span class="cursor"></span>';
    setTimeout(tick,(buf[buf.length-1]==='.'||buf[buf.length-1]==='?')?speed*5:speed);}}
    else{{el.innerHTML=highlight(buf);if(done)done();}}
  }})();
}}
async function runReport(){{
  if(busy)return;busy=true;
  var out=document.getElementById('output'),loading=document.getElementById('loading'),
      report=document.getElementById('report'),errmsg=document.getElementById('errmsg'),
      hint=document.getElementById('hint'),arr=document.getElementById('arr');
  report.style.display='none';errmsg.style.display='none';
  loading.style.display='flex';out.classList.add('show');
  hint.textContent='Generating analysis…';arr.style.transform='rotate(90deg)';
  var vaText=VARI_MEAN>=0.35?"moderate to dense":VARI_MEAN>=0.2?"sparse":"very sparse / bare soil";
  var riskPct=(RISK_SCORE*100).toFixed(1);
  var prompt=
"You are a senior field archaeologist writing a structured site assessment.\\n\\n"+
"SITE DATA:\\n  Location: "+LOCATION+" ("+LAT.toFixed(4)+"N, "+LON.toFixed(4)+"E)\\n"+
"  Slope: "+SLOPE+"deg | Elevation: "+ELEVATION+"m | Detection confidence: "+(DET_CONF*100).toFixed(0)+"%\\n"+
"  VARI: "+VARI_MEAN.toFixed(3)+" ("+vaText+")\\n  Vegetation: "+COVERAGE+"\\n"+
"  Erosion Risk: "+riskPct+"% — "+RISK_LEVEL+"\\n  Artifacts: "+ARTIFACTS+"\\n\\n"+
"Write EXACTLY 5 labelled lines. No preamble.\\n"+
"EROSION RISK: "+RISK_LEVEL+" — cite slope ("+SLOPE+"deg), elevation ("+ELEVATION+"m), score ("+riskPct+"%).\\n"+
WHY_LABEL+": Explain WHY slope+elevation+VARI drives this risk and its archaeological implications.\\n"+
"TERRAIN: Describe terrain character. What site types does this terrain preserve or destroy?\\n"+
"VEGETATION: Interpret VARI ("+vaText+") and breakdown. How does it affect preservation?\\n"+
"RECOMMENDATION: One specific actionable conservation or investigation step.\\n\\n"+
"Rules: label:colon required. Plain English. No markdown. Max 40 words per line.";
  try{{
    var resp=await fetch('https://api.groq.com/openai/v1/chat/completions',{{
      method:'POST',
      headers:{{'Content-Type':'application/json','Authorization':'Bearer '+GROQ_KEY}},
      body:JSON.stringify({{model:'llama-3.3-70b-versatile',messages:[{{role:'user',content:prompt}}],temperature:0.45,max_tokens:500,stream:false}})
    }});
    if(!resp.ok)throw new Error('API error '+resp.status);
    var data=await resp.json();
    var full=(data.choices&&data.choices[0]&&data.choices[0].message&&data.choices[0].message.content)||'';
    loading.style.display='none';report.style.display='block';
    typewriter(report,full.trim(),13,function(){{hint.textContent='Analysis complete — click to regenerate';arr.style.transform='rotate(0deg)';busy=false;}});
  }}catch(err){{
    loading.style.display='none';errmsg.style.display='block';
    errmsg.textContent='Error: '+err.message;hint.textContent='Click to retry';arr.style.transform='rotate(0deg)';busy=false;
  }}
}}
</script>
</body>
</html>"""

# ====================== SIDEBAR ======================
def render_sidebar() -> Dict:
    with st.sidebar:
        st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
            transform: translateX(0) !important;
            min-width: 21rem !important;
        }
        [data-testid="stSidebar"][aria-expanded="false"] {
            margin-left: 0 !important;
            transform: translateX(0) !important;
            width: 21rem !important;
        }
        [data-testid="collapsedControl"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-brand">ArchAI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-sub">Archaeological Intelligence Platform</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">Theme</div>', unsafe_allow_html=True)
        selected_theme = st.radio(
            "Theme", ["Dark", "Light"],
            index=0 if st.session_state["theme"] == "Dark" else 1,
            horizontal=True, label_visibility="collapsed",
        )
        if selected_theme != st.session_state["theme"]:
            st.session_state["theme"] = selected_theme
            st.rerun()

        st.markdown('<div class="sidebar-section">Detection Model</div>', unsafe_allow_html=True)
        local_path = st.text_input("Weights path", value="model/best.pt", label_visibility="collapsed")
        st.caption("Model found" if Path(local_path.strip()).exists() else "Demo mode — model not found")
        conf = st.slider("Confidence threshold", 10, 90, 40, 5)

        st.markdown('<div class="sidebar-section">Erosion Model</div>', unsafe_allow_html=True)
        erosion_path = st.text_input("Erosion model (.pkl)", value="erosion_model.pkl", label_visibility="collapsed")
        st.caption("Erosion model found" if erosion_path and Path(erosion_path.strip()).exists() else "Formula fallback active")

        st.markdown('<div class="sidebar-section">Location Search</div>', unsafe_allow_html=True)
        location_input = st.text_input(
            "Location name",
            value=st.session_state.get("location_name", ""),
            placeholder="e.g. Hampi, Karnataka",
            label_visibility="collapsed",
        )
        st.caption("Add district for small villages")

        if st.button("Locate Coordinates", use_container_width=True):
            if not location_input.strip():
                st.warning("Enter a location name.")
            else:
                with st.spinner("Searching…"):
                    lat, lon, msg = geocode_location(location_input)
                if lat is not None:
                    st.session_state.update({"lat": lat, "lon": lon,
                                             "location_name": location_input, "geo_msg": msg})
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

        st.markdown('<div class="sidebar-section">Coordinates</div>', unsafe_allow_html=True)
        lat = st.number_input("Latitude",  value=float(st.session_state.get("lat", 20.5937)), format="%.6f")
        lon = st.number_input("Longitude", value=float(st.session_state.get("lon", 78.9629)), format="%.6f")
        if lat != st.session_state.get("lat"): st.session_state["lat"] = lat
        if lon != st.session_state.get("lon"): st.session_state["lon"] = lon

        st.markdown('<div class="sidebar-section">Terrain Parameters</div>', unsafe_allow_html=True)
        auto      = st.session_state.get("auto_terrain", {})
        slope     = auto.get("slope", 0.0)
        elev      = auto.get("elevation", 0.0)
        auto_conf = auto.get("confidence", 0.0)
        if auto and (slope > 0 or elev > 0):
            c1, c2, c3 = st.columns(3)
            c1.metric("Slope", f"{slope:.1f}°")
            c2.metric("Elev",  f"{elev:.0f}m")
            c3.metric("Conf",  f"{auto_conf:.0%}")
        else:
            st.caption("Upload an image in Analysis to auto-detect terrain.")

        st.markdown('<div class="sidebar-section">AI Analysis</div>', unsafe_allow_html=True)
        risk      = st.session_state.get("risk", 0.0)
        vari_mean = st.session_state.get("vari_mean", 0.0)
        coverage  = st.session_state.get("coverage", {})
        dets      = st.session_state.get("dets", [])
        loc_name  = st.session_state.get("location_name", "Unknown Site")
        risk_lbl  = "LOW" if risk < 0.33 else ("MODERATE" if risk < 0.66 else "HIGH")

        widget_html = build_ai_report_widget(
            loc_name, lat, lon, slope, elev, risk, risk_lbl,
            vari_mean, coverage, len(dets), auto_conf,
        )
        st.components.v1.html(widget_html, height=510, scrolling=False)

    return dict(local_path=local_path, conf=conf, erosion_path=erosion_path,
                lat=lat, lon=lon, slope=slope, elev=elev)

# ====================== TAB: ANALYSIS ======================
def tab_analysis(params, det_model, det_mode, er_model, feat_names):
    st.markdown("## Image Analysis")

    if det_model is None:
        st.info("Demo mode active — YOLO model not loaded. VARI and erosion analysis are fully operational.")
    else:
        st.success(f"Detection model loaded — {params['local_path']}")

    uploaded = st.file_uploader("Upload satellite or drone image", type=["jpg", "jpeg", "png", "tif", "tiff"])
    if not uploaded:
        st.caption("Accepted formats: JPG, PNG, TIF. Maximum recommended resolution: 4000 px.")
        return

    raw = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Cannot decode image.")
        return

    h, w = img.shape[:2]
    if w > 1280:
        img = cv2.resize(img, (1280, int(h * 1280 / w)))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with st.spinner("Detecting terrain parameters…"):
        terrain = auto_detect_terrain(rgb)
    st.session_state["auto_terrain"] = terrain

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Original Image")
        st.image(rgb, use_container_width=True)
    with c2:
        st.markdown("#### Artifact Detection")
        with st.spinner("Running inference…"):
            det_img, dets = run_detection(img, det_model, det_mode or "local", params["conf"])
        st.image(det_img, use_container_width=True)
        st.metric("Artifacts detected", len(dets))

    st.session_state["dets"] = dets

    if dets:
        st.dataframe(
            pd.DataFrame([{"Label": d["label"], "Confidence": f"{d['conf']:.2%}",
                           "Centre": f"({d['cx']}, {d['cy']})"} for d in dets]),
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")

    vari = compute_vari(rgb)
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### VARI Vegetation Index")
        st.image(colorise_vari(vari), use_container_width=True)
        vari_mean = float(vari.mean())
        st.metric("Mean VARI", f"{vari_mean:.3f}")
        st.caption("Very low vegetation." if vari_mean < -0.05
                   else ("Sparse vegetation." if vari_mean < 0.2 else "Moderate to dense vegetation."))

    seg, cov = segment_vegetation(vari)
    with c4:
        st.markdown("#### Vegetation Segmentation")
        st.image(seg, use_container_width=True)
        for lbl, pct in cov.items():
            st.progress(int(pct), text=f"{lbl}  —  {pct}%")

    st.session_state["vari_mean"] = float(vari.mean())
    st.session_state["coverage"]  = cov

    st.markdown("---")
    st.markdown("## Erosion Risk Assessment")

    auto_slope = float(terrain["slope"])
    auto_elev  = float(terrain["elevation"])
    auto_conf  = float(terrain["confidence"])
    ndvi_val   = float(vari.mean())

    risk = predict_erosion_score(er_model, feat_names, auto_slope, auto_elev, ndvi_val)
    st.session_state["risk"] = risk

    lbl = "LOW" if risk < 0.33 else ("MODERATE" if risk < 0.66 else "HIGH")
    fg  = "#7ec899" if risk < 0.33 else ("#d4a84b" if risk < 0.66 else "#d46b6b")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Risk Score", f"{risk:.3f}")
    m2.metric("Risk Level", lbl)
    m3.metric("Slope", f"{auto_slope:.1f}°")
    m4.metric("Elevation", f"{auto_elev:.0f} m")
    m5.metric("Detection Confidence", f"{auto_conf:.0%}")

    st.markdown(f"""
    <div style="margin:12px 0 4px;height:6px;background:var(--border);border-radius:3px;overflow:hidden;">
      <div style="height:100%;width:{min(risk*100,100):.1f}%;background:{fg};border-radius:3px;"></div>
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--text-muted);letter-spacing:0.12em;text-transform:uppercase;margin-bottom:12px;">
      Risk score: {risk:.3f} / 1.000 &nbsp;|&nbsp; Classification: {lbl}
    </div>""", unsafe_allow_html=True)

    with st.expander("Terrain Detection Methodology"):
        st.markdown(
            f"| Parameter | Value | Detection Method |\n"
            f"|-----------|-------|------------------|\n"
            f"| Slope | {terrain['slope']}° | Sobel gradient magnitude |\n"
            f"| Elevation | {terrain['elevation']} m | Brightness + texture + shadow composite |\n"
            f"| VARI (NDVI proxy) | {ndvi_val:.3f} | R/G/B visible vegetation index |\n"
            f"| Detection confidence | {terrain['confidence']:.0%} | Image contrast and dynamic range |"
        )


# ====================== TAB: MOUND DETECTION ======================
def tab_mound_detection(params, det_model, det_mode):
    st.markdown("## AI-Assisted Survey Optimization & Object Detection")
    st.caption(
        "Upload satellite or drone imagery — the AI detects ALL visible objects, classifies them "
        "as Man-made or Natural, filters out irrelevant zones, and estimates survey time/cost saved."
    )

    if det_model is None:
        st.info(
            "Demo mode — YOLO model not loaded. Image-driven synthetic candidates will be generated "
            "based on texture analysis to demonstrate the classification pipeline."
        )

    col_up, col_ctrl = st.columns([3, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Upload satellite / drone image",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            key="mound_upload",
        )
    with col_ctrl:
        survey_area = st.number_input(
            "Survey Area (sq.km)", min_value=1.0, max_value=500.0, value=50.0, step=5.0)
        filter_mode = st.checkbox("Show only Man-made candidates", value=True)
        run_btn     = st.button("Run Detection", use_container_width=True)

    if not uploaded:
        st.caption("Accepted formats: JPG, PNG, TIF.")
        st.markdown("""
        <div style="display:flex;gap:16px;margin-top:8px;font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:0.08em;">
          <span style="color:#d46b6b;">■ Man-made</span>
          <span style="color:#7ec899;">■ Natural</span>
          <span style="color:#d4a84b;">■ Uncertain</span>
        </div>""", unsafe_allow_html=True)
        return

    raw = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Cannot decode image.")
        return

    h, w = img.shape[:2]
    if w > 1280:
        img = cv2.resize(img, (1280, int(h * 1280 / w)))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if run_btn or st.session_state.get("mound_results"):
        if run_btn:
            with st.spinner("Running detection and classification…"):
                results = run_mound_pipeline(rgb, det_model, params["conf"], filter_high_conf=filter_mode)
            st.session_state["mound_results"] = results
        else:
            results = st.session_state.get("mound_results", [])

        if not results:
            st.warning("No candidates detected. Try lowering the confidence threshold.")
            return

        savings = compute_cost_savings(results, total_area_sqkm=survey_area)

        st.markdown("### Survey Cost & Time Savings")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total Detected",     savings["total"])
        m2.metric("Man-made",           savings["manmade"],  delta="Priority")
        m3.metric("Natural (filtered)", savings["natural"],  delta=f"-{savings['pct_filtered']:.0f}% area")
        m4.metric("Uncertain",          savings["uncertain"])
        m5.metric("Days Saved",         savings["days_saved"], delta="vs traditional")
        m6.metric("Cost Saved (USD)",   f"${savings['cost_saved']:,}")

        st.markdown(f"""
        <div style="margin:8px 0 16px;padding:10px 14px;background:var(--bg-elevated);
                    border:1px solid var(--border-light);border-radius:4px;
                    font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--text-secondary);
                    letter-spacing:0.07em;">
          Traditional survey: <strong style="color:var(--accent);">${savings['cost_trad']:,}</strong> &nbsp;|&nbsp;
          AI-optimised: <strong style="color:var(--risk-low-fg);">${savings['cost_ai']:,}</strong> &nbsp;|&nbsp;
          Priority area: <strong style="color:var(--accent);">{savings['area_priority']} sq.km</strong>
          of {survey_area:.0f} sq.km &nbsp;|&nbsp;
          {savings['pct_filtered']:.1f}% filtered as natural
        </div>""", unsafe_allow_html=True)

        st.markdown("### Detection Overlay")
        ov_img   = draw_mound_overlay(rgb, results, filter_high_conf=filter_mode)
        heat_img = build_detection_heatmap(rgb, results)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### Original Image")
            st.image(rgb, use_container_width=True)
        with c2:
            st.markdown("#### Classification Overlay")
            st.image(ov_img, use_container_width=True)
            st.markdown("""<div style="display:flex;gap:12px;font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:0.07em;margin-top:4px;">
              <span style="color:#d46b6b;">■ Man-made (thick border + label)</span>
              <span style="color:#7ec899;">■ Natural (thin border)</span>
              <span style="color:#d4a84b;">■ Uncertain (thin border)</span>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown("#### Detection Density Heatmap")
            st.image(heat_img, use_container_width=True)
            st.caption("Brighter = higher detection density / confidence")

        st.markdown("### Classification Results")
        df_rows = []
        for i, r in enumerate(results):
            df_rows.append({
                "#":         i + 1,
                "Label":     r["label"],
                "Class":     r["cls_label"],
                "Conf":      f"{r['conf']:.2%}",
                "Cls Score": f"{r['cls_score']:.3f}",
                "Shape Reg": f"{r['shape_reg']:.3f}",
                "Tex Var":   f"{r['tex_var']:.3f}",
                "VARI":      f"{r['vari_val']:.3f}",
                "Priority":  "✓" if r.get("highlight") else "—",
            })
        df = pd.DataFrame(df_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("---")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            report_lines = [
                "SURVEY OPTIMIZATION REPORT", "=" * 44,
                f"Site         : {st.session_state.get('location_name', '—')}",
                f"Coordinates  : {params['lat']:.6f} N, {params['lon']:.6f} E",
                f"Survey Area  : {survey_area:.0f} sq.km", "",
                "DETECTION SUMMARY", "-" * 30,
                f"  Total Detected       : {savings['total']}",
                f"  Man-made Candidates  : {savings['manmade']}",
                f"  Natural              : {savings['natural']}",
                f"  Uncertain            : {savings['uncertain']}",
                f"  Area Filtered Out    : {savings['pct_filtered']}%",
                f"  Priority Area        : {savings['area_priority']} sq.km", "",
                "COST & TIME SAVINGS", "-" * 30,
                f"  Traditional (USD)    : ${savings['cost_trad']:,}",
                f"  AI-Optimised (USD)   : ${savings['cost_ai']:,}",
                f"  Estimated Saving     : ${savings['cost_saved']:,}",
                f"  Field Days Saved     : {savings['days_saved']}",
            ]
            report_txt = "\n".join(report_lines)
            st.download_button("Download Survey Report (.txt)", data=report_txt,
                               file_name="survey_report.txt", mime="text/plain")
        with col_dl2:
            st.download_button("Download Detection Data (.csv)", data=df.to_csv(index=False),
                               file_name="detections.csv", mime="text/csv")

        st.markdown("---")
        st.markdown("### AI Field Survey Report")
        st.components.v1.html(
            build_mound_report_widget(st.session_state.get("location_name", ""),
                                      params["lat"], params["lon"], savings, results),
            height=440, scrolling=False,
        )

        with st.expander("Classification Methodology"):
            st.markdown("""
| Feature | Weight | Description |
|---------|--------|-------------|
| Shape Regularity | 35% | Aspect ratio — man-made structures tend toward regular shapes |
| Texture Variance | 25% | Laplacian variance — natural objects have high texture |
| VARI Index | 25% | Vegetation — man-made surfaces typically lower vegetation |
| Detection Confidence | 15% | Model certainty score |

Score ≥ 0.65 → **Man-made** · Score ≤ 0.45 → **Natural** · Else → **Uncertain**
            """)
    else:
        st.caption("Upload an image and click **Run Detection** to begin.")


# ====================== DEFORESTATION AI MODULE ======================
def generate_vegetation_mask(img_rgb: np.ndarray, vari_threshold: float = 0.18):
    f     = img_rgb.astype(np.float32)
    r, g, b = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    denom = np.where(np.abs(g + r - b) < 1e-6, 1e-6, g + r - b)
    vari  = np.clip((g - r) / denom, -1.0, 1.0)
    mask  = vari > vari_threshold
    return mask, vari


def remove_vegetation(img_rgb: np.ndarray, mask: np.ndarray, intensity: float = 0.75) -> np.ndarray:
    out   = img_rgb.astype(np.float32).copy()
    earth = np.array([160.0, 130.0, 100.0], dtype=np.float32)

    for c in range(3):
        channel = out[:, :, c]
        channel[mask] = channel[mask] * (1.0 - intensity) + earth[c] * intensity
        out[:, :, c]  = channel

    green_excess = np.maximum(0.0, out[:, :, 1] - 0.5 * (out[:, :, 0] + out[:, :, 2]))
    suppress     = mask.astype(np.float32) * intensity
    out[:, :, 1] = out[:, :, 1] - green_excess * suppress
    return np.clip(out, 0, 255).astype(np.uint8)


def enhance_ground_features(img_rgb: np.ndarray, mask: np.ndarray,
                              clahe_clip: float = 3.0,
                              edge_strength: float = 0.6) -> np.ndarray:
    lab     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    L_clahe = clahe.apply(L)
    blur    = cv2.GaussianBlur(L_clahe, (0, 0), 3)
    L_sharp = cv2.addWeighted(L_clahe, 1.6, blur, -0.6, 0)
    enhanced_rgb = cv2.cvtColor(cv2.merge([L_sharp, A, B]), cv2.COLOR_LAB2RGB)

    gray_e  = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY)
    edges   = cv2.Canny(gray_e, 40, 120)
    edge_col = np.array([255, 210, 80], dtype=np.float32)
    edge_mask = (edges > 0) & (~mask)
    ground_float = enhanced_rgb.astype(np.float32)
    for c in range(3):
        ground_float[:, :, c] = np.where(
            edge_mask,
            ground_float[:, :, c] * (1 - edge_strength) + edge_col[c] * edge_strength,
            ground_float[:, :, c],
        )
    ground_float[mask] = ground_float[mask] * 0.45
    return np.clip(ground_float, 0, 255).astype(np.uint8)


def detect_hidden_patterns(img_rgb: np.ndarray, mask: np.ndarray,
                            deforested: np.ndarray) -> Tuple[np.ndarray, dict, np.ndarray]:
    gray    = cv2.cvtColor(deforested, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w    = gray.shape

    lap      = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    tex_map  = cv2.GaussianBlur(lap, (0, 0), max(h, w) // 60 or 5)
    tex_norm = tex_map / (tex_map.max() + 1e-6)

    edges     = cv2.Canny(gray.astype(np.uint8), 35, 110).astype(np.float32) / 255.0
    edge_den  = cv2.GaussianBlur(edges, (0, 0), max(h, w) // 50 or 5)
    edge_norm = edge_den / (edge_den.max() + 1e-6)

    sx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    sy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    lin_score = (sx + sy) / (np.abs(sx - sy) + sx + sy + 1e-6)
    lin_norm  = cv2.GaussianBlur(lin_score.astype(np.float32), (0, 0), max(h, w) // 55 or 5)
    lin_norm  = lin_norm / (lin_norm.max() + 1e-6)

    veg_absence = (~mask).astype(np.float32)
    veg_smooth  = cv2.GaussianBlur(veg_absence, (0, 0), max(h, w) // 40 or 8)

    anomaly = (0.30 * tex_norm + 0.30 * edge_norm + 0.20 * lin_norm + 0.20 * veg_smooth)
    anomaly = cv2.GaussianBlur(anomaly, (0, 0), max(h, w) // 35 or 10)
    anomaly = anomaly / (anomaly.max() + 1e-6)

    anom_u8  = (anomaly * 255).astype(np.uint8)
    heatmap  = cv2.cvtColor(cv2.applyColorMap(anom_u8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    blended  = cv2.addWeighted(deforested, 0.50, heatmap, 0.50, 0)

    threshold    = float(np.percentile(anomaly, 88))
    hotspot_mask = anomaly >= threshold
    total_px     = h * w
    hot_pct      = round(hotspot_mask.sum() / total_px * 100, 1)
    hot_u8       = (hotspot_mask * 255).astype(np.uint8)
    n_labels, _  = cv2.connectedComponents(hot_u8)
    struct_count = max(0, n_labels - 1)

    stats = {
        "hotspot_pct":  hot_pct,
        "struct_count": struct_count,
        "mean_anomaly": round(float(anomaly.mean()), 4),
        "peak_anomaly": round(float(anomaly.max()), 4),
        "veg_coverage": round(mask.sum() / total_px * 100, 1),
    }
    return blended, stats, anomaly


def build_vegetation_mask_visual(mask: np.ndarray) -> np.ndarray:
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    vis[mask]  = (60, 160, 60)
    vis[~mask] = (40, 35, 30)
    return vis


def build_deforest_report_widget(location_name, lat_val, lon_val, stats: dict,
                                  vari_threshold: float, intensity: float) -> str:
    loc      = (location_name or "Unknown Site").replace('"', '\\"').replace("'", "\\'")
    hot_pct  = stats["hotspot_pct"]
    structs  = stats["struct_count"]
    veg_cov  = stats["veg_coverage"]
    mean_an  = stats["mean_anomaly"]
    peak_an  = stats["peak_anomaly"]
    ground   = round(100.0 - veg_cov, 1)

    potential = "HIGH" if hot_pct > 20 or structs > 6 else ("MODERATE" if hot_pct > 10 or structs > 3 else "LOW")
    pot_col   = "#d46b6b" if potential == "HIGH" else ("#d4a84b" if potential == "MODERATE" else "#7ec899")

    return f"""<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500&family=JetBrains+Mono:wght@300;400&family=Archivo+Narrow:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:transparent;font-family:'Archivo Narrow',sans-serif;color:#d4cfc8}}
.panel{{background:linear-gradient(160deg,#0e1118 0%,#111520 100%);border:1px solid #1e2535;border-top:2px solid #2a6e4a;border-radius:4px;padding:16px 18px 14px;}}
.panel-label{{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:0.22em;text-transform:uppercase;color:#4a5268;margin-bottom:10px;display:flex;align-items:center;gap:10px;}}
.panel-label::after{{content:'';flex:1;height:1px;background:linear-gradient(90deg,#1e2535,transparent);}}
.data-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-bottom:10px;}}
.dc{{background:rgba(255,255,255,0.03);border:1px solid #1e2535;border-radius:3px;padding:7px 9px;}}
.dcl{{font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:0.13em;text-transform:uppercase;color:#4a5268;margin-bottom:3px;}}
.dcv{{font-family:'Cormorant Garamond',serif;font-size:15px;color:#d4cfc8;}}
.trigger{{display:flex;align-items:center;justify-content:space-between;cursor:pointer;padding:9px 11px;background:rgba(255,255,255,0.02);border:1px solid #1e2535;border-radius:3px;transition:all 0.2s;user-select:none;}}
.trigger:hover{{border-color:#2a6e4a;background:rgba(42,110,74,0.06);}}
.trigger-text{{font-family:'Archivo Narrow',sans-serif;font-size:11px;letter-spacing:0.1em;text-transform:uppercase;color:#7a8399;}}
.trigger-arrow{{font-family:'JetBrains Mono',monospace;font-size:11px;color:#2a6e4a;transition:transform 0.2s;}}
.trigger:hover .trigger-arrow{{color:#4ec98a;}}
#output{{margin-top:10px;display:none}} #output.show{{display:block}}
.loading-row{{display:flex;align-items:center;gap:10px;font-family:'JetBrains Mono',monospace;font-size:10px;color:#4a5268;letter-spacing:0.08em;padding:6px 0;}}
.bar-loader{{flex:1;height:2px;background:#1e2535;border-radius:1px;overflow:hidden;position:relative;}}
.bar-loader::after{{content:'';position:absolute;top:0;left:-40%;width:40%;height:100%;background:linear-gradient(90deg,transparent,#4ec98a,transparent);animation:scan 1.4s linear infinite;}}
@keyframes scan{{to{{left:140%}}}}
#report{{background:rgba(255,255,255,0.025);border:1px solid #1e2535;border-left:3px solid #2a6e4a;border-radius:0 3px 3px 0;padding:12px 14px;font-family:'JetBrains Mono',monospace;font-size:10.5px;line-height:1.95;color:#a0b8a8;white-space:pre-wrap;display:none;}}
.cursor{{display:inline-block;width:6px;height:11px;background:#4ec98a;margin-left:2px;vertical-align:middle;border-radius:1px;animation:blink .5s infinite;}}
@keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:0}}}}
#errmsg{{font-family:'JetBrains Mono',monospace;font-size:10px;color:#d46b6b;padding:7px 10px;background:rgba(124,58,58,0.08);border:1px solid rgba(124,58,58,0.2);border-radius:3px;display:none;}}
</style>
</head>
<body>
<div class="panel">
  <div class="panel-label">AI Deforestation Analysis Report</div>
  <div class="data-grid">
    <div class="dc"><div class="dcl">Veg Coverage</div><div class="dcv">{veg_cov:.1f}%</div></div>
    <div class="dc"><div class="dcl">Ground Exposed</div><div class="dcv">{ground:.1f}%</div></div>
    <div class="dc"><div class="dcl">Hotspot Area</div><div class="dcv">{hot_pct:.1f}%</div></div>
    <div class="dc"><div class="dcl">Hidden Structures</div><div class="dcv" style="color:{pot_col};">{structs}</div></div>
    <div class="dc"><div class="dcl">Mean Anomaly</div><div class="dcv">{mean_an:.4f}</div></div>
    <div class="dc"><div class="dcl">Peak Anomaly</div><div class="dcv">{peak_an:.4f}</div></div>
    <div class="dc" style="grid-column:span 3;background:rgba(42,110,74,0.08);border-color:rgba(42,110,74,0.3);">
      <div class="dcl">Site Potential</div>
      <div class="dcv" style="font-size:13px;color:{pot_col};">{potential} — {structs} anomalous zones in {hot_pct:.1f}% of ground area</div>
    </div>
  </div>
  <div class="trigger" id="triggerBtn" onclick="runReport()">
    <span class="trigger-text" id="hint">Generate AI hidden-ruins analysis</span>
    <span class="trigger-arrow" id="arr">&#x25BA;</span>
  </div>
  <div id="output">
    <div class="loading-row" id="loading"><div class="bar-loader"></div><span>Analysing hidden patterns…</span></div>
    <div id="report"></div>
    <div id="errmsg"></div>
  </div>
</div>
<script>
var busy=false;
var GROQ_KEY="{GROQ_API_KEY}";
var LOC="{loc}",LAT={lat_val:.4f},LON={lon_val:.4f};
var VEG_COV={veg_cov:.1f},GROUND={ground:.1f},HOT_PCT={hot_pct:.1f};
var STRUCTS={structs},MEAN_AN={mean_an:.4f},PEAK_AN={peak_an:.4f};
var POTENTIAL="{potential}",VARI_T={vari_threshold:.2f},INTENSITY={intensity:.2f};

function highlight(txt){{
  return txt.replace(/\\n/g,'<br>')
    .replace(/(SUMMARY|HIDDEN FEATURES|VEGETATION IMPACT|RECOMMENDATION):/g,
      '<span style="font-size:8.5px;letter-spacing:0.2em;color:#4ec98a;font-weight:600;">$1:</span>');
}}
function typewriter(el,text,speed,done){{
  el.innerHTML='<span class="cursor"></span>';var i=0,buf='';
  (function tick(){{
    if(i<text.length){{buf+=text[i++];el.innerHTML=highlight(buf)+'<span class="cursor"></span>';
    setTimeout(tick,(buf[buf.length-1]==='.'||buf[buf.length-1]==='?')?speed*5:speed);}}
    else{{el.innerHTML=highlight(buf);if(done)done();}}
  }})();
}}
async function runReport(){{
  if(busy)return;busy=true;
  var out=document.getElementById('output'),loading=document.getElementById('loading'),
      report=document.getElementById('report'),errmsg=document.getElementById('errmsg'),
      hint=document.getElementById('hint'),arr=document.getElementById('arr');
  report.style.display='none';errmsg.style.display='none';
  loading.style.display='flex';out.classList.add('show');
  hint.textContent='Generating analysis…';arr.style.transform='rotate(90deg)';
  var prompt=
"You are a senior field archaeologist interpreting AI-powered digital deforestation analysis.\\n\\n"+
"DATA:\\n  Site: "+LOC+" ("+LAT.toFixed(4)+"N, "+LON.toFixed(4)+"E)\\n"+
"  Vegetation coverage: "+VEG_COV+"% | Ground exposed: "+GROUND+"%\\n"+
"  Anomaly hotspot area: "+HOT_PCT+"% | Hidden structures: "+STRUCTS+"\\n"+
"  Mean anomaly: "+MEAN_AN+" | Peak: "+PEAK_AN+"\\n"+
"  Site potential: "+POTENTIAL+"\\n\\n"+
"Write EXACTLY 4 labelled lines. No preamble.\\n"+
"SUMMARY: Overall picture — "+VEG_COV+"% vegetation, "+STRUCTS+" anomaly zones, "+POTENTIAL+" potential.\\n"+
"HIDDEN FEATURES: What buried structure types are consistent with "+STRUCTS+" zones at "+HOT_PCT+"% coverage?\\n"+
"VEGETATION IMPACT: How has "+VEG_COV+"% canopy protected or damaged buried structures at "+LOC+"?\\n"+
"RECOMMENDATION: One specific next-step calibrated to "+POTENTIAL+" potential and "+STRUCTS+" anomaly zones.\\n\\n"+
"Rules: Labels uppercase + colon. Plain English. No markdown. Max 45 words per line.";
  try{{
    var resp=await fetch('https://api.groq.com/openai/v1/chat/completions',{{
      method:'POST',
      headers:{{'Content-Type':'application/json','Authorization':'Bearer '+GROQ_KEY}},
      body:JSON.stringify({{model:'llama-3.3-70b-versatile',messages:[{{role:'user',content:prompt}}],temperature:0.4,max_tokens:520,stream:false}})
    }});
    if(!resp.ok)throw new Error('API error '+resp.status);
    var data=await resp.json();
    var full=(data.choices&&data.choices[0]&&data.choices[0].message&&data.choices[0].message.content)||'';
    loading.style.display='none';report.style.display='block';
    typewriter(report,full.trim(),13,function(){{hint.textContent='Report complete — click to regenerate';arr.style.transform='rotate(0deg)';busy=false;}});
  }}catch(err){{
    loading.style.display='none';errmsg.style.display='block';
    errmsg.textContent='Error: '+err.message;hint.textContent='Click to retry';arr.style.transform='rotate(0deg)';busy=false;
  }}
}}
</script>
</body>
</html>"""


# ====================== TAB: DEFORESTATION AI ======================
def tab_deforestation(params):
    st.markdown("## Digital Deforestation & Hidden Ruins Detection")
    st.caption(
        "Upload a satellite or drone image. The AI digitally removes vegetation, "
        "enhances ground visibility, and detects potential buried archaeological structures."
    )

    col_up, col_ctrl = st.columns([3, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Upload satellite / drone image",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            key="deforest_upload",
        )
    with col_ctrl:
        vari_threshold    = st.slider("Vegetation threshold (VARI)", 0.05, 0.50, 0.18, 0.01)
        removal_intensity = st.slider("Removal intensity", 0.10, 1.00, 0.75, 0.05)
        show_veg_mask     = st.checkbox("Show vegetation mask",  value=True)
        enhance_ground    = st.checkbox("Enhance ground detail", value=True)
        run_deforest      = st.button("Run Digital Deforestation", use_container_width=True)

    if not uploaded:
        st.caption("Accepted formats: JPG, PNG, TIF. Recommended resolution: 800–4000 px.")
        st.markdown("""
        <div style="display:flex;gap:18px;margin-top:8px;font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:0.08em;">
          <span style="color:#3cb371;">■ Dense vegetation (masked)</span>
          <span style="color:#d4a84b;">■ Ground / exposed soil</span>
          <span style="color:#ff6b4a;">■ Anomaly hotspot (potential ruin)</span>
        </div>""", unsafe_allow_html=True)
        st.session_state["deforest_results"] = None
        return

    raw = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Cannot decode image. Upload a valid JPG, PNG, or TIF.")
        return

    h, w = img.shape[:2]
    if w > 1280:
        img = cv2.resize(img, (1280, int(h * 1280 / w)))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cached = st.session_state.get("deforest_results")
    should_run = run_deforest or (cached is None)

    if should_run:
        with st.spinner("Computing vegetation mask…"):
            veg_mask, vari_map = generate_vegetation_mask(rgb, vari_threshold)
        with st.spinner("Digitally removing vegetation…"):
            deforested = remove_vegetation(rgb, veg_mask, removal_intensity)
        with st.spinner("Enhancing ground features…"):
            ground_enhanced = enhance_ground_features(rgb, veg_mask) if enhance_ground else deforested.copy()
        with st.spinner("Detecting hidden patterns…"):
            heatmap_img, stats, anomaly_raw = detect_hidden_patterns(rgb, veg_mask, deforested)

        st.session_state["deforest_results"] = {
            "veg_mask":         veg_mask,
            "vari_map":         vari_map,
            "deforested":       deforested,
            "ground_enhanced":  ground_enhanced,
            "heatmap_img":      heatmap_img,
            "stats":            stats,
            "vari_threshold":   vari_threshold,
            "removal_intensity": removal_intensity,
        }
        res = st.session_state["deforest_results"]
    else:
        res = cached
        required_keys = ("veg_mask", "deforested", "ground_enhanced", "heatmap_img", "stats")
        if any(res.get(k) is None for k in required_keys):
            st.warning("Cached results incomplete — please click **Run Digital Deforestation**.")
            return
        veg_mask          = res["veg_mask"]
        vari_map          = res["vari_map"]
        deforested        = res["deforested"]
        ground_enhanced   = res["ground_enhanced"]
        heatmap_img       = res["heatmap_img"]
        stats             = res["stats"]
        vari_threshold    = res["vari_threshold"]
        removal_intensity = res["removal_intensity"]

    stats             = res["stats"]
    veg_mask          = res["veg_mask"]
    vari_map          = res["vari_map"]
    deforested        = res["deforested"]
    ground_enhanced   = res["ground_enhanced"]
    heatmap_img       = res["heatmap_img"]
    vari_threshold    = res["vari_threshold"]
    removal_intensity = res["removal_intensity"]

    st.markdown("### Deforestation Analysis Results")
    potential = ("HIGH"     if stats["hotspot_pct"] > 20 or stats["struct_count"] > 6 else
                 "MODERATE" if stats["hotspot_pct"] > 10 or stats["struct_count"] > 3 else "LOW")
    pot_col   = "#d46b6b" if potential == "HIGH" else ("#d4a84b" if potential == "MODERATE" else "#7ec899")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Veg Coverage",      f"{stats['veg_coverage']:.1f}%")
    m2.metric("Ground Exposed",    f"{round(100 - stats['veg_coverage'], 1):.1f}%")
    m3.metric("Hotspot Area",      f"{stats['hotspot_pct']:.1f}%")
    m4.metric("Hidden Structures", stats["struct_count"])
    m5.metric("Mean Anomaly",      f"{stats['mean_anomaly']:.4f}")
    m6.metric("Site Potential",    potential)

    st.markdown(f"""
    <div style="margin:6px 0 16px;padding:9px 14px;background:rgba(42,110,74,0.07);
                border:1px solid rgba(42,110,74,0.25);border-radius:4px;
                font-family:'JetBrains Mono',monospace;font-size:10px;
                color:var(--text-secondary);letter-spacing:0.07em;">
      Digital deforestation removed <strong style="color:#4ec98a;">{stats['veg_coverage']:.1f}%</strong>
      vegetation &nbsp;|&nbsp;
      <strong style="color:{pot_col};">{stats['struct_count']}</strong> anomalous zones in
      <strong style="color:{pot_col};">{stats['hotspot_pct']:.1f}%</strong> of ground area &nbsp;|&nbsp;
      Site potential: <strong style="color:{pot_col};">{potential}</strong>
    </div>""", unsafe_allow_html=True)

    st.markdown("### Visual Comparison")

    if show_veg_mask:
        cols = st.columns(4)
        panel_titles = ["Original Image", "Vegetation Mask",
                        "Deforestation View", "Ground Enhancement"]
        panel_images = [rgb, build_vegetation_mask_visual(veg_mask), deforested, ground_enhanced]
        panel_caps   = [
            "Input imagery",
            "Green = vegetation  ·  Dark = exposed ground",
            f"Vegetation suppressed at {removal_intensity:.0%} intensity",
            "CLAHE + Sobel edge enhancement",
        ]
    else:
        cols = st.columns(3)
        panel_titles = ["Original Image", "Deforestation View", "Ground Enhancement"]
        panel_images = [rgb, deforested, ground_enhanced]
        panel_caps   = [
            "Input imagery",
            f"Vegetation suppressed at {removal_intensity:.0%} intensity",
            "CLAHE + Sobel edge enhancement",
        ]

    for col, title, img_data, cap in zip(cols, panel_titles, panel_images, panel_caps):
        with col:
            st.markdown(f"#### {title}")
            st.image(img_data, use_container_width=True)
            st.caption(cap)

    st.markdown("### Hidden Structure Anomaly Heatmap")
    st.image(heatmap_img, use_container_width=True)
    st.caption(
        "Brighter/hotter zones = composite of low vegetation + high texture variance + geometric edge density. "
        "These indicate potential buried structures, foundations, or soil/crop mark anomalies."
    )

    with st.expander("VARI Vegetation Index Map"):
        vari_vis = cv2.applyColorMap(
            ((vari_map + 1) / 2 * 255).astype(np.uint8), cv2.COLORMAP_SUMMER
        )
        st.image(cv2.cvtColor(vari_vis, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption(f"VARI threshold: {vari_threshold:.2f}  ·  Pixels above threshold classified as vegetation.")

    st.markdown("### Anomaly Zone Summary")
    st.markdown("""
| Pattern Type | Detection Basis | Archaeological Significance |
|---|---|---|
| Linear edges | Sobel horizontal + vertical gradients | Wall foundations, field boundaries, road alignments |
| Textural anomaly | Laplacian variance spike on ground pixels | Disturbed soil, rubble spreads, buried platforms |
| Colour difference | VARI deviation from surrounding ground | Crop marks, soil marks over buried ditches or walls |
| Geometric density | Connected-component count in edge map | Settlement platforms, terrace systems, courtyard layouts |
    """)

    st.markdown("---")
    col_dl1, col_dl2, col_dl3 = st.columns(3)

    def img_to_bytes(arr: np.ndarray) -> bytes:
        arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png", arr_bgr)
        return buf.tobytes() if ok else b""

    with col_dl1:
        st.download_button("Download Deforestation View (.png)",
                           data=img_to_bytes(deforested),
                           file_name="deforestation_view.png", mime="image/png")
    with col_dl2:
        st.download_button("Download Ground Enhancement (.png)",
                           data=img_to_bytes(ground_enhanced),
                           file_name="ground_enhanced.png", mime="image/png")
    with col_dl3:
        st.download_button("Download Anomaly Heatmap (.png)",
                           data=img_to_bytes(heatmap_img),
                           file_name="anomaly_heatmap.png", mime="image/png")

    with st.expander("Processing Methodology"):
        st.markdown(f"""
| Step | Method | Parameters |
|------|--------|-----------|
| Vegetation detection | VARI index (RGB proxy for NDVI) | Threshold: {vari_threshold:.2f} |
| Vegetation removal | Channel suppression + earth-tone blending | Intensity: {removal_intensity:.0%} |
| Ground enhancement | CLAHE (clip 3.0) + unsharp mask + Canny edges | Tile: 8×8, Canny: 40–120 |
| Anomaly detection | Texture (30%) + Edge density (30%) + Linearity (20%) + Ground exposure (20%) | — |
| Structure counting | Connected-component labelling on top-12% anomaly pixels | — |
        """)

    st.markdown("---")
    st.markdown("### AI Hidden-Ruins Field Report")
    st.components.v1.html(
        build_deforest_report_widget(
            st.session_state.get("location_name", ""),
            params["lat"], params["lon"],
            stats, vari_threshold, removal_intensity,
        ),
        height=430, scrolling=False,
    )


# ====================== TAB: MAP ======================
def tab_map(params):
    st.markdown("## Interactive Map")
    lat      = params["lat"]
    lon      = params["lon"]
    dets     = st.session_state.get("dets", [])
    mound_r  = st.session_state.get("mound_results", [])
    risk     = st.session_state.get("risk", 0.0)
    clr      = "green" if risk < 0.33 else ("orange" if risk < 0.66 else "red")
    loc_name = st.session_state.get("location_name", "")

    st.caption(
        f"Active site: {loc_name}  —  {lat:.6f} N, {lon:.6f} E" if loc_name
        else f"Coordinates: {lat:.6f} N, {lon:.6f} E"
    )

    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        st.error("Invalid coordinates.")
        return

    fmap = folium.Map(
        location=[lat, lon], zoom_start=15,
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
    )
    folium.Marker(
        [lat, lon],
        popup=folium.Popup(
            f"<b>{loc_name or 'Site Origin'}</b><br>Lat: {lat:.6f}<br>Lon: {lon:.6f}<br>Risk: {risk:.2%}",
            max_width=220),
        tooltip=loc_name or "Site Origin",
        icon=folium.Icon(color=clr, icon="university", prefix="fa"),
    ).add_to(fmap)
    for i, d in enumerate(dets):
        folium.CircleMarker(
            [lat + i * 0.0001, lon + i * 0.0001], radius=8,
            color="#c8a14b", fill=True, fill_opacity=0.7,
            popup=f"{d['label']} ({d['conf']:.2%})", tooltip=d["label"],
        ).add_to(fmap)

    color_cls = {"Man-made": "red", "Natural": "green", "Uncertain": "orange"}
    for i, r in enumerate(mound_r):
        if not (r.get("highlight") or r["cls_label"] == "Man-made"):
            continue
        folium.CircleMarker(
            [lat + (i + len(dets)) * 0.00015, lon + (i + len(dets)) * 0.00015],
            radius=10, color=color_cls.get(r["cls_label"], "gray"),
            fill=True, fill_opacity=0.75,
            popup=f"Object #{i+1}: {r['cls_label']} ({r['conf']:.2%})",
            tooltip=f"Detected — {r['cls_label']}",
        ).add_to(fmap)

    folium.Circle([lat, lon], radius=200, color=clr, fill=True, fill_opacity=0.08,
                  popup="Risk zone — 200 m radius").add_to(fmap)
    st_folium(fmap, height=520, use_container_width=True)

    st.markdown("---")
    st.caption("External viewers")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"[Google Maps Satellite](https://www.google.com/maps/@{lat},{lon},17z/data=!3m1!1e3)")
    c2.markdown(f"[Bing Maps](https://www.bing.com/maps?cp={lat}~{lon}&lvl=17&style=a)")
    c3.markdown(f"[OpenStreetMap](https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=17/{lat}/{lon})")
    c4.markdown(f"[Google Earth Web](https://earth.google.com/web/search/{lat},{lon})")
    st.text_input("Coordinates for Google Earth Pro", value=f"{lat:.6f}, {lon:.6f}")


# ====================== TAB: REPORTS ======================
def tab_reports(params):
    st.markdown("## Export & Reports")
    dets    = st.session_state.get("dets", [])
    risk    = st.session_state.get("risk", 0.0)
    vari    = st.session_state.get("vari_mean", 0.0)
    cov     = st.session_state.get("coverage", {})
    mound_r = st.session_state.get("mound_results", [])

    if not cov:
        st.info("Run an analysis first (Analysis tab) to generate export data.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### KMZ Export")
        kml = build_kml(params["lat"], params["lon"], dets, risk)
        st.download_button("Download KMZ", data=build_kmz(kml),
                           file_name="archaeological_site.kmz",
                           mime="application/vnd.google-earth.kmz")
        with st.expander("Preview KML source"):
            st.code(kml, language="xml")

    with c2:
        st.markdown("#### Site Report")
        loc_name = st.session_state.get("location_name", "—")
        lbl      = "LOW" if risk < 0.33 else ("MODERATE" if risk < 0.66 else "HIGH")
        manmade  = sum(1 for r in mound_r if r["cls_label"] == "Man-made")
        lines = (
            ["ARCHAEOLOGICAL SITE ANALYSIS REPORT", "=" * 44,
             f"Location      : {loc_name}",
             f"Coordinates   : {params['lat']:.6f} N, {params['lon']:.6f} E",
             f"Slope         : {params['slope']}°",
             f"Elevation     : {params['elev']} m",
             f"Erosion Risk  : {risk:.2%}  [{lbl}]",
             f"Mean VARI     : {vari:.3f}",
             f"Objects Found : {len(mound_r)} total / {manmade} man-made",
             "", "VEGETATION COVERAGE", "-" * 30]
            + [f"  {k:<14}: {v:.1f}%" for k, v in cov.items()]
            + ["", f"DETECTED ARTIFACTS  ({len(dets)})", "-" * 30]
            + ([f"  [{i+1}] {d['label']:<18} conf={d['conf']:.2%}" for i, d in enumerate(dets)]
               if dets else ["  None detected"])
        )
        report = "\n".join(lines)
        st.text_area("Preview", report, height=360)
        st.download_button("Download Report (.txt)", data=report,
                           file_name="site_report.txt", mime="text/plain")


# ====================== TAB: ABOUT ======================
def tab_about():
    st.markdown("## About")
    st.markdown("""
**Developer:** Hari Krishanan M  
**Platform:** AI-Driven Archaeological Site Analysis  
**Version:** 4.2 — Overlay Fix + Auto Model Download

---

#### Bug Fixes in This Release

| Bug | Root Cause | Fix Applied |
|-----|-----------|-------------|
| Mound section only detected mounds/ruins | Label whitelist filtered all other objects | Removed label filter — ALL YOLO detections accepted; demo mode uses Laplacian blob detection |
| Deforestation `NoneType` subscript error | `res["rgb"]` stored in session state became `None` on page re-render | `rgb` always decoded fresh from the live uploader; session state stores only processed arrays; added `None` guards |
| Sidebar hidden / not visible | Streamlit auto-collapses sidebar on narrow viewports; `collapsedControl` arrow invisible | CSS forces sidebar open (`transform:translateX(0)`, `display:block`); collapse arrow always visible with `z-index:999999` |
| Natural / Uncertain boxes invisible on Classification Overlay | `cv2.addWeighted` on nearly-identical arrays produced invisible boxes (broken alpha-blend) | All boxes drawn directly: Natural/Uncertain get 1px thin border + short label; Man-made gets thick 3px border + filled tag; non-highlighted drawn first so Man-made always on top |
| Model not in repository | `best.pt` too large for GitHub | Auto-download from Google Drive on first run using `gdown` |

---

#### Setup

| Step | Action |
|------|--------|
| 1 | Install dependencies: `pip install streamlit ultralytics folium streamlit-folium opencv-python groq joblib gdown` |
| 2 | Run the app: `streamlit run dashboard_app.py` |
| 3 | Model will auto-download from Google Drive on first run |

---

#### Capabilities

| Module | Description |
|--------|-------------|
| Theme Toggle | Dark / Light via sidebar radio |
| Geocoding | 4-engine fallback: Nominatim, Photon, Structured, India-scoped |
| Artifact Detection | YOLOv11 inference on satellite or drone imagery |
| VARI Index | Visible Atmospherically Resistant Index for vegetation analysis |
| Erosion Risk | Slope + elevation + vegetation composite score |
| Object Detection | ALL objects detected and classified (Man-made / Natural / Uncertain) |
| Deforestation AI | Digital vegetation removal, ground enhancement, anomaly heatmap |
| AI Field Reports | Groq LLaMA 3.3 70B — structured 4–5 line assessments |
| Map | Google Satellite basemap with artifact and object overlays |
| Export | KMZ, plain-text reports, PNG processed images, CSV detection data |

---

#### Object Classification Logic

| Stage | Rule / Feature | Weight |
|-------|---------------|--------|
| Strict rule 1 | VARI > 0.35 → Natural (vegetation present) | Override |
| Strict rule 2 | Shape regularity < 0.45 → Natural (irregular) | Override |
| Shape regularity | Regular bounding box → Man-made | 35% |
| Texture variance | Low Laplacian variance → worked surface | 25% |
| VARI index | Low vegetation → exposed stone/soil | 25% |
| Detection confidence | Model certainty | 15% |

Score > 0.65 → Man-made · Score < 0.45 → Natural · Else → Uncertain

---

#### Overlay Legend (v4.2)

| Visual Style | Class |
|---|---|
| Thick red border + filled label tag | Man-made (priority) |
| Thin green border + short "N xx%" label | Natural |
| Thin gold border + short "U xx%" label | Uncertain |
""")


# ====================== MAIN ======================
def main():
    params = render_sidebar()
    det_model, det_mode  = load_local_yolo(params["local_path"]) if params["local_path"] else (None, None)
    er_model, feat_names = load_erosion_model(params["erosion_path"])

    st.markdown("""
    <div class="title-bar">
      <div class="page-title">ArchAI</div>
      <div class="page-subtitle">Archaeological Intelligence Platform &nbsp;&middot;&nbsp; Hari Krishanan M &nbsp;&middot;&nbsp; v4.2</div>
    </div>""", unsafe_allow_html=True)

    t1, t2, t3, t4, t5, t6 = st.tabs(
        ["Analysis", "Object Detection", "Deforestation AI", "Map", "Reports", "About"]
    )
    with t1: tab_analysis(params, det_model, det_mode, er_model, feat_names)
    with t2: tab_mound_detection(params, det_model, det_mode)
    with t3: tab_deforestation(params)
    with t4: tab_map(params)
    with t5: tab_reports(params)
    with t6: tab_about()

if __name__ == "__main__":
    main()