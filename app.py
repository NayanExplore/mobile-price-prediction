"""
app.py  —  Streamlit Web App for Mobile Price Prediction
Run with:  streamlit run app.py
"""

import os, joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ─── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mobile Price Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Space Grotesk', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 16px; margin-bottom: 2rem; text-align: center;
    }
    .main-header h1 { color: #e94560; font-size: 2.5rem; margin: 0; }
    .main-header p  { color: #a0aec0; margin: 0.5rem 0 0; }
    .price-card {
        padding: 1.5rem; border-radius: 12px; text-align: center;
        margin: 1rem 0; font-weight: 700; font-size: 1.8rem;
    }
    .low-price     { background: #e8f5e9; color: #2e7d32; border: 2px solid #66bb6a; }
    .mid-price     { background: #e3f2fd; color: #1565c0; border: 2px solid #42a5f5; }
    .high-price    { background: #fff3e0; color: #e65100; border: 2px solid #ffa726; }
    .premium-price { background: #fce4ec; color: #880e4f; border: 2px solid #ec407a; }
    .metric-box {
        background: #f8fafc; border-radius: 10px; padding: 1rem;
        border-left: 4px solid #e94560; margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── constants ────────────────────────────────────────────────────────────────
PRICE_LABELS = {0: "💚 Low Budget", 1: "💙 Mid Range", 2: "🧡 High End", 3: "❤️ Premium"}
PRICE_STYLES = {0: "low-price", 1: "mid-price", 2: "high-price", 3: "premium-price"}
PRICE_RANGES = {
    0: "Under ₹8,000",
    1: "₹8,000 – ₹20,000",
    2: "₹20,000 – ₹45,000",
    3: "Above ₹45,000"
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/best_model.pkl")
FEATURES = [
    "battery_power","ram","internal_memory","mobile_wt",
    "px_height","px_width","sc_h","sc_w","talk_time",
    "fc","pc","n_cores","clock_speed",
    "blue","dual_sim","four_g","three_g","touch_screen","wifi",
    "pixel_density","screen_area"
]

# ─── load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

# ─── header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📱 Mobile Price Predictor</h1>
    <p>ML-powered price range prediction based on phone specifications</p>
</div>
""", unsafe_allow_html=True)

# ─── sidebar inputs ───────────────────────────────────────────────────────────
st.sidebar.header("📋 Phone Specifications")
st.sidebar.markdown("---")

with st.sidebar:
    st.subheader("🔋 Battery & Performance")
    battery_power  = st.slider("Battery Power (mAh)", 500,  5000, 4000, 100)
    ram            = st.selectbox("RAM (MB)", [512,1024,2048,3072,4096,6144,8192,12288], index=5)
    internal_memory= st.selectbox("Storage (GB)", [8,16,32,64,128,256,512], index=4)
    n_cores        = st.slider("CPU Cores", 1, 8, 8)
    clock_speed    = st.slider("Clock Speed (GHz)", 0.5, 3.0, 2.4, 0.1)
    talk_time      = st.slider("Talk Time (hours)", 2, 24, 18)

    st.subheader("📸 Camera")
    pc = st.slider("Primary Camera (MP)", 0, 64, 48)
    fc = st.slider("Front Camera (MP)", 0, 20, 16)

    st.subheader("📐 Display")
    px_height = st.slider("Resolution Height (px)", 480, 2960, 2400, 20)
    px_width  = st.slider("Resolution Width (px)",  360, 1440, 1080, 20)
    sc_h      = st.slider("Screen Height (cm)", 5, 20, 15)
    sc_w      = st.slider("Screen Width (cm)",  2, 12,  7)

    st.subheader("⚖️ Physical")
    mobile_wt = st.slider("Weight (grams)", 80, 250, 195)

    st.subheader("📡 Connectivity")
    col1, col2 = st.columns(2)
    with col1:
        blue        = st.checkbox("Bluetooth", True)
        dual_sim    = st.checkbox("Dual SIM",  True)
        four_g      = st.checkbox("4G / LTE",  True)
    with col2:
        three_g     = st.checkbox("3G",         True)
        touch_screen= st.checkbox("Touch Screen",True)
        wifi        = st.checkbox("Wi-Fi",       True)

# ─── build input vector ───────────────────────────────────────────────────────
pixel_density = (px_height * px_width) / ((sc_h * sc_w) + 1e-5)
screen_area   = sc_h * sc_w

input_data = pd.DataFrame([{
    "battery_power":  battery_power,
    "ram":            ram,
    "internal_memory":internal_memory,
    "mobile_wt":      mobile_wt,
    "px_height":      px_height,
    "px_width":       px_width,
    "sc_h":           sc_h,
    "sc_w":           sc_w,
    "talk_time":      talk_time,
    "fc":             fc,
    "pc":             pc,
    "n_cores":        n_cores,
    "clock_speed":    clock_speed,
    "blue":           int(blue),
    "dual_sim":       int(dual_sim),
    "four_g":         int(four_g),
    "three_g":        int(three_g),
    "touch_screen":   int(touch_screen),
    "wifi":           int(wifi),
    "pixel_density":  pixel_density,
    "screen_area":    screen_area
}])

# ─── prediction ───────────────────────────────────────────────────────────────
col_pred, col_spec, col_chart = st.columns([1.2, 1.2, 1.6])

with col_pred:
    st.subheader("🎯 Prediction")
    if model is not None:
        prediction = int(model.predict(input_data)[0])
        proba = model.predict_proba(input_data)[0] if hasattr(model, "predict_proba") else None

        st.markdown(f"""
        <div class="price-card {PRICE_STYLES[prediction]}">
            {PRICE_LABELS[prediction]}<br>
            <span style="font-size:1rem; font-weight:400">{PRICE_RANGES[prediction]}</span>
        </div>
        """, unsafe_allow_html=True)

        if proba is not None:
            st.subheader("📊 Confidence")
            labels  = [PRICE_LABELS[i] for i in range(4)]
            colors  = ["#66bb6a","#42a5f5","#ffa726","#ec407a"]
            fig_bar = go.Figure(go.Bar(
                x=proba, y=labels, orientation="h",
                marker_color=colors, text=[f"{p:.1%}" for p in proba],
                textposition="outside"
            ))
            fig_bar.update_layout(
                height=220, margin=dict(l=0,r=40,t=10,b=10),
                xaxis=dict(range=[0,1.1], showticklabels=False),
                plot_bgcolor="white", paper_bgcolor="white"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("⚠️ Model not found. Please run `python train.py` first.")
        st.info("The app will predict once the model is trained.")

with col_spec:
    st.subheader("📱 Spec Summary")
    specs = {
        "🔋 Battery":    f"{battery_power:,} mAh",
        "🧠 RAM":        f"{ram:,} MB ({ram//1024} GB)" if ram >= 1024 else f"{ram} MB",
        "💾 Storage":    f"{internal_memory} GB",
        "⚡ Processor":  f"{n_cores}-core @ {clock_speed} GHz",
        "📷 Camera":     f"{pc}MP + {fc}MP (front)",
        "🖥️ Display":   f"{px_width}×{px_height}px | {sc_w}×{sc_h}cm",
        "⚖️ Weight":     f"{mobile_wt}g",
        "📶 Connectivity": ", ".join(filter(None, [
            "4G" if four_g else "",
            "3G" if three_g else "",
            "BT" if blue else "",
            "WiFi" if wifi else ""
        ]))
    }
    for label, value in specs.items():
        st.markdown(f"""
        <div class="metric-box">
            <strong>{label}</strong><br>{value}
        </div>""", unsafe_allow_html=True)

with col_chart:
    st.subheader("📈 Spec Radar")
    # Normalize values to 0-100 for radar
    radar_values = [
        min(battery_power / 5000, 1) * 100,
        min(ram / 12288, 1) * 100,
        min(internal_memory / 512, 1) * 100,
        min(pc / 64, 1) * 100,
        min(n_cores / 8, 1) * 100,
        min(clock_speed / 3.0, 1) * 100,
        (100 - min(mobile_wt / 250, 1) * 100),  # lighter = better
        min(fc / 20, 1) * 100,
    ]
    radar_cats = ["Battery","RAM","Storage","Camera","Cores","Speed","Light","Front Cam"]
    radar_values_closed = radar_values + [radar_values[0]]
    radar_cats_closed   = radar_cats   + [radar_cats[0]]

    fig_radar = go.Figure(go.Scatterpolar(
        r=radar_values_closed, theta=radar_cats_closed,
        fill="toself", fillcolor="rgba(233,69,96,0.2)",
        line=dict(color="#e94560", width=2),
        marker=dict(size=6, color="#e94560")
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,100])),
        showlegend=False, height=380,
        margin=dict(l=30, r=30, t=30, b=30),
        paper_bgcolor="white"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ─── bottom section ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📚 About This Project")
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.info("""
    **🛠️ Tech Stack**
    - Python + Scikit-learn
    - Random Forest / Gradient Boosting
    - Pandas, NumPy for data processing
    - Streamlit for deployment
    - Plotly for visualization
    """)

with col_b:
    st.success("""
    **📊 Model Performance**
    - 5 models compared
    - Cross-validation used
    - Best model auto-selected
    - Features engineered (pixel density, screen area)
    """)

with col_c:
    st.warning("""
    **🚀 How to Extend**
    - Use real Kaggle dataset
    - Add XGBoost / LightGBM
    - Hyperparameter tuning (GridSearchCV)
    - Deploy to Render / HuggingFace Spaces
    """)

st.markdown(
    "<p style='text-align:center; color:#a0aec0; margin-top:2rem'>"
    "Mobile Price Predictor • Built with ❤️ using Python & Streamlit</p>",
    unsafe_allow_html=True
)
