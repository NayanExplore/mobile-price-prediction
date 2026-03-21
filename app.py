import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib
import requests
import os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Mobile Price Predictor Pro", page_icon="📱", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.header-wrap { background: linear-gradient(135deg, #0a0a0f 0%, #111128 50%, #0d1117 100%); border: 1px solid rgba(99,179,237,0.15); border-radius: 20px; padding: 2.5rem 2rem; text-align: center; margin-bottom: 1.5rem; }
.header-title { font-family: 'Syne', sans-serif; font-size: 2.6rem; font-weight: 800; background: linear-gradient(90deg, #63b3ed, #76e4f7, #b794f4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; }
.header-sub { color: #718096; margin: 0.4rem 0 0; font-size: 1rem; }
.accuracy-pill { background: linear-gradient(90deg, #2d3748, #1a202c); border: 1px solid rgba(99,179,237,0.3); border-radius: 99px; padding: 0.4rem 1rem; font-size: 0.85rem; color: #63b3ed; font-weight: 600; display: inline-block; margin-top: 0.5rem; }
.price-card { border-radius: 16px; padding: 1.8rem; text-align: center; margin: 0.8rem 0; font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1.6rem; border: 1.5px solid; }
.band-0 { background:#f0fff4; color:#22543d; border-color:#68d391; }
.band-1 { background:#ebf8ff; color:#2c5282; border-color:#63b3ed; }
.band-2 { background:#fffaf0; color:#7b341e; border-color:#f6ad55; }
.band-3 { background:linear-gradient(135deg,#fff9db,#fefcbf); color:#744210; border-color:#f6e05e; }
.spec-card { background: #0d1117; border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 0.9rem 1rem; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center; }
.spec-label { color: #718096; font-size: 0.85rem; }
.spec-value { color: #e2e8f0; font-weight: 500; font-size: 0.9rem; }
.badge { display: inline-block; padding: 0.2rem 0.7rem; border-radius: 99px; font-size: 0.75rem; font-weight: 600; margin: 0.15rem; }
.badge-blue { background:#2b6cb0; color:#bee3f8; }
.badge-green { background:#276749; color:#c6f6d5; }
.badge-purple { background:#553c9a; color:#e9d8fd; }
.badge-orange { background:#c05621; color:#feebc8; }
</style>
""", unsafe_allow_html=True)

PRICE_BANDS = {
    0: "💚 Budget — Under ₹10,000",
    1: "💙 Mid Range — ₹10,000 – ₹25,000",
    2: "🧡 High End — ₹25,000 – ₹55,000",
    3: "👑 Premium — Above ₹55,000",
}
BAND_STYLES = {0:"band-0", 1:"band-1", 2:"band-2", 3:"band-3"}

PROCESSORS = {
    "Snapdragon 8 Gen 3":100, "Apple A17 Bionic":100, "Dimensity 9300":95,
    "Snapdragon 8 Gen 2":90, "Apple A16 Bionic":88, "Snapdragon 7s Gen 2":75,
    "Dimensity 8200":72, "Exynos 1380":65, "Snapdragon 695":50,
    "Dimensity 700":45, "Helio G99":42, "Helio G85":25,
    "Snapdragon 480":28, "Unisoc T610":20,
}
BRANDS = ["Samsung","Apple","OnePlus","Xiaomi","Realme","Vivo","Oppo","Nokia","Motorola"]
DISPLAY_TYPES = ["AMOLED","Super AMOLED","IPS LCD","LTPO AMOLED","OLED"]
FINGERPRINT_TYPES = ["In-display","Side","Rear","None"]

@st.cache_resource
def load_model():
    bundle = joblib.load("model_bundle.pkl")
    return (bundle["model"], bundle["accuracy"], bundle["le_brand"],
            bundle["le_processor"], bundle["le_display"],
            bundle["le_fingerprint"], bundle["features"])

model, accuracy, le_brand, le_processor, le_display, le_fingerprint, FEATURES = load_model()

st.markdown(f'<div class="header-wrap"><h1 class="header-title">📱 Mobile Price Predictor Pro</h1><p class="header-sub">XGBoost · Brand & Processor Aware · 4 Price Bands</p><div class="accuracy-pill">✅ Model Accuracy: {accuracy:.2%}</div></div>', unsafe_allow_html=True)

st.sidebar.markdown("## 🔧 Configure Your Phone")
with st.sidebar:
    st.markdown("### 📱 Brand & Processor")
    brand = st.selectbox("Brand", BRANDS)
    processor = st.selectbox("Processor", list(PROCESSORS.keys()))
    proc_score = PROCESSORS[processor]
    st.markdown(f"<small style='color:#63b3ed'>Processor Score: **{proc_score}/100**</small>", unsafe_allow_html=True)
    st.markdown("### 🧠 Memory & Storage")
    ram = st.selectbox("RAM", [2048,3072,4096,6144,8192,12288,16384], format_func=lambda x: f"{x//1024} GB" if x>=1024 else f"{x} MB", index=3)
    internal_memory = st.selectbox("Storage", [32,64,128,256,512,1024], format_func=lambda x: f"{x} GB", index=3)
    st.markdown("### 🔋 Battery & Charging")
    battery_power = st.slider("Battery (mAh)", 3000, 6000, 4500, 100)
    fast_charging = st.select_slider("Fast Charging (W)", [0,18,33,45,67,100,120], value=45)
    wireless_charging = st.checkbox("Wireless Charging", True)
    st.markdown("### 📷 Camera")
    primary_cam = st.select_slider("Main Camera (MP)", [12,48,50,64,108,200], value=50)
    front_cam = st.select_slider("Front Camera (MP)", [8,16,20,32,50], value=16)
    ois = st.checkbox("OIS", True)
    night_mode = st.checkbox("Night Mode", True)
    periscope_zoom = st.checkbox("Periscope Zoom", False)
    st.markdown("### 🖥️ Display")
    display_type = st.selectbox("Display Type", DISPLAY_TYPES)
    refresh_rate = st.select_slider("Refresh Rate (Hz)", [60,90,120,144], value=120)
    st.markdown("### ⚡ Performance")
    n_cores = st.select_slider("CPU Cores", [4,6,8], value=8)
    clock_speed = st.slider("Clock Speed (GHz)", 1.8, 3.2, 2.8, 0.1)
    st.markdown("### 📡 Connectivity")
    c1, c2 = st.columns(2)
    with c1:
        five_g = st.checkbox("5G", True)
        wifi_6 = st.checkbox("Wi-Fi 6", True)
    with c2:
        nfc = st.checkbox("NFC", True)
    st.markdown("### 🔒 Security & Build")
    fingerprint = st.selectbox("Fingerprint", FINGERPRINT_TYPES)
    mobile_wt = st.slider("Weight (grams)", 150, 230, 185)

try: brand_enc = le_brand.transform([brand])[0]
except: brand_enc = 0
try: processor_enc = le_processor.transform([processor])[0]
except: processor_enc = 0
try: display_enc = le_display.transform([display_type])[0]
except: display_enc = 0
try: fingerprint_enc = le_fingerprint.transform([fingerprint])[0]
except: fingerprint_enc = 0

input_data = pd.DataFrame([{
    "processor_score":proc_score, "ram":ram, "internal_memory":internal_memory,
    "battery_power":battery_power, "n_cores":n_cores, "clock_speed":clock_speed,
    "primary_cam":primary_cam, "front_cam":front_cam, "ois":int(ois),
    "night_mode":int(night_mode), "periscope_zoom":int(periscope_zoom),
    "refresh_rate":refresh_rate, "five_g":int(five_g), "wifi_6":int(wifi_6),
    "nfc":int(nfc), "fast_charging":fast_charging,
    "wireless_charging":int(wireless_charging), "mobile_wt":mobile_wt,
    "brand_enc":brand_enc, "processor_enc":processor_enc,
    "display_enc":display_enc, "fingerprint_enc":fingerprint_enc,
}])

prediction = int(model.predict(input_data)[0])
proba = model.predict_proba(input_data)[0]
full_proba = np.zeros(4)
for i, c in enumerate(model.classes_):
    if 0 <= c < 4:
        full_proba[c] = proba[i]

band_class = min(prediction, 3)
col_left, col_mid, col_right = st.columns([1.2, 1.3, 1.5])

with col_left:
    st.markdown("### 🎯 Predicted Price Range")
    st.markdown(f'<div class="price-card {BAND_STYLES[band_class]}">{PRICE_BANDS[band_class]}</div>', unsafe_allow_html=True)
    st.markdown("**🔍 Confidence Breakdown:**")
    for idx in range(4):
        p = full_proba[idx]
        bar_w = int(p*100)
        color = "#63b3ed" if idx == prediction else "#4a5568"
        st.markdown(f'<div style="margin-bottom:8px"><div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#a0aec0"><span>{PRICE_BANDS[idx]}</span><span>{p:.1%}</span></div><div style="background:#1a202c;border-radius:99px;height:8px;margin-top:3px"><div style="background:{color};width:{bar_w}%;height:8px;border-radius:99px"></div></div></div>', unsafe_allow_html=True)

with col_mid:
    st.markdown("### 📋 Spec Summary")
    def spec_row(label, value):
        st.markdown(f'<div class="spec-card"><span class="spec-label">{label}</span><span class="spec-value">{value}</span></div>', unsafe_allow_html=True)
    spec_row("📱 Brand", brand)
    spec_row("⚡ Processor", processor)
    spec_row("🧠 RAM", f"{ram//1024} GB" if ram>=1024 else f"{ram} MB")
    spec_row("💾 Storage", f"{internal_memory} GB")
    spec_row("🔋 Battery", f"{battery_power} mAh")
    spec_row("⚡ Charging", f"{fast_charging}W" + (" + Wireless" if wireless_charging else ""))
    spec_row("📷 Camera", f"{primary_cam}MP + {front_cam}MP front")
    spec_row("🖥️ Display", f"{display_type} · {refresh_rate}Hz")
    spec_row("🔧 Proc Score", f"{proc_score}/100")
    badges = []
    if five_g: badges.append('<span class="badge badge-blue">5G</span>')
    if wifi_6: badges.append('<span class="badge badge-blue">Wi-Fi 6</span>')
    if nfc: badges.append('<span class="badge badge-green">NFC</span>')
    if ois: badges.append('<span class="badge badge-purple">OIS</span>')
    if night_mode: badges.append('<span class="badge badge-purple">Night Mode</span>')
    if periscope_zoom: badges.append('<span class="badge badge-orange">Periscope Zoom</span>')
    if wireless_charging: badges.append('<span class="badge badge-green">Wireless Charging</span>')
    st.markdown("<div style='margin-top:0.8rem'>" + " ".join(badges) + "</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("### 📊 Confidence Chart")
    colors = ["#68d391","#63b3ed","#f6ad55","#f6e05e"]
    fig = go.Figure(go.Bar(
        y=[PRICE_BANDS[i] for i in range(4)],
        x=full_proba, orientation="h",
        marker_color=colors,
        text=[f"{p:.1%}" if p>0.02 else "" for p in full_proba],
        textposition="outside", textfont=dict(size=12)
    ))
    fig.update_layout(height=280, margin=dict(l=0,r=60,t=10,b=10), xaxis=dict(range=[0,max(full_proba)*1.35], showticklabels=False, showgrid=False), yaxis=dict(tickfont=dict(size=11)), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#a0aec0"))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
col_radar, col_compare = st.columns([1,1])

with col_radar:
    st.markdown("### 📡 Spec Radar")
    rv = [proc_score, min(ram/16384,1)*100, min(internal_memory/1024,1)*100, min(primary_cam/200,1)*100, min(battery_power/6000,1)*100, min(fast_charging/120,1)*100, min(refresh_rate/144,1)*100, (int(five_g)+int(wifi_6)+int(nfc))/3*100]
    cats = ["Processor","RAM","Storage","Camera","Battery","Charging","Display","Connectivity"]
    fig2 = go.Figure(go.Scatterpolar(r=rv+[rv[0]], theta=cats+[cats[0]], fill="toself", fillcolor="rgba(99,179,237,0.15)", line=dict(color="#63b3ed",width=2), marker=dict(size=5,color="#63b3ed")))
    fig2.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100],tickfont=dict(size=8),gridcolor="#2d3748"),angularaxis=dict(gridcolor="#2d3748"),bgcolor="rgba(0,0,0,0)"), showlegend=False, height=350, margin=dict(l=40,r=40,t=30,b=30), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#a0aec0"))
    st.plotly_chart(fig2, use_container_width=True)

with col_compare:
    st.markdown("### 🏆 How It Compares")
    my_score = min(proc_score + min(ram/16384,1)*20 + int(five_g)*5 + min(refresh_rate/144,1)*5, 100)
    compare = {"Budget Phone":20, "Mid Range":45, "Your Phone":round(my_score,1), "Flagship":95}
    fig3 = go.Figure(go.Bar(x=list(compare.keys()), y=list(compare.values()), marker_color=["#4a5568","#4a5568","#63b3ed","#4a5568"], text=[str(v) for v in compare.values()], textposition="outside", textfont=dict(size=11,color="#e2e8f0")))
    fig3.update_layout(height=320, yaxis=dict(range=[0,120],title="Score",gridcolor="#2d3748",tickfont=dict(color="#a0aec0")), xaxis=dict(tickfont=dict(color="#e2e8f0")), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=40,b=10), font=dict(color="#a0aec0"))
    st.plotly_chart(fig3, use_container_width=True)

st.markdown(f"<p style='text-align:center;color:#4a5568;font-size:0.85rem'>Mobile Price Predictor Pro &nbsp;•&nbsp; XGBoost &nbsp;•&nbsp; 4 Price Bands &nbsp;•&nbsp; Accuracy: {accuracy:.2%} &nbsp;•&nbsp; Built with ❤️ by Nayan</p>", unsafe_allow_html=True)
