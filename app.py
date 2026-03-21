import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
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
.price-card { border-radius: 16px; padding: 1.8rem; text-align: center; margin: 0.8rem 0; font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1.5rem; border: 1.5px solid; }
.band-0  { background:#f0fff4; color:#22543d; border-color:#68d391; }
.band-1  { background:#f0fff4; color:#276749; border-color:#48bb78; }
.band-2  { background:#ebf8ff; color:#2c5282; border-color:#63b3ed; }
.band-3  { background:#ebf8ff; color:#2a4365; border-color:#4299e1; }
.band-4  { background:#fffff0; color:#744210; border-color:#f6e05e; }
.band-5  { background:#fffaf0; color:#7b341e; border-color:#f6ad55; }
.band-6  { background:#fff5f5; color:#742a2a; border-color:#fc8181; }
.band-7  { background:#fff5f5; color:#63171b; border-color:#f56565; }
.band-8  { background:#faf5ff; color:#44337a; border-color:#b794f4; }
.band-9  { background:#faf5ff; color:#322659; border-color:#9f7aea; }
.band-10 { background:#f7fafc; color:#1a202c; border-color:#a0aec0; }
.band-11 { background:linear-gradient(135deg,#fff9db,#fefcbf); color:#744210; border-color:#f6e05e; }
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

PRICE_BANDS = {0:"Under ₹5,000",1:"₹5,000 – ₹8,000",2:"₹8,000 – ₹10,000",3:"₹10,000 – ₹13,000",4:"₹13,000 – ₹16,000",5:"₹16,000 – ₹20,000",6:"₹20,000 – ₹25,000",7:"₹25,000 – ₹30,000",8:"₹30,000 – ₹40,000",9:"₹40,000 – ₹50,000",10:"₹50,000 – ₹70,000",11:"Above ₹70,000"}
BAND_EMOJIS = {0:"💚",1:"💚",2:"💙",3:"💙",4:"💛",5:"🧡",6:"🔴",7:"🔴",8:"💜",9:"💜",10:"🩶",11:"👑"}
PROCESSORS = {"Snapdragon 8 Gen 3":100,"Apple A17 Bionic":100,"Dimensity 9300":95,"Snapdragon 8 Gen 2":90,"Apple A16 Bionic":88,"Snapdragon 7s Gen 2":75,"Dimensity 8200":72,"Exynos 1380":65,"Snapdragon 695":50,"Dimensity 700":45,"Helio G99":42,"Helio G85":25,"Snapdragon 480":28,"Unisoc T610":20}
BRANDS = ["Samsung","Apple","OnePlus","Xiaomi","Realme","Vivo","Oppo","Nokia","Motorola"]
DISPLAY_TYPES = ["AMOLED","Super AMOLED","IPS LCD","LTPO AMOLED","OLED"]
FINGERPRINT_TYPES = ["In-display","Side","Rear","None"]

@st.cache_resource
def train_model():
    np.random.seed(42)
    N = 2000
    brands_w = {"Samsung":0.18,"Apple":0.12,"OnePlus":0.10,"Xiaomi":0.15,"Realme":0.12,"Vivo":0.10,"Oppo":0.10,"Nokia":0.08,"Motorola":0.05}
    proc_list = list(PROCESSORS.keys())
    proc_scores = np.array(list(PROCESSORS.values()))
    proc_w = proc_scores / proc_scores.sum()
    brand_col = np.random.choice(list(brands_w.keys()), N, p=list(brands_w.values()))
    processor_col = np.random.choice(proc_list, N, p=proc_w)
    processor_score = np.array([PROCESSORS[p] for p in processor_col])
    ram = np.random.choice([2048,3072,4096,6144,8192,12288,16384], N, p=[0.05,0.10,0.20,0.25,0.20,0.15,0.05])
    internal_memory = np.random.choice([32,64,128,256,512,1024], N, p=[0.05,0.15,0.35,0.25,0.15,0.05])
    battery_power = np.random.randint(3000, 6001, N)
    n_cores = np.random.choice([4,6,8], N, p=[0.2,0.3,0.5])
    clock_speed = np.round(np.random.uniform(1.8, 3.2, N), 1)
    primary_cam = np.random.choice([12,48,50,64,108,200], N, p=[0.10,0.20,0.25,0.20,0.15,0.10])
    front_cam = np.random.choice([8,16,20,32,50], N, p=[0.15,0.30,0.25,0.20,0.10])
    ois = np.random.randint(0,2,N)
    night_mode = np.random.randint(0,2,N)
    periscope_zoom = np.random.choice([0,1], N, p=[0.7,0.3])
    refresh_rate = np.random.choice([60,90,120,144], N, p=[0.20,0.25,0.35,0.20])
    five_g = np.random.choice([0,1], N, p=[0.3,0.7])
    wifi_6 = np.random.choice([0,1], N, p=[0.4,0.6])
    nfc = np.random.choice([0,1], N, p=[0.3,0.7])
    fast_charging = np.random.choice([0,18,33,45,67,100,120], N, p=[0.05,0.10,0.20,0.25,0.20,0.10,0.10])
    wireless_charging = np.random.choice([0,1], N, p=[0.5,0.5])
    mobile_wt = np.random.randint(150, 230, N)
    display_type = np.random.choice(["AMOLED","Super AMOLED","IPS LCD","LTPO AMOLED"], N, p=[0.30,0.25,0.25,0.20])
    fingerprint = np.random.choice(["None","Side","In-display","Rear"], N, p=[0.05,0.30,0.40,0.25])
    df = pd.DataFrame({"brand":brand_col,"processor":processor_col,"processor_score":processor_score,"ram":ram,"internal_memory":internal_memory,"battery_power":battery_power,"n_cores":n_cores,"clock_speed":clock_speed,"primary_cam":primary_cam,"front_cam":front_cam,"ois":ois,"night_mode":night_mode,"periscope_zoom":periscope_zoom,"refresh_rate":refresh_rate,"five_g":five_g,"wifi_6":wifi_6,"nfc":nfc,"fast_charging":fast_charging,"wireless_charging":wireless_charging,"mobile_wt":mobile_wt,"display_type":display_type,"fingerprint":fingerprint})
    score = ((df["processor_score"]/100)*35+(df["ram"]/16384)*20+(df["internal_memory"]/1024)*8+(df["primary_cam"]/200)*7+df["five_g"]*5+(df["refresh_rate"]/144)*5+(df["fast_charging"]/120)*5+df["ois"]*3+df["wireless_charging"]*3+df["periscope_zoom"]*3+df["wifi_6"]*2+df["nfc"]*2+(df["battery_power"]/6000)*2+np.random.uniform(0,5,N))
    score = (score-score.min())/(score.max()-score.min())*100
    df["price_band"] = pd.cut(score, bins=[0,8,17,25,33,42,50,58,67,75,83,92,101], labels=list(range(12)))
    df["price_band"] = df["price_band"].cat.codes
    df = df[df["price_band"] >= 0].reset_index(drop=True)
    le_brand = LabelEncoder().fit(df["brand"])
    le_processor = LabelEncoder().fit(df["processor"])
    le_display = LabelEncoder().fit(df["display_type"])
    le_fingerprint = LabelEncoder().fit(df["fingerprint"])
    df["brand_enc"] = le_brand.transform(df["brand"])
    df["processor_enc"] = le_processor.transform(df["processor"])
    df["display_enc"] = le_display.transform(df["display_type"])
    df["fingerprint_enc"] = le_fingerprint.transform(df["fingerprint"])
    FEATURES = ["processor_score","ram","internal_memory","battery_power","n_cores","clock_speed","primary_cam","front_cam","ois","night_mode","periscope_zoom","refresh_rate","five_g","wifi_6","nfc","fast_charging","wireless_charging","mobile_wt","brand_enc","processor_enc","display_enc","fingerprint_enc"]
    X = df[FEATURES]
    y = df["price_band"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42, eval_metric="mlogloss", verbosity=0)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc, le_brand, le_processor, le_display, le_fingerprint, FEATURES

model, accuracy, le_brand, le_processor, le_display, le_fingerprint, FEATURES = train_model()

st.markdown(f'<div class="header-wrap"><h1 class="header-title">📱 Mobile Price Predictor Pro</h1><p class="header-sub">XGBoost · Brand & Processor Aware · 12 Narrow Price Bands</p><div class="accuracy-pill">✅ Model Accuracy: {accuracy:.2%}</div></div>', unsafe_allow_html=True)

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

input_data = pd.DataFrame([{"processor_score":proc_score,"ram":ram,"internal_memory":internal_memory,"battery_power":battery_power,"n_cores":n_cores,"clock_speed":clock_speed,"primary_cam":primary_cam,"front_cam":front_cam,"ois":int(ois),"night_mode":int(night_mode),"periscope_zoom":int(periscope_zoom),"refresh_rate":refresh_rate,"five_g":int(five_g),"wifi_6":int(wifi_6),"nfc":int(nfc),"fast_charging":fast_charging,"wireless_charging":int(wireless_charging),"mobile_wt":mobile_wt,"brand_enc":brand_enc,"processor_enc":processor_enc,"display_enc":display_enc,"fingerprint_enc":fingerprint_enc}])

prediction = int(model.predict(input_data)[0])
proba = model.predict_proba(input_data)[0]
full_proba = np.zeros(12)
for i, c in enumerate(model.classes_):
    if 0 <= c < 12:
        full_proba[c] = proba[i]

band_class = min(prediction, 11)
col_left, col_mid, col_right = st.columns([1.2, 1.3, 1.5])

with col_left:
    st.markdown("### 🎯 Predicted Price Range")
    st.markdown(f'<div class="price-card band-{band_class}">{BAND_EMOJIS[band_class]} {PRICE_BANDS[band_class]}</div>', unsafe_allow_html=True)
    top3 = sorted(enumerate(full_proba), key=lambda x: x[1], reverse=True)[:3]
    st.markdown("**🔍 Top Matches:**")
    for idx, p in top3:
        bar_w = int(p*100)
        color = "#63b3ed" if idx == prediction else "#4a5568"
        st.markdown(f'<div style="margin-bottom:8px"><div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#a0aec0"><span>{PRICE_BANDS[idx]}</span><span>{p:.1%}</span></div><div style="background:#1a202c;border-radius:99px;height:6px;margin-top:3px"><div style="background:{color};width:{bar_w}%;height:6px;border-radius:99px"></div></div></div>', unsafe_allow_html=True)

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
    fig = go.Figure(go.Bar(y=[PRICE_BANDS[i] for i in range(12)], x=full_proba, orientation="h", marker=dict(color=full_proba, colorscale=[[0,"#2d3748"],[0.5,"#2b6cb0"],[1,"#63b3ed"]], showscale=False), text=[f"{p:.1%}" if p>0.03 else "" for p in full_proba], textposition="outside", textfont=dict(size=11)))
    fig.update_layout(height=420, margin=dict(l=0,r=60,t=10,b=10), xaxis=dict(range=[0,max(full_proba)*1.35], showticklabels=False, showgrid=False), yaxis=dict(tickfont=dict(size=10)), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#a0aec0"))
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
    my_score = min(proc_score+min(ram/16384,1)*20+int(five_g)*5+min(refresh_rate/144,1)*5, 100)
    compare = {"Budget Phone":20,"Mid Range":45,"Your Phone":round(my_score,1),"Flagship":95}
    fig3 = go.Figure(go.Bar(x=list(compare.keys()), y=list(compare.values()), marker_color=["#4a5568","#4a5568","#63b3ed","#4a5568"], text=[str(v) for v in compare.values()], textposition="outside", textfont=dict(size=11,color="#e2e8f0")))
    fig3.update_layout(height=320, yaxis=dict(range=[0,120],title="Score",gridcolor="#2d3748",tickfont=dict(color="#a0aec0")), xaxis=dict(tickfont=dict(color="#e2e8f0")), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=40,b=10), font=dict(color="#a0aec0"))
    st.plotly_chart(fig3, use_container_width=True)

st.markdown(f"<p style='text-align:center;color:#4a5568;font-size:0.85rem'>Mobile Price Predictor Pro &nbsp;•&nbsp; XGBoost &nbsp;•&nbsp; 12 Narrow Price Bands &nbsp;•&nbsp; Accuracy: {accuracy:.2%} &nbsp;•&nbsp; Built with ❤️ by Nayan</p>", unsafe_allow_html=True)
