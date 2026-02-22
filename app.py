import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Mobile Price Predictor", page_icon="📱", layout="wide")

st.markdown("""
<style>
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

PRICE_LABELS = {0: "💚 Low Budget", 1: "💙 Mid Range", 2: "🧡 High End", 3: "❤️ Premium"}
PRICE_STYLES = {0: "low-price", 1: "mid-price", 2: "high-price", 3: "premium-price"}
PRICE_RANGES = {0: "Under ₹8,000", 1: "₹8,000 – ₹20,000", 2: "₹20,000 – ₹45,000", 3: "Above ₹45,000"}

FEATURES = [
    "battery_power", "ram", "internal_memory", "mobile_wt",
    "px_height", "px_width", "sc_h", "sc_w", "talk_time",
    "fc", "pc", "n_cores", "clock_speed",
    "blue", "dual_sim", "four_g", "three_g", "touch_screen", "wifi"
]

@st.cache_resource
def train_model():
    np.random.seed(42)
    N = 2000
    data = {
        "battery_power":   np.random.randint(500, 5001, N),
        "ram":             np.random.choice([512, 1024, 2048, 3072, 4096, 6144, 8192, 12288], N),
        "internal_memory": np.random.choice([8, 16, 32, 64, 128, 256, 512], N),
        "mobile_wt":       np.random.randint(80, 250, N),
        "px_height":       np.random.randint(480, 2960, N),
        "px_width":        np.random.randint(360, 1440, N),
        "sc_h":            np.random.randint(5, 20, N),
        "sc_w":            np.random.randint(2, 12, N),
        "talk_time":       np.random.randint(2, 25, N),
        "fc":              np.random.randint(0, 20, N),
        "pc":              np.random.randint(0, 64, N),
        "n_cores":         np.random.randint(1, 9, N),
        "clock_speed":     np.round(np.random.uniform(0.5, 3.0, N), 1),
        "blue":            np.random.randint(0, 2, N),
        "dual_sim":        np.random.randint(0, 2, N),
        "four_g":          np.random.randint(0, 2, N),
        "three_g":         np.random.randint(0, 2, N),
        "touch_screen":    np.random.randint(0, 2, N),
        "wifi":            np.random.randint(0, 2, N),
    }
    df = pd.DataFrame(data)
    score = (
        (df["ram"] / 12288) * 40 +
        (df["battery_power"] / 5000) * 15 +
        (df["internal_memory"] / 512) * 10 +
        (df["pc"] / 64) * 10 +
        (df["n_cores"] / 8) * 10 +
        df["four_g"] * 5 +
        (df["clock_speed"] / 3.0) * 5 +
        (df["fc"] / 20) * 5 +
        np.random.uniform(0, 5, N)
    )
    df["price_range"] = pd.cut(score, bins=[0, 25, 50, 75, 101], labels=[0, 1, 2, 3]).astype(int)
    X = df[FEATURES].copy()
    y = df["price_range"].copy()
    X["pixel_density"] = (X["px_height"] * X["px_width"]) / ((X["sc_h"] * X["sc_w"]) + 1e-5)
    X["screen_area"] = X["sc_h"] * X["sc_w"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

model, accuracy = train_model()

st.markdown("""
<div class="main-header">
    <h1>📱 Mobile Price Predictor</h1>
    <p>ML-powered price range prediction based on phone specifications</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("📋 Phone Specifications")
st.sidebar.markdown(f"✅ Model Accuracy: **{accuracy:.2%}**")
st.sidebar.markdown("---")

with st.sidebar:
    st.subheader("🔋 Battery & Performance")
    battery_power   = st.slider("Battery Power (mAh)", 500, 5000, 4000, 100)
    ram             = st.selectbox("RAM (MB)", [512, 1024, 2048, 3072, 4096, 6144, 8192, 12288], index=5)
    internal_memory = st.selectbox("Storage (GB)", [8, 16, 32, 64, 128, 256, 512], index=4)
    n_cores         = st.slider("CPU Cores", 1, 8, 8)
    clock_speed     = st.slider("Clock Speed (GHz)", 0.5, 3.0, 2.4, 0.1)
    talk_time       = st.slider("Talk Time (hours)", 2, 24, 18)
    st.subheader("📸 Camera")
    pc = st.slider("Primary Camera (MP)", 0, 64, 48)
    fc = st.slider("Front Camera (MP)", 0, 20, 16)
    st.subheader("📐 Display")
    px_height = st.slider("Resolution Height", 480, 2960, 2400, 20)
    px_width  = st.slider("Resolution Width", 360, 1440, 1080, 20)
    sc_h      = st.slider("Screen Height (cm)", 5, 20, 15)
    sc_w      = st.slider("Screen Width (cm)", 2, 12, 7)
    st.subheader("⚖️ Physical")
    mobile_wt = st.slider("Weight (grams)", 80, 250, 195)
    st.subheader("📡 Connectivity")
    col1, col2 = st.columns(2)
    with col1:
        blue         = st.checkbox("Bluetooth", True)
        dual_sim     = st.checkbox("Dual SIM", True)
        four_g       = st.checkbox("4G", True)
    with col2:
        three_g      = st.checkbox("3G", True)
        touch_screen = st.checkbox("Touch Screen", True)
        wifi         = st.checkbox("Wi-Fi", True)

pixel_density = (px_height * px_width) / ((sc_h * sc_w) + 1e-5)
screen_area   = sc_h * sc_w

input_data = pd.DataFrame([{
    "battery_power":  battery_power,
    "ram":            ram,
    "internal_memory": internal_memory,
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

prediction = int(model.predict(input_data)[0])
proba      = model.predict_proba(input_data)[0]

col_pred, col_chart = st.columns([1, 1.5])

with col_pred:
    st.subheader("🎯 Prediction")
    st.markdown(f"""
    <div class="price-card {PRICE_STYLES[prediction]}">
        {PRICE_LABELS[prediction]}<br>
        <span style="font-size:1rem;font-weight:400">{PRICE_RANGES[prediction]}</span>
    </div>""", unsafe_allow_html=True)

    st.subheader("📊 Confidence")
    colors = ["#66bb6a", "#42a5f5", "#ffa726", "#ec407a"]
    fig = go.Figure(go.Bar(
        x=proba,
        y=[PRICE_LABELS[i] for i in range(4)],
        orientation="h",
        marker_color=colors,
        text=[f"{p:.1%}" for p in proba],
        textposition="outside"
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=0, r=40, t=10, b=10),
        xaxis=dict(range=[0, 1.1], showticklabels=False),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    st.plotly_chart(fig, use_container_width=True)

with col_chart:
    st.subheader("📈 Spec Radar")
    radar_values = [
        min(battery_power / 5000, 1) * 100,
        min(ram / 12288, 1) * 100,
        min(internal_memory / 512, 1) * 100,
        min(pc / 64, 1) * 100,
        min(n_cores / 8, 1) * 100,
        min(clock_speed / 3.0, 1) * 100,
        (100 - min(mobile_wt / 250, 1) * 100),
        min(fc / 20, 1) * 100,
    ]
    cats = ["Battery", "RAM", "Storage", "Camera", "Cores", "Speed", "Light", "Front Cam"]
    rv = radar_values + [radar_values[0]]
    rc = cats + [cats[0]]
    fig2 = go.Figure(go.Scatterpolar(
        r=rv,
        theta=rc,
        fill="toself",
        fillcolor="rgba(233,69,96,0.2)",
        line=dict(color="#e94560", width=2),
        marker=dict(size=6, color="#e94560")
    ))
    fig2.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=380,
        margin=dict(l=30, r=30, t=30, b=30),
        paper_bgcolor="white"
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.markdown(
    f"<p style='text-align:center;color:#a0aec0'>Mobile Price Predictor • Built with ❤️ using Python & Streamlit • Accuracy: {accuracy:.2%}</p>",
    unsafe_allow_html=True
)
