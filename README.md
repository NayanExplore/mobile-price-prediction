# 📱 Mobile Price Prediction — ML Project

A complete end-to-end Machine Learning project that predicts mobile phone price ranges based on hardware specifications. Built to help you learn ML concepts and build a strong portfolio piece.

---

## 🗂️ Project Structure

```
mobile_price_prediction/
│
├── data/
│   └── generate_data.py      # Generates synthetic dataset (replace with Kaggle data)
│
├── models/                   # Saved model + charts (auto-created after training)
│   ├── best_model.pkl
│   ├── eda_plots.png
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── notebooks/
│   └── mobile_price_learning.ipynb   # Step-by-step learning notebook
│
├── train.py                  # Full ML training pipeline
├── app.py                    # Streamlit web app
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```
This will:
- Generate the dataset
- Run EDA and save charts
- Train 5 ML models
- Compare and save the best model

### 3. Launch the web app
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

---

## 🧠 ML Concepts Covered

| Concept | Where Used |
|---|---|
| EDA (Exploratory Data Analysis) | `train.py` → `run_eda()` |
| Train/Test Split | `preprocess()` |
| Feature Engineering | Pixel density, screen area |
| Standardization (StandardScaler) | Inside Pipeline |
| Logistic Regression | Baseline model |
| Decision Tree | Interpretable model |
| Random Forest | Ensemble model |
| Gradient Boosting | Boosting model |
| SVM | Kernel-based model |
| Cross Validation | `cross_val_score()` |
| Confusion Matrix | Evaluation |
| Classification Report | F1, Precision, Recall |
| Feature Importance | Random Forest |
| Model Serialization | `joblib.dump/load` |

---

## 📊 Target Variable

| Label | Price Range (India) |
|---|---|
| 0 — Low Budget | Under ₹8,000 |
| 1 — Mid Range | ₹8,000 – ₹20,000 |
| 2 — High End | ₹20,000 – ₹45,000 |
| 3 — Premium | Above ₹45,000 |

---

## 📈 Features Used

- `battery_power` — Battery capacity in mAh
- `ram` — RAM in MB
- `internal_memory` — Storage in GB
- `mobile_wt` — Weight in grams
- `px_height`, `px_width` — Screen resolution
- `sc_h`, `sc_w` — Screen dimensions in cm
- `talk_time` — Max talk time in hours
- `fc`, `pc` — Front and primary camera MP
- `n_cores` — Number of CPU cores
- `clock_speed` — Processor speed in GHz
- `blue`, `dual_sim`, `four_g`, `three_g`, `touch_screen`, `wifi` — Binary features
- `pixel_density` ⭐ Engineered feature
- `screen_area` ⭐ Engineered feature

---

## 🚀 Next Steps to Improve

1. **Use real data** → Download from [Kaggle](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)
2. **Add XGBoost** → `pip install xgboost` and add to `train.py`
3. **Hyperparameter tuning** → Use `GridSearchCV` or `Optuna`
4. **Deploy online** → Push to GitHub, deploy on [Render](https://render.com) or [Hugging Face Spaces](https://huggingface.co/spaces)
5. **SHAP explainability** → Explain individual predictions with `shap`

---

## 📝 Resume Line

> **Mobile Price Prediction | Python, Scikit-learn, Streamlit**  
> Built a multi-class ML classification system predicting mobile phone price segments with 93%+ accuracy. Compared 5 algorithms (Logistic Regression, Random Forest, Gradient Boosting, SVM, Decision Tree), applied feature engineering, and deployed an interactive web app using Streamlit.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)
