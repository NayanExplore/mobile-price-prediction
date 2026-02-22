"""
train.py
--------
Full ML pipeline:
  1. Load data
  2. EDA summary
  3. Preprocessing
  4. Train multiple models
  5. Evaluate & compare
  6. Save best model
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, joblib
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# ── 0. paths ─────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), "data/mobile_data.csv")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

PRICE_LABELS = {0: "Low", 1: "Mid", 2: "High", 3: "Premium"}
FEATURES = [
    "battery_power","ram","internal_memory","mobile_wt",
    "px_height","px_width","sc_h","sc_w","talk_time",
    "fc","pc","n_cores","clock_speed",
    "blue","dual_sim","four_g","three_g","touch_screen","wifi"
]

# ── 1. load ───────────────────────────────────────────────────────────────────
def load_data():
    if not os.path.exists(DATA_PATH):
        # auto-generate if not present
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        sys.path.insert(0, data_dir)
        import importlib.util
        spec = importlib.util.spec_from_file_location("generate_data", os.path.join(data_dir,"generate_data.py"))
        gd = importlib.util.module_from_spec(spec); spec.loader.exec_module(gd)
        df = gd.generate_mobile_data()
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)
    print(f"📦 Loaded data: {df.shape}")
    return df

# ── 2. EDA ────────────────────────────────────────────────────────────────────
def run_eda(df):
    print("\n─── EDA ───")
    print(df.describe().T[["mean","std","min","max"]].round(2))
    print("\nPrice Range Distribution:")
    print(df["price_range"].map(PRICE_LABELS).value_counts())

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Mobile Price Prediction — EDA", fontsize=16, fontweight="bold")

    # price distribution
    counts = df["price_range"].value_counts().sort_index()
    axes[0,0].bar([PRICE_LABELS[i] for i in counts.index], counts.values,
                  color=["#4CAF50","#2196F3","#FF9800","#F44336"])
    axes[0,0].set_title("Price Range Distribution")
    axes[0,0].set_ylabel("Count")

    # RAM vs price
    df.boxplot(column="ram", by="price_range", ax=axes[0,1])
    axes[0,1].set_title("RAM by Price Range")
    axes[0,1].set_xticklabels([PRICE_LABELS[i] for i in range(4)])
    axes[0,1].set_xlabel("")

    # battery vs price
    df.boxplot(column="battery_power", by="price_range", ax=axes[0,2])
    axes[0,2].set_title("Battery Power by Price Range")
    axes[0,2].set_xticklabels([PRICE_LABELS[i] for i in range(4)])
    axes[0,2].set_xlabel("")

    # correlation heatmap
    corr = df[FEATURES + ["price_range"]].corr()
    top_features = corr["price_range"].abs().sort_values(ascending=False)[1:11].index
    sns.heatmap(df[top_features].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", ax=axes[1,0], cbar=False)
    axes[1,0].set_title("Top Feature Correlations")

    # internal memory vs price
    df.boxplot(column="internal_memory", by="price_range", ax=axes[1,1])
    axes[1,1].set_title("Internal Memory by Price Range")
    axes[1,1].set_xticklabels([PRICE_LABELS[i] for i in range(4)])
    axes[1,1].set_xlabel("")

    # feature importance (correlation with target)
    importance = corr["price_range"].abs().sort_values(ascending=False)[1:11]
    axes[1,2].barh(importance.index[::-1], importance.values[::-1], color="#5C6BC0")
    axes[1,2].set_title("Feature Correlation with Price")

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "eda_plots.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("📊 EDA plots saved.")

# ── 3. preprocess ─────────────────────────────────────────────────────────────
def preprocess(df):
    X = df[FEATURES].copy()
    y = df["price_range"].copy()

    # Feature engineering
    X["pixel_density"] = (X["px_height"] * X["px_width"]) / ((X["sc_h"] * X["sc_w"]) + 1e-5)
    X["screen_area"]   = X["sc_h"] * X["sc_w"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

# ── 4. models ─────────────────────────────────────────────────────────────────
def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(max_depth=10, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=150, random_state=42))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", random_state=42))
        ]),
    }

# ── 5. train & evaluate ───────────────────────────────────────────────────────
def train_evaluate(models, X_train, X_test, y_train, y_test):
    results = {}
    print("\n─── Training Models ───")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cv  = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()
        results[name] = {"model": model, "accuracy": acc, "cv_accuracy": cv}
        print(f"  {name:25s} → Test: {acc:.4f} | CV: {cv:.4f}")

    # comparison chart
    names = list(results.keys())
    accs  = [results[n]["accuracy"] for n in names]
    cvs   = [results[n]["cv_accuracy"] for n in names]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - 0.2, accs, 0.4, label="Test Accuracy", color="#42A5F5")
    bars2 = ax.bar(x + 0.2, cvs,  0.4, label="CV Accuracy",   color="#66BB6A")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0.5, 1.05)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.legend()
    for bar in list(bars1) + list(bars2):
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "model_comparison.png"), dpi=120, bbox_inches="tight")
    plt.close()

    return results

# ── 6. save best model ────────────────────────────────────────────────────────
def save_best(results, X_test, y_test):
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best      = results[best_name]
    print(f"\n🏆 Best Model: {best_name} ({best['accuracy']:.4f})")

    y_pred = best["model"].predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=[PRICE_LABELS[i] for i in range(4)]))

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=[PRICE_LABELS[i] for i in range(4)])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # feature importance (Random Forest only)
    if best_name == "Random Forest":
        fi = best["model"].named_steps["clf"].feature_importances_
        fn = list(FEATURES) + ["pixel_density", "screen_area"]
        fi_series = pd.Series(fi, index=fn).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 7))
        fi_series.plot(kind="barh", ax=ax, color="#7E57C2")
        ax.set_title("Feature Importances — Random Forest", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, "feature_importance.png"), dpi=120, bbox_inches="tight")
        plt.close()
        print("📊 Feature importance chart saved.")

    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump(best["model"], model_path)
    print(f"💾 Model saved → {model_path}")
    return best_name, best["accuracy"]

# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df              = load_data()
    run_eda(df)
    X_train, X_test, y_train, y_test = preprocess(df)
    models          = build_models()
    results         = train_evaluate(models, X_train, X_test, y_train, y_test)
    best_name, acc  = save_best(results, X_test, y_test)
    print(f"\n✅ Pipeline complete. Best: {best_name} | Accuracy: {acc:.2%}")
