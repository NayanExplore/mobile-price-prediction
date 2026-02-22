"""
generate_data.py
----------------
Generates a synthetic mobile phone dataset for training.
In a real project, replace this with the Kaggle dataset:
https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000

def generate_mobile_data(n=N):
    data = {
        "battery_power":  np.random.randint(500,  5001, n),   # mAh
        "ram":            np.random.choice([512, 1024, 2048, 3072, 4096, 6144, 8192, 12288], n),  # MB
        "internal_memory":np.random.choice([8, 16, 32, 64, 128, 256, 512], n),  # GB
        "mobile_wt":      np.random.randint(80,   250,  n),   # grams
        "px_height":      np.random.randint(480,  2960, n),
        "px_width":       np.random.randint(360,  1440, n),
        "sc_h":           np.random.randint(5,    20,   n),   # screen cm height
        "sc_w":           np.random.randint(2,    12,   n),   # screen cm width
        "talk_time":      np.random.randint(2,    25,   n),   # hours
        "fc":             np.random.randint(0,    20,   n),   # front camera MP
        "pc":             np.random.randint(0,    64,   n),   # primary camera MP
        "n_cores":        np.random.randint(1,    9,    n),
        "clock_speed":    np.round(np.random.uniform(0.5, 3.0, n), 1),
        "blue":           np.random.randint(0,    2,    n),   # Bluetooth
        "dual_sim":       np.random.randint(0,    2,    n),
        "four_g":         np.random.randint(0,    2,    n),
        "three_g":        np.random.randint(0,    2,    n),
        "touch_screen":   np.random.randint(0,    2,    n),
        "wifi":           np.random.randint(0,    2,    n),
    }

    df = pd.DataFrame(data)

    # Create price_range based on realistic rules (0=low, 1=mid, 2=high, 3=premium)
    score = (
        (df["ram"] / 12288) * 40 +
        (df["battery_power"] / 5000) * 15 +
        (df["internal_memory"] / 512) * 10 +
        (df["pc"] / 64) * 10 +
        (df["n_cores"] / 8) * 10 +
        (df["four_g"]) * 5 +
        (df["clock_speed"] / 3.0) * 5 +
        (df["fc"] / 20) * 5 +
        np.random.uniform(0, 5, n)   # noise
    )

    bins = [0, 25, 50, 75, 101]
    labels = [0, 1, 2, 3]
    df["price_range"] = pd.cut(score, bins=bins, labels=labels).astype(int)

    return df

if __name__ == "__main__":
    df = generate_mobile_data()
    df.to_csv("mobile_data.csv", index=False)
    print(f"✅ Dataset saved: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df["price_range"].value_counts().sort_index())
