import os
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

os.makedirs("results", exist_ok=True)
df = pd.read_csv("results/wesad_multimodal.csv").dropna()
groups = df["subject"]

MODALITIES = {
    "1. Cardiac Only (ECG)": [
        "hr_mean", "hr_std", "hrv_rmssd", "hrv_sdnn", "hrv_pnn50", "hrv_lf", "hrv_hf", "hrv_lf_hf"
    ],
    "2. Electrodermal Only (EDA)": [
        "eda_tonic_mean", "eda_tonic_std", "eda_phasic_mean", "eda_phasic_max", "sweat_peaks"
    ],
    "3. Respiration & Temp": [
        "rsp_rate_mean", "rsp_rate_std", "rsp_amp_mean", "temp_mean", "temp_slope"
    ],
    "4. Wearable Wrist Only (EDA + Temp)": [
        "eda_tonic_mean", "eda_tonic_std", "eda_phasic_mean", "eda_phasic_max", "sweat_peaks", "temp_mean", "temp_slope"
    ],
    "5. Full Multimodal Fusion (All 18 Features)": [
        c for c in df.columns if c not in ["subject", "label"]
    ]
}

def evaluate_loso(X, y, groups, is_multiclass=False):
    logo = LeaveOneGroupOut()
    all_preds, all_true = [], []

    for train_idx, test_idx in logo.split(X, y, groups):
        model = XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss" if is_multiclass else "logloss"
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])

        all_preds.extend(preds)
        all_true.extend(y.iloc[test_idx])

    acc = accuracy_score(all_true, all_preds) * 100
    avg_type = "macro" if is_multiclass else "binary"
    f1 = f1_score(all_true, all_preds, average=avg_type) * 100
    return round(acc, 2), round(f1, 2)

y_binary = (df["label"] == 2).astype(int)
y_3class = df["label"].map({1: 0, 2: 1, 3: 2})

results = []
print("Running Leave-One-Subject-Out (LOSO) Multimodal Ablation Study...\n")

for name, cols in MODALITIES.items():
    X_sub = df[cols]
    bin_acc, bin_f1 = evaluate_loso(X_sub, y_binary, groups, is_multiclass=False)
    tri_acc, tri_f1 = evaluate_loso(X_sub, y_3class, groups, is_multiclass=True)

    results.append({
        "Modality Setup": name,
        "Features": len(cols),
        "Binary Acc (%)": bin_acc,
        "Binary F1 (%)": bin_f1,
        "3-Class Acc (%)": tri_acc,
        "3-Class Macro F1 (%)": tri_f1
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_csv("results/ablation_results.csv", index=False)
print("\nSaved results to results/ablation_results.csv")