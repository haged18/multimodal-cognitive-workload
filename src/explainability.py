import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier

os.makedirs("figures", exist_ok=True)
df = pd.read_csv("results/wesad_multimodal.csv").dropna()

feature_cols = [c for c in df.columns if c not in ["subject", "label"]]
X = df[feature_cols]
y_binary = (df["label"] == 2).astype(int)
y_3class = df["label"].map({1: 0, 2: 1, 3: 2})
groups = df["subject"]

logo = LeaveOneGroupOut()
subject_scores = []
shap_values_list = []
X_test_list = []

for train_idx, test_idx in logo.split(X, y_binary, groups):
    sub_id = groups.iloc[test_idx].iloc[0]
    X_train, y_train = X.iloc[train_idx], y_binary.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y_binary.iloc[test_idx]

    model = XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds) * 100
    f1 = f1_score(y_test, preds, zero_division=0) * 100
    subject_scores.append({"Subject": sub_id, "Accuracy": acc, "F1": f1})

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test)
    shap_values_list.append(sv)
    X_test_list.append(X_test)

# --- FIGURE 1: Out-of-Fold SHAP Summary Plot ---
all_shap = np.vstack(shap_values_list)
all_X_test = pd.concat(X_test_list, axis=0)

plt.figure(figsize=(10, 6))
shap.summary_plot(all_shap, all_X_test, show=False, max_display=12)
plt.title("Top Physiological Drivers of Cognitive Workload (Out-of-Fold SHAP)", fontsize=12, pad=12)
plt.tight_layout()
plt.savefig("figures/shap_summary.png", dpi=300, bbox_inches="tight")
plt.close()

# --- FIGURE 2: Per-Subject LOSO Accuracy Bar Chart ---
sub_df = pd.DataFrame(subject_scores)
print("\nPer-Subject LOSO Performance:")
print(sub_df.to_string(index=False))

plt.figure(figsize=(11, 5))
colors = ["#d9534f" if acc < 90 else "#2b5c8f" for acc in sub_df["Accuracy"]]
bars = plt.bar(sub_df["Subject"], sub_df["Accuracy"], color=colors, edgecolor="black", alpha=0.85)
plt.axhline(sub_df["Accuracy"].mean(), color="green", linestyle="--", linewidth=1.5,
            label=f"Mean LOSO Accuracy ({sub_df['Accuracy'].mean():.2f}%)")
plt.ylim(60, 105)
plt.ylabel("Accuracy (%)")
plt.xlabel("Held-Out Test Subject")
plt.title("Leave-One-Subject-Out (LOSO) Generalization Across 15 Subjects")
plt.legend()
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha="center", fontsize=8.5)
plt.tight_layout()
plt.savefig("figures/loso_per_subject.png", dpi=300)
plt.close()

# --- FIGURE 3: 3-Class Confusion Matrix ---
all_preds_3c, all_true_3c = [], []
for train_idx, test_idx in logo.split(X, y_3class, groups):
    clf3 = XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="mlogloss"
    )
    clf3.fit(X.iloc[train_idx], y_3class.iloc[train_idx])
    all_preds_3c.extend(clf3.predict(X.iloc[test_idx]))
    all_true_3c.extend(y_3class.iloc[test_idx])

cm = confusion_matrix(all_true_3c, all_preds_3c, normalize="true") * 100
fig, ax = plt.subplots(figsize=(6.5, 5.5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Baseline", "Cognitive Load", "Amusement"])
disp.plot(cmap="Blues", values_format=".1f", ax=ax, colorbar=False)
plt.title("3-Class Normalized Confusion Matrix (%)", fontsize=12)
plt.tight_layout()
plt.savefig("figures/confusion_matrix_3class.png", dpi=300)
plt.close()

print("\nSaved 3 publication figures in ./figures/:")
print(" - figures/shap_summary.png")
print(" - figures/loso_per_subject.png")
print(" - figures/confusion_matrix_3class.png")