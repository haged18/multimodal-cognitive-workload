import os
import pickle
import warnings
import numpy as np
import pandas as pd
import neurokit2 as nk

warnings.filterwarnings("ignore")

SUBJECTS = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S13", "S14", "S15", "S16", "S17"]
FS_CHEST = 700
FS_WRIST = 4
WINDOW_SEC = 60
STEP_SEC = 30

def extract_subject(sub):
    print(f"Processing {sub}...")
    with open(f"WESAD/{sub}/{sub}.pkl", "rb") as f:
        data = pickle.load(f, encoding="latin1")

    labels = data["label"]
    ecg = data["signal"]["chest"]["ECG"].flatten()
    resp = data["signal"]["chest"]["Resp"].flatten()
    eda = data["signal"]["wrist"]["EDA"].flatten()
    temp = data["signal"]["wrist"]["TEMP"].flatten()

    # 1. Process full-recording signals before windowing
    ecg_clean = nk.ecg_clean(ecg, sampling_rate=FS_CHEST)
    _, peaks = nk.ecg_peaks(ecg_clean, sampling_rate=FS_CHEST)
    r_peaks = peaks["ECG_R_Peaks"]
    heart_rate = nk.signal_rate(r_peaks, sampling_rate=FS_CHEST, desired_length=len(ecg))

    rsp_df, _ = nk.rsp_process(resp, sampling_rate=FS_CHEST)
    eda_df, _ = nk.eda_process(eda, sampling_rate=FS_WRIST)

    win_chest = WINDOW_SEC * FS_CHEST
    step_chest = STEP_SEC * FS_CHEST
    rows = []

    for start in range(0, len(labels) - win_chest, step_chest):
        end = start + win_chest
        win_labels = labels[start:end]

        # Keep pure windows for Baseline (1), Cognitive Load/Stress (2), Amusement (3)
        unique = np.unique(win_labels)
        if len(unique) != 1 or unique[0] not in [1, 2, 3]:
            continue

        # --- CARDIAC (ECG) ---
        win_peaks = r_peaks[(r_peaks >= start) & (r_peaks < end)] - start
        if len(win_peaks) < 35:
            continue

        hrv_t = nk.hrv_time(win_peaks, sampling_rate=FS_CHEST, show=False)
        hrv_f = nk.hrv_frequency(win_peaks, sampling_rate=FS_CHEST, show=False)
        win_hr = heart_rate[start:end]

        # --- RESPIRATION (700 Hz) ---
        rsp_win = rsp_df.iloc[start:end]

        # --- WRIST EDA & TEMP (4 Hz) ---
        w_start = int((start / FS_CHEST) * FS_WRIST)
        w_end = int((end / FS_CHEST) * FS_WRIST)
        eda_win = eda_df.iloc[w_start:w_end]
        temp_win = temp[w_start:w_end]

        temp_slope = np.polyfit(np.arange(len(temp_win)), temp_win, 1)[0] if len(temp_win) > 1 else 0.0

        rows.append({
            "subject": sub,
            "label": int(unique[0]),
            # Cardiac features
            "hr_mean": np.mean(win_hr),
            "hr_std": np.std(win_hr),
            "hrv_rmssd": hrv_t["HRV_RMSSD"].values[0],
            "hrv_sdnn": hrv_t["HRV_SDNN"].values[0],
            "hrv_pnn50": hrv_t["HRV_pNN50"].values[0],
            "hrv_lf": hrv_f["HRV_LF"].values[0],
            "hrv_hf": hrv_f["HRV_HF"].values[0],
            "hrv_lf_hf": hrv_f["HRV_LFHF"].values[0],
            # Electrodermal features
            "eda_tonic_mean": eda_win["EDA_Tonic"].mean(),
            "eda_tonic_std": eda_win["EDA_Tonic"].std(),
            "eda_phasic_mean": eda_win["EDA_Phasic"].mean(),
            "eda_phasic_max": eda_win["EDA_Phasic"].max(),
            "sweat_peaks": eda_win["SCR_Peaks"].sum(),
            # Respiration features
            "rsp_rate_mean": rsp_win["RSP_Rate"].mean(),
            "rsp_rate_std": rsp_win["RSP_Rate"].std(),
            "rsp_amp_mean": rsp_win["RSP_Amplitude"].mean(),
            # Temperature features
            "temp_mean": np.mean(temp_win),
            "temp_slope": temp_slope,
        })

    df_sub = pd.DataFrame(rows)
    df_sub.replace([np.inf, -np.inf], np.nan, inplace=True)
    feature_cols = [c for c in df_sub.columns if c not in ["subject", "label"]]
    df_sub[feature_cols] = df_sub[feature_cols].fillna(df_sub[feature_cols].median())

    # Subject-wise Z-score normalization
    df_sub[feature_cols] = (df_sub[feature_cols] - df_sub[feature_cols].mean()) / (df_sub[feature_cols].std() + 1e-8)
    return df_sub

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    all_dfs = [extract_subject(s) for s in SUBJECTS]
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv("results/wesad_multimodal.csv", index=False)
    print(f"\nDone! Saved results/wesad_multimodal.csv with {len(final_df)} rows and {final_df.shape[1] - 2} features.")