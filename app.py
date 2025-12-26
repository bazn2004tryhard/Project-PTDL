import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# =========================
#   11 DESCRIPTIVE STATS
#   (Thêm mean, bỏ Q2 vì Q2 = median)
# =========================
STAT_LIST_11 = [
    "count", "min", "max", "mean", "median", "mode",
    "Q1", "Q3", "IQR", "variance", "stdev",
]

# ========== Helpers ==========
def sigmoid(z: np.ndarray) -> np.ndarray:
    # z-score giúp z đỡ bão hoà; vẫn clip cho an toàn
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))

def compute_descriptive_stats_11(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = df[cols].copy()
    for c in cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    out = pd.DataFrame(index=cols)
    out["count"] = X.count()
    out["min"] = X.min()
    out["max"] = X.max()

    out["mean"] = X.mean()
    out["median"] = X.median()
    out["Q1"] = X.quantile(0.25)
    out["Q3"] = X.quantile(0.75)
    out["IQR"] = out["Q3"] - out["Q1"]

    out["variance"] = X.var(ddof=1)
    out["stdev"] = X.std(ddof=1)

    def _mode_first(s: pd.Series):
        m = s.mode(dropna=True)
        return np.nan if len(m) == 0 else m.iloc[0]

    out["mode"] = X.apply(_mode_first, axis=0)

    # Trả về đúng thứ tự 11 độ đo
    return out[STAT_LIST_11]

def plot_stat_bar(stats_df: pd.DataFrame, stat_name: str, top_n: int = 30):
    s = stats_df[stat_name].copy().replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) > top_n:
        s = s.reindex(s.abs().sort_values(ascending=False).head(top_n).index)

    fig, ax = plt.subplots()
    ax.bar(s.index, s.values)
    ax.set_title(f"{stat_name} (top {min(top_n, len(s))} features)")
    ax.tick_params(axis="x", rotation=90)
    st.pyplot(fig)

def plot_overview(stats_df: pd.DataFrame):
    # Normalize theo cột để heatmap dễ nhìn (mỗi stat scale khác nhau)
    data = stats_df.copy().replace([np.inf, -np.inf], np.nan)
    col_std = data.std(axis=0).replace(0, 1)
    data = (data - data.mean(axis=0)) / col_std
    data = data.fillna(0)

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(data.values, aspect="auto")
    ax.set_title("Overview (normalized) — 11 descriptive stats (V1..V28 x stats)")
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    st.pyplot(fig)

def random_transaction_from_meanstd(df_clean: pd.DataFrame, feature_cols: list[str]) -> dict:
    # Random theo mean/std dataset (realistic hơn min/max)
    row = {}
    for c in feature_cols:
        mu = float(df_clean[c].mean())
        sd = float(df_clean[c].std(ddof=0))
        if not np.isfinite(sd) or sd == 0:
            row[c] = mu if np.isfinite(mu) else 0.0
        else:
            v = float(np.random.normal(mu, sd))
            row[c] = 0.0 if not np.isfinite(v) else v
    return row

# =========================
#   PREPROCESSING
# =========================
def preprocess_df(df: pd.DataFrame, feature_cols: list[str], label_candidates=("Class", "label")):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    label_col = None
    for lc in label_candidates:
        if lc in df.columns:
            label_col = lc
            break

    if label_col == "Class":
        df = df.rename(columns={"Class": "label"})
        label_col = "label"

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if label_col is not None and label_col in df.columns:
        df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)

    return df, label_col

def preprocess_single_tx(tx: dict, feature_cols: list[str]) -> dict:
    clean = {}
    for c in feature_cols:
        v = tx.get(c, 0.0)
        try:
            v = float(v)
        except Exception:
            v = 0.0
        if not np.isfinite(v):
            v = 0.0
        clean[c] = v
    return clean

def apply_model_zscore(x: np.ndarray, z_mean: np.ndarray | None, z_std: np.ndarray | None) -> np.ndarray:
    if z_mean is None or z_std is None:
        return x
    z_std_safe = np.where(z_std == 0, 1.0, z_std)
    return (x - z_mean) / z_std_safe

def predict_prob(tx_clean: dict, feature_cols: list[str], coef: np.ndarray, intercept: float,
                 z_mean: np.ndarray | None, z_std: np.ndarray | None) -> float:
    x = np.array([tx_clean[c] for c in feature_cols], dtype=float)
    x = apply_model_zscore(x, z_mean, z_std)  # ✅ z-score giống lúc train
    return float(sigmoid(x @ coef + intercept))

# =========================
#   Streamlit UI
# =========================
st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("🚨 Credit Card Fraud Detection")

# ===== Load model.pkl =====
try:
    with open("model.pkl", "rb") as f:
        artifact = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Không thấy model.pkl. Hãy chạy: python train_spark.py trước để tạo model.pkl")
    st.stop()

feature_cols: list[str] = artifact["feature_cols"]
coef: np.ndarray = np.asarray(artifact["coef"], dtype=float)
intercept: float = float(artifact["intercept"])

z_mean = np.asarray(artifact.get("z_mean"), dtype=float) if artifact.get("z_mean") is not None else None
z_std = np.asarray(artifact.get("z_std"), dtype=float) if artifact.get("z_std") is not None else None
trained_on_zscore = bool(artifact.get("trained_on_zscore", False))
metrics = artifact.get("metrics", {})

st.caption(
    f"Model: Logistic Regression | train_zscore={trained_on_zscore} | "
    f"ROC-AUC={metrics.get('roc_auc','?')} | PR-AUC={metrics.get('pr_auc','?')}"
)

st.sidebar.header("📌 Menu")
mode = st.sidebar.radio("Chọn chức năng", ["🔮 Dự đoán 1 giao dịch", "📊 Phân tích mô tả (11 độ đo)"])

# ========== 1) Prediction ==========
if mode == "🔮 Dự đoán 1 giao dịch":
    st.subheader("🔮 Dự đoán 1 giao dịch có gian lận hay không")

    uploaded = st.file_uploader(
        "📤 Upload creditcard.csv (để lấy giao dịch thật / fraud thật / random theo mean-std)",
        type=["csv"],
        key="pred_upload"
    )

    df_up = None
    label_col = None
    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        df_up, label_col = preprocess_df(df_up, feature_cols)

        if label_col is not None:
            n = len(df_up)
            fraud_n = int((df_up[label_col] == 1).sum())
            st.info(f"Dataset: {n} rows | fraud={fraud_n} ({fraud_n/n*100:.3f}%)")

        with st.expander("Xem nhanh dataset upload (5 dòng đầu)"):
            st.dataframe(df_up.head(5), use_container_width=True)

    if "tx" not in st.session_state:
        st.session_state.tx = {c: 0.0 for c in feature_cols}

    colA, colB = st.columns([2, 1])

    with colB:
        st.markdown("### ⚙️ Hành động")
        threshold = st.slider(
            "Ngưỡng phân loại (threshold)",
            0.05, 0.95,
            float(artifact.get("default_threshold", 0.5)),
            0.01
        )

        if st.button("🎯 Lấy ngẫu nhiên 1 giao dịch THẬT", use_container_width=True):
            if df_up is None:
                st.warning("Bạn cần upload dataset trước.")
            else:
                r = df_up.sample(1).iloc[0]
                st.session_state.tx = {c: float(r[c]) for c in feature_cols}
                st.success("Đã lấy 1 giao dịch thật!")

        if st.button("🧨 Lấy ngẫu nhiên 1 giao dịch FRAUD (label=1)", use_container_width=True):
            if df_up is None or label_col is None:
                st.warning("Bạn cần upload dataset có cột Class/label.")
            else:
                fraud_df = df_up[df_up[label_col] == 1]
                if len(fraud_df) == 0:
                    st.warning("Dataset không có fraud rows.")
                else:
                    r = fraud_df.sample(1).iloc[0]
                    st.session_state.tx = {c: float(r[c]) for c in feature_cols}
                    st.success("Đã lấy 1 giao dịch FRAUD thật!")

        if st.button("🎲 Sinh ngẫu nhiên (theo mean/std)", use_container_width=True):
            if df_up is None:
                st.warning("Bạn cần upload dataset trước.")
            else:
                st.session_state.tx = random_transaction_from_meanstd(df_up, feature_cols)
                st.success("Đã sinh ngẫu nhiên theo mean/std!")

        with st.expander("Demo nhanh: 10 mẫu để bạn thấy có Fraud/Not Fraud"):
            mix = st.radio("Nguồn 10 giao dịch", ["Random mean/std", "5 fraud + 5 non-fraud (từ dataset)"], horizontal=True)
            if st.button("▶️ Chạy demo 10 giao dịch"):
                rows = []
                if mix == "5 fraud + 5 non-fraud (từ dataset)":
                    if df_up is None or label_col is None:
                        st.warning("Cần upload dataset có label.")
                    else:
                        fraud_df = df_up[df_up[label_col] == 1]
                        non_df = df_up[df_up[label_col] == 0]
                        if len(fraud_df) == 0 or len(non_df) == 0:
                            st.warning("Thiếu fraud hoặc non-fraud trong dataset.")
                        else:
                            sample_df = pd.concat(
                                [fraud_df.sample(5, replace=len(fraud_df) < 5),
                                 non_df.sample(5, replace=len(non_df) < 5)],
                                ignore_index=True
                            )
                            for _, r in sample_df.iterrows():
                                tx = {c: float(r[c]) for c in feature_cols}
                                p = predict_prob(tx, feature_cols, coef, intercept, z_mean, z_std)
                                rows.append({"prob": p, "pred": int(p >= threshold), "true_label": int(r[label_col])})
                else:
                    if df_up is None:
                        st.warning("Upload dataset để random mean/std 'đúng phân phối' hơn.")
                    base = df_up if df_up is not None else pd.DataFrame({c: [0.0] for c in feature_cols})
                    for _ in range(10):
                        tx = random_transaction_from_meanstd(base, feature_cols)
                        p = predict_prob(tx, feature_cols, coef, intercept, z_mean, z_std)
                        rows.append({"prob": p, "pred": int(p >= threshold)})

                if rows:
                    demo = pd.DataFrame(rows).sort_values("prob", ascending=False)
                    st.dataframe(demo, use_container_width=True)
                    st.write("Số pred=1:", int((demo["pred"] == 1).sum()))

    with colA:
        st.markdown("### 🧾 Form nhập giao dịch")
        with st.form("tx_form"):
            if "Time" in feature_cols or "Amount" in feature_cols:
                c1, c2 = st.columns(2)
                if "Time" in feature_cols:
                    with c1:
                        st.session_state.tx["Time"] = st.number_input(
                            "Time", value=float(st.session_state.tx.get("Time", 0.0))
                        )
                if "Amount" in feature_cols:
                    with c2:
                        st.session_state.tx["Amount"] = st.number_input(
                            "Amount", value=float(st.session_state.tx.get("Amount", 0.0))
                        )

            st.markdown("#### V1 ... V28")
            vcols = [c for c in feature_cols if c.startswith("V")]
            for i in range(0, len(vcols), 4):
                cols = st.columns(4)
                for j, colname in enumerate(vcols[i:i+4]):
                    with cols[j]:
                        st.session_state.tx[colname] = st.number_input(
                            colname,
                            value=float(st.session_state.tx.get(colname, 0.0)),
                            format="%.6f"
                        )

            submitted = st.form_submit_button("🔮 Dự đoán")

    if submitted:
        tx_clean = preprocess_single_tx(st.session_state.tx, feature_cols)
        st.session_state.tx = tx_clean

        prob = predict_prob(tx_clean, feature_cols, coef, intercept, z_mean, z_std)
        pred = 1 if prob >= threshold else 0

        st.markdown("---")
        st.subheader("✅ Kết quả dự đoán")

        col1, col2, col3 = st.columns(3)
        col1.metric("Fraud probability", f"{prob:.6f}")
        col2.metric("Prediction", "Fraud (1)" if pred == 1 else "Not Fraud (0)")
        col3.metric("Threshold", f"{threshold:.2f}")

        if pred == 1:
            st.error("⚠️ Giao dịch có khả năng **gian lận** (Fraud).")
        else:
            st.success("✅ Giao dịch có khả năng **hợp lệ** (Not Fraud).")

        with st.expander("Xem lại dữ liệu giao dịch đã nhập"):
            st.dataframe(pd.DataFrame([tx_clean])[feature_cols], use_container_width=True)

# ========== 2) Descriptive stats ==========
else:
    st.subheader("📊 Phân tích mô tả (11 độ đo) — chỉ V1..V28 (bỏ Time, Amount)")
    uploaded2 = st.file_uploader("📤 Upload dataset (creditcard.csv)", type=["csv"], key="analysis_upload")
    if uploaded2 is None:
        st.info("Upload dataset để xem phân tích mô tả.")
        st.stop()

    df = pd.read_csv(uploaded2)
    df, _ = preprocess_df(df, feature_cols)

    vcols_only = [c for c in feature_cols if c.startswith("V")]
    Xdf = df[vcols_only].copy()
    stats_df = compute_descriptive_stats_11(Xdf, vcols_only)

    st.write("Chọn độ đo để hiển thị biểu đồ, hoặc chọn tổng quát để xem heatmap 11 độ đo (đã normalize).")
    stat_choice = st.selectbox("Chọn độ đo", ["Tổng quát (11 độ đo)"] + STAT_LIST_11)
    top_n = st.slider("Hiển thị top N feature (bar chart)", 10, 28, 28, 1)

    st.markdown("---")
    if stat_choice == "Tổng quát (11 độ đo)":
        plot_overview(stats_df)
    else:
        plot_stat_bar(stats_df, stat_choice, top_n=top_n)

    st.markdown("### 📋 Bảng thống kê (11 độ đo) — V1..V28")
    st.dataframe(stats_df, use_container_width=True)
