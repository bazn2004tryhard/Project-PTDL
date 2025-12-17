import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ========== Helpers ==========
STAT_LIST = [
    "count", "mean", "std", "min", "25%", "50%", "75%", "max",
    "variance", "skew", "kurtosis", "missing_rate"
]

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50, 50)  # tránh overflow
    return 1.0 / (1.0 + np.exp(-z))

def compute_descriptive_stats(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    desc = df[cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc["variance"] = df[cols].var(numeric_only=True)
    desc["skew"] = df[cols].skew(numeric_only=True)
    desc["kurtosis"] = df[cols].kurtosis(numeric_only=True)
    desc["missing_rate"] = df[cols].isna().mean()
    desc = desc[STAT_LIST]
    return desc

def plot_stat_bar(stats_df: pd.DataFrame, stat_name: str, top_n: int = 30):
    s = stats_df[stat_name].copy()
    s = s.replace([np.inf, -np.inf], np.nan).dropna()

    if len(s) > top_n:
        s = s.reindex(s.abs().sort_values(ascending=False).head(top_n).index)

    fig, ax = plt.subplots()
    ax.bar(s.index, s.values)
    ax.set_title(f"{stat_name} (top {min(top_n, len(s))} features)")
    ax.tick_params(axis="x", rotation=90)
    st.pyplot(fig)

def plot_overview(stats_df: pd.DataFrame):
    # Heatmap đơn giản (chưa normalize)
    data = stats_df.copy()
    data = data.replace([np.inf, -np.inf], np.nan).fillna(0)

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(data.values, aspect="auto")
    ax.set_title("Overview 12 descriptive stats (V1..V28 x stats)")
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    st.pyplot(fig)

def random_transaction_from_minmax(minmax: pd.DataFrame, feature_cols: list[str]) -> dict:
    row = {}
    for c in feature_cols:
        mn = float(minmax.loc[c, "min"])
        mx = float(minmax.loc[c, "max"])
        if not np.isfinite(mn) or not np.isfinite(mx) or mn == mx:
            row[c] = 0.0
        else:
            row[c] = float(np.random.uniform(mn, mx))
    return row


# ========== Streamlit UI ==========
st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("🚨 Credit Card Fraud Detection")
st.caption("Logistic Regression (train bằng PySpark, deploy bằng numpy/pandas).")

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

# Sidebar menu
st.sidebar.header("📌 Menu")
mode = st.sidebar.radio("Chọn chức năng", ["🔮 Dự đoán 1 giao dịch", "📊 Phân tích mô tả"])


# ========== 1) Prediction Mode ==========
if mode == "🔮 Dự đoán 1 giao dịch":
    st.subheader("🔮 Dự đoán 1 giao dịch có gian lận hay không")
    st.write("Bạn có thể **nhập tay** hoặc **sinh ngẫu nhiên** (cần upload dataset để lấy min/max).")

    uploaded = st.file_uploader(
        "📤 (Tuỳ chọn) Upload creditcard.csv để app lấy min/max cho nút random",
        type=["csv"],
        key="pred_upload"
    )

    # chuẩn bị minmax nếu có upload
    minmax = None
    if uploaded is not None:
        df_up = pd.read_csv(uploaded)

        # chỉ lấy các feature có trong file
        present_cols = [c for c in feature_cols if c in df_up.columns]
        mm = pd.DataFrame({
            "min": df_up[present_cols].min(numeric_only=True),
            "max": df_up[present_cols].max(numeric_only=True),
        })

        # với cột thiếu, gán min=max=0
        for c in feature_cols:
            if c not in mm.index:
                mm.loc[c] = {"min": 0.0, "max": 0.0}
        minmax = mm.loc[feature_cols]

    # session state giữ input hiện tại
    if "tx" not in st.session_state:
        st.session_state.tx = {c: 0.0 for c in feature_cols}

    colA, colB = st.columns([2, 1])

    with colB:
        st.markdown("### ⚙️ Hành động")
        if st.button("🎲 Sinh ngẫu nhiên 1 giao dịch", use_container_width=True):
            if minmax is None:
                st.warning("Bạn cần upload dataset để sinh ngẫu nhiên hợp lý (lấy min/max).")
            else:
                st.session_state.tx = random_transaction_from_minmax(minmax, feature_cols)
                st.success("Đã sinh ngẫu nhiên 1 giao dịch!")

        threshold = st.slider("Ngưỡng phân loại (threshold)", 0.05, 0.95, 0.50, 0.01)

    with colA:
        st.markdown("### 🧾 Form nhập giao dịch")
        st.write("Dataset có nhiều cột (V1..V28). Bạn có thể để mặc định 0 nếu không nhập.")

        with st.form("tx_form"):
            # Time + Amount
            c1, c2 = st.columns(2)
            with c1:
                st.session_state.tx["Time"] = st.number_input(
                    "Time",
                    value=float(st.session_state.tx.get("Time", 0.0))
                )
            with c2:
                st.session_state.tx["Amount"] = st.number_input(
                    "Amount",
                    value=float(st.session_state.tx.get("Amount", 0.0))
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
        x = np.array([float(st.session_state.tx.get(c, 0.0)) for c in feature_cols], dtype=float)
        prob = float(sigmoid(x @ coef + intercept))
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
            st.dataframe(pd.DataFrame([st.session_state.tx])[feature_cols], use_container_width=True)


# ========== 2) Descriptive Analysis Mode ==========
else:
    st.subheader("📊 Phân tích mô tả (12 độ đo) — chỉ V1..V28 (bỏ Time, Amount)")

    uploaded2 = st.file_uploader(
        "📤 Upload dataset (creditcard.csv)",
        type=["csv"],
        key="analysis_upload"
    )
    if uploaded2 is None:
        st.info("Upload dataset để xem phân tích mô tả (min/max/mean/std...).")
        st.stop()

    df = pd.read_csv(uploaded2)

    # chỉ lấy V1..V28
    vcols_only = [c for c in feature_cols if c.startswith("V")]

    missing_v = [c for c in vcols_only if c not in df.columns]
    if missing_v:
        st.error(f"❌ Dataset thiếu {len(missing_v)} cột V. Ví dụ: {missing_v[:5]}")
        st.stop()

    Xdf = df[vcols_only].copy()
    stats_df = compute_descriptive_stats(Xdf, vcols_only)

    st.write("Chọn độ đo để hiển thị biểu đồ, hoặc chọn tổng quát để xem heatmap 12 độ đo.")

    stat_choice = st.selectbox("Chọn độ đo", ["Tổng quát (12 độ đo)"] + STAT_LIST)
    top_n = st.slider("Hiển thị top N feature (bar chart)", 10, 28, 28, 1)

    st.markdown("---")
    if stat_choice == "Tổng quát (12 độ đo)":
        plot_overview(stats_df)
    else:
        plot_stat_bar(stats_df, stat_choice, top_n=top_n)

    st.markdown("### 📋 Bảng thống kê (12 độ đo) — V1..V28")
    st.dataframe(stats_df, use_container_width=True)
