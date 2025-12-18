import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Fraud Detection (1 file)", layout="wide")
st.title("🚨 Credit Card Fraud Detection — 1 file (train & predict in Streamlit)")

# =========================
# Helpers
# =========================
def preprocess(df: pd.DataFrame):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if "Class" not in df.columns:
        raise ValueError("Dataset phải có cột 'Class' (0/1).")

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    # ép numeric + xử lý NaN/inf
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X, y

def random_from_dataset(dfX: pd.DataFrame):
    r = dfX.sample(1).iloc[0]
    return r.to_dict()

def random_from_meanstd(dfX: pd.DataFrame):
    row = {}
    for c in dfX.columns:
        mu = float(dfX[c].mean())
        sd = float(dfX[c].std(ddof=0))
        if not np.isfinite(sd) or sd == 0:
            row[c] = mu if np.isfinite(mu) else 0.0
        else:
            v = float(np.random.normal(mu, sd))
            row[c] = 0.0 if not np.isfinite(v) else v
    return row

# =========================
# Upload + Read
# =========================
uploaded = st.file_uploader("📤 Upload creditcard.csv", type=["csv"])

if uploaded is None:
    st.info("Hãy upload file creditcard.csv để train mô hình và xem phân tích mô tả.")
    st.stop()

df = pd.read_csv(uploaded)

try:
    X, y = preprocess(df)
except Exception as e:
    st.error(f"❌ Lỗi dữ liệu: {e}")
    st.stop()

st.success(f"✅ Loaded: {len(df)} rows | {X.shape[1]} features")
fraud_n = int((y == 1).sum())
st.caption(f"Tỉ lệ fraud: {fraud_n}/{len(y)} = {fraud_n/len(y)*100:.4f}%")

with st.expander("Xem 5 dòng đầu"):
    st.dataframe(df.head(5), use_container_width=True)

# =========================
# Train parameters (sidebar)
# =========================
st.sidebar.header("⚙️ Train settings")
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.3, 0.05)
C = st.sidebar.number_input("LogisticRegression C", value=1.0, min_value=0.0001, step=0.1, format="%.4f")
max_iter = st.sidebar.number_input("max_iter", value=1000, min_value=100, step=100)

# ✅ bật/tắt Z-score (mặc định bật vì dataset có Time/Amount scale lớn)
use_zscore = st.sidebar.checkbox("Chuẩn hoá Z-score (StandardScaler)", value=True)

@st.cache_resource(show_spinner=True)
def train_model(X: pd.DataFrame, y: pd.Series, test_size: float, C: float, max_iter: int, use_zscore: bool):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = None
    X_train_used = X_train
    X_test_used = X_test

    if use_zscore:
        scaler = StandardScaler()
        X_train_used = scaler.fit_transform(X_train)  # fit trên TRAIN
        X_test_used = scaler.transform(X_test)        # transform TEST

    model = LogisticRegression(C=C, max_iter=max_iter)
    model.fit(X_train_used, y_train)

    y_pred = model.predict(X_test_used)
    acc = accuracy_score(y_test, y_pred)

    try:
        y_proba = model.predict_proba(X_test_used)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = None

    return model, scaler, acc, auc, list(X_train.columns)

model, scaler, acc, auc, feature_cols = train_model(X, y, test_size, C, max_iter, use_zscore)

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy (test)", f"{acc:.6f}")
c2.metric("ROC-AUC (test)", f"{auc:.6f}" if auc is not None else "N/A")
c3.metric("Z-score", "ON ✅" if use_zscore else "OFF ❌")

# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["🔮 Dự đoán giao dịch", "📊 Phân tích mô tả (3.3)"])

# =========================
# TAB 1: Prediction (GIỮ NGUYÊN)
# =========================
with tab1:
    st.subheader("🔮 Dự đoán 1 giao dịch")

    if "tx" not in st.session_state:
        st.session_state.tx = {c: 0.0 for c in feature_cols}

    colA, colB = st.columns([2, 1])

    with colB:
        st.markdown("### 🎲 Tạo dữ liệu nhanh")
        threshold = st.slider("Ngưỡng phân loại (threshold)", 0.05, 0.95, 0.5, 0.01)

        if st.button("🎯 Lấy ngẫu nhiên 1 dòng THẬT", use_container_width=True):
            st.session_state.tx = random_from_dataset(X)
            st.success("Đã lấy 1 giao dịch thật từ dataset!")

        if st.button("🎲 Sinh ngẫu nhiên theo mean/std", use_container_width=True):
            st.session_state.tx = random_from_meanstd(X)
            st.success("Đã sinh ngẫu nhiên theo mean/std!")

    with colA:
        st.markdown("### 🧾 Nhập giao dịch (có thể bấm random để tự điền)")

        # Time + Amount
        if "Time" in feature_cols or "Amount" in feature_cols:
            t1, t2 = st.columns(2)
            if "Time" in feature_cols:
                with t1:
                    st.session_state.tx["Time"] = st.number_input(
                        "Time", value=float(st.session_state.tx.get("Time", 0.0))
                    )
            if "Amount" in feature_cols:
                with t2:
                    st.session_state.tx["Amount"] = st.number_input(
                        "Amount", value=float(st.session_state.tx.get("Amount", 0.0))
                    )

        # V1..V28
        vcols = [c for c in feature_cols if c.startswith("V")]
        if vcols:
            st.markdown("#### V1 ... V28")
            for i in range(0, len(vcols), 4):
                cols = st.columns(4)
                for j, name in enumerate(vcols[i:i + 4]):
                    with cols[j]:
                        st.session_state.tx[name] = st.number_input(
                            name,
                            value=float(st.session_state.tx.get(name, 0.0)),
                            format="%.6f"
                        )
        else:
            st.markdown("#### Các feature")
            for i in range(0, len(feature_cols), 3):
                cols = st.columns(3)
                for j, name in enumerate(feature_cols[i:i + 3]):
                    with cols[j]:
                        st.session_state.tx[name] = st.number_input(
                            name,
                            value=float(st.session_state.tx.get(name, 0.0))
                        )

    if st.button("🔮 Dự đoán", use_container_width=True):
        x_input = np.array([[float(st.session_state.tx[c]) for c in feature_cols]], dtype=float)

        if use_zscore and scaler is not None:
            x_used = scaler.transform(x_input)
        else:
            x_used = x_input

        pred = int(model.predict(x_used)[0])
        proba = float(model.predict_proba(x_used)[0, 1])

        st.markdown("---")
        st.subheader("✅ Kết quả")
        k1, k2, k3 = st.columns(3)
        k1.metric("Fraud probability", f"{proba:.6f}")
        k2.metric("Prediction", "Fraud (1) ❌" if pred == 1 else "Not Fraud (0) ✅")
        k3.metric("Threshold", f"{threshold:.2f}")

        if proba >= threshold:
            st.error("⚠️ Giao dịch có khả năng gian lận.")
        else:
            st.success("✅ Giao dịch có khả năng bình thường.")

        with st.expander("Xem lại dữ liệu giao dịch"):
            st.dataframe(pd.DataFrame([st.session_state.tx])[feature_cols], use_container_width=True)

# =========================
# TAB 2: Descriptive analysis (CHỈ THU NHỎ HÌNH)
# =========================
with tab2:
    st.subheader("📊 3.3 Phân tích mô tả")

    # 1) Count Class
    st.markdown("### 1) Biểu đồ cột đếm số lượng giao dịch hợp lệ và gian lận")
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    sns.countplot(x="Class", data=df, ax=ax)
    ax.set_xlabel("Class (0: Hợp lệ, 1: Gian lận)")
    ax.set_ylabel("Số lượng")
    plt.tight_layout()
    st.pyplot(fig)

    # 2) Amount histogram by Class
    st.markdown("### 2) Histogram phân bố Amount theo loại giao dịch (0/1)")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.histplot(df[df["Class"] == 0]["Amount"], bins=50, label="Hợp lệ (0)", alpha=0.6)
    sns.histplot(df[df["Class"] == 1]["Amount"], bins=50, label="Gian lận (1)", alpha=0.6)
    ax.set_xlabel("Amount")
    ax.set_ylabel("Tần suất")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    # 3) Time histogram by Class
    st.markdown("### 3) Histogram phân bố Time theo loại giao dịch (0/1)")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.histplot(df[df["Class"] == 0]["Time"], bins=50, label="Hợp lệ (0)", alpha=0.6)
    sns.histplot(df[df["Class"] == 1]["Time"], bins=50, label="Gian lận (1)", alpha=0.6)
    ax.set_xlabel("Time")
    ax.set_ylabel("Tần suất")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    # 4) Correlation matrix
    st.markdown("### 4) Ma trận tương quan giữa các biến")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

    # 5) Correlation with Class (bar)
    st.markdown("### 5) Bar chart tương quan giữa các biến và nhãn Class")
    if "Class" in corr.columns:
        corr_class = corr["Class"].drop("Class").sort_values(key=np.abs, ascending=False)
        fig, ax = plt.subplots(figsize=(5.5, 4))
        corr_class.plot(kind="bar", ax=ax)
        ax.set_ylabel("Correlation with Class")
        ax.set_xlabel("Features")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Không tìm thấy cột 'Class' trong ma trận tương quan.")

    # 6) Amount histogram + KDE
    st.markdown("### 6) Histogram + KDE cho Amount (toàn bộ dữ liệu)")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.histplot(df["Amount"], bins=50, kde=True, ax=ax)
    ax.set_xlabel("Amount")
    ax.set_ylabel("Tần suất")
    plt.tight_layout()
    st.pyplot(fig)

    # 7) Boxplot Amount by Class
    st.markdown("### 7) Boxplot so sánh Amount giữa Class=0 và Class=1")
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    sns.boxplot(x="Class", y="Amount", data=df, ax=ax)
    ax.set_xlabel("Class (0: Hợp lệ, 1: Gian lận)")
    ax.set_ylabel("Amount")
    plt.tight_layout()
    st.pyplot(fig)

    
