import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

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
# Upload + Train
# =========================
uploaded = st.file_uploader("📤 Upload creditcard.csv", type=["csv"])

if uploaded is None:
    st.info("Hãy upload file creditcard.csv để train mô hình và dự đoán.")
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

# Train parameters
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

st.markdown("---")
st.subheader("🔮 Dự đoán 1 giao dịch")

# =========================
# Input transaction (with random buttons)
# =========================
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

    # Hiển thị Time + Amount (nếu có)
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

    # V1..V28 (nếu có)
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
        # fallback: nếu dataset không theo V1..V28 thì show tất cả columns
        st.markdown("#### Các feature")
        for i in range(0, len(feature_cols), 3):
            cols = st.columns(3)
            for j, name in enumerate(feature_cols[i:i + 3]):
                with cols[j]:
                    st.session_state.tx[name] = st.number_input(
                        name,
                        value=float(st.session_state.tx.get(name, 0.0))
                    )

# =========================
# Predict
# =========================
if st.button("🔮 Dự đoán", use_container_width=True):
    # đảm bảo đúng thứ tự feature
    x_input = np.array([[float(st.session_state.tx[c]) for c in feature_cols]], dtype=float)

    # ✅ nếu dùng Z-score thì transform input giống lúc train
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
