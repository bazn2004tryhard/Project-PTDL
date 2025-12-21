import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# =========================
#   11 DESCRIPTIVE STATS
# =========================
STAT_LIST_11 = [
    "count", "min", "max", "median", "mode",
    "Q1", "Q2", "Q3", "IQR", "variance", "stdev",
]

# ========== Helpers ==========
#khai báo hàm nhận vào 1 np array và trả về numpy array
def sigmoid(z: np.ndarray) -> np.ndarray: 
    #giới hạn z nằm khỏng từ -35 đến 35 nếu < -35 thì trả về -35
    z = np.clip(z, -35, 35) 
    # trả về 1 mảng chứa các số nằm từ 0 đến 1: 1/(1+ e^-z) z càng lớn càng gần 1
    return 1.0 / (1.0 + np.exp(-z)) 

def compute_descriptive_stats_11(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    #copy các cột thành 1 df X
    X = df[cols].copy() 
    for c in cols:
        #quy đổi tất cả về dạng số "3.5" = 3.5, "abc" = NaN, inf = inf
        X[c] = pd.to_numeric(X[c], errors="coerce") 
        #thay tất cả các số inf có trong X thành NaN
    X = X.replace([np.inf, -np.inf], np.nan) 

    #tạo ra các dòng df có tiêu đề thuộc cols nhưng không có cột nào
    out = pd.DataFrame(index=cols) 
    out["count"] = X.count() 
    out["min"] = X.min()
    out["max"] = X.max()

    out["median"] = X.median()
    out["Q1"] = X.quantile(0.25)
    out["Q2"] = X.quantile(0.50)
    out["Q3"] = X.quantile(0.75)
    out["IQR"] = out["Q3"] - out["Q1"]

    out["variance"] = X.var(ddof=1)
    out["stdev"] = X.std(ddof=1)

    def _mode_first(s: pd.Series):
        #tính mode và bỏ đi các giá trị NaN
        m = s.mode(dropna=True) 
        # trả về NaN nếu không có mode nào và trả về mode đầu tiên nếu có nhiều mode
        return np.nan if len(m) == 0 else m.iloc[0] 
    #lấy từng cột trong X và đư vào hàm
    out["mode"] = X.apply(_mode_first, axis=0)
    #trả về df có thứ tự cột của biến STAT_LIST_11
    return out[STAT_LIST_11] 
#df cần vẽ, tên cột vẽ, số cột hiến thị
def plot_stat_bar(stats_df: pd.DataFrame, stat_name: str, top_n: int = 30): 
    #lấy các cột cần vẽ, thay inf bằng NaN, loại bỏ giá trị khuyểt
    s = stats_df[stat_name].copy().replace([np.inf, -np.inf], np.nan).dropna() 
    #nếu top_n nhỏ hơn số hiện có
    if len(s) > top_n:
        #chọn cột cần vẽ, lấy abs của từng giá trị sắp xếp giảm dần, lấy top_n index(tên cột) ra
        s = s.reindex(s.abs().sort_values(ascending=False).head(top_n).index)
    #fig : cả khung vẽ, ax: các cấu hình khung vẽ
    fig, ax = plt.subplots() 
    # cấu hình trục x là index("mean", "std", ...) và trục y là giá trị của s
    ax.bar(s.index, s.values) 
    # set tiêu đề cho khung vẽ
    ax.set_title(f"{stat_name} (top {min(top_n, len(s))} features)") 
    # trục x xoay dọc 90 độ
    ax.tick_params(axis="x", rotation=90)
    # vẽ
    st.pyplot(fig) 

def plot_overview(stats_df: pd.DataFrame):
    # Normalize theo cột để heatmap dễ nhìn (mỗi stat scale khác nhau)
    # thay các giá trị inf thành NaN để xử lý
    data = stats_df.copy().replace([np.inf, -np.inf], np.nan) 
    # lấy std của cột, thay 0 thành 1 để tí dưới chia
    col_std = data.std(axis=0).replace(0, 1) 
    #chuẩn háo z_Score
    data = (data - data.mean(axis=0)) / col_std 
    # thay các dòng có giá trị NaN thành 0
    data = data.fillna(0) 

    fig, ax = plt.subplots(figsize=(12, 7))
    # vẽ biểu đồ heatmap, đầu vào là data dữ liệu mảng 2d, tỉ lệ tự căn
    im = ax.imshow(data.values, aspect="auto")
    ax.set_title("Overview (normalized) — 11 descriptive stats (V1..V28 x stats)")
    #cấu hình vị trí tên trục x (cấu hình hình vẽ)
    ax.set_xticks(range(len(data.columns))) 
    ax.set_xticklabels(data.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)
    #faction: diện tích, pad: khoảng cách
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    st.pyplot(fig)

def random_transaction_from_meanstd(df_clean: pd.DataFrame, feature_cols: list[str]) -> dict:
    # Random theo mean/std dataset (realistic hơn min/max)
    row = {}
    for c in feature_cols:
        mu = float(df_clean[c].mean())
        sd = float(df_clean[c].std(ddof=0))
        #kiểm tra nếu sd == 0 hoặc inf hoặc NaN
        if not np.isfinite(sd) or sd == 0: 
            row[c] = mu if np.isfinite(mu) else 0.0
        else:
            # phân phối chuẩn trả về 1 số ngẫu nhiên chênh lệch so với mu k quá độ lệch chuẩn sd
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
    #nếu nhãn đầu vào label_candidates trùng tên với df.columns thì label_col = nhãn và break
    for lc in label_candidates: 
        if lc in df.columns:
            label_col = lc
            break
    #nếu tên nhãn có tên là "Class" thfi đổi tên thành "label"        
    if label_col == "Class": 
        df = df.rename(columns={"Class": "label"})
        label_col = "label"

    for c in feature_cols: #nếu cột trong df bị thiếu thì thêm cột đấy và cho tất cả giá trọ = 0
        if c not in df.columns:
            df[c] = 0.0
    #chuyển tất cả dữ liệu sang dữ liệu số nếu không đc thì trả về NaN
    for c in feature_cols: 
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # nếu có tồn tai inf thì thay bằng NaN sau đó thay tất cả NaN thành 0
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0) 

    if label_col is not None and label_col in df.columns:
        # chuyển đổi hết về dạng số nếu gặp chữ thì chuyển về NaN, thay tất cả NaN = 0 và ép tất cả về int
        df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int) 

    return df, label_col

def preprocess_single_tx(tx: dict, feature_cols: list[str]) -> dict:
    clean = {}
    for c in feature_cols:
        #nếu tx chưa tồn tại key c thì thêm vào và cho nó = 0
        v = tx.get(c, 0.0) 
        # đôit v thành float không đc thì set v = 0, nếu NaN thì không chạy catch
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
    #nếu z_std = 0 thì trả về 1
    z_std_safe = np.where(z_std == 0, 1.0, z_std) 
    return (x - z_mean) / z_std_safe

def predict_prob(tx_clean: dict, feature_cols: list[str], coef: np.ndarray, intercept: float,
                 z_mean: np.ndarray | None, z_std: np.ndarray | None) -> float:
    x = np.array([tx_clean[c] for c in feature_cols], dtype=float)
    # z-score giống lúc train
    x = apply_model_zscore(x, z_mean, z_std)  
    # trả về số từ 0 đến 1 theo hàm sigmoid ở trên và lấy tích vô hướng x*coef + intercept
    return float(sigmoid(x @ coef + intercept)) 

# =========================
#   Streamlit UI
# =========================
# thay đổi tiêu đề tab và chọn layout toàn chiều ngang
st.set_page_config(page_title="Fraud Detection", layout="wide") 
#set title tiêu đề lớn nhất của trang
st.title(" Credit Card Fraud Detection") 

# ===== Load model.pkl =====
try:
    with open("model.pkl", "rb") as f:
        #load file model.plk chứa coef, intercepter z_mean, z_std ...
        artifact = pickle.load(f) 
except FileNotFoundError:
    st.error(" Không thấy model.pkl. Hãy chạy: python train_spark.py trước để tạo model.pkl")
    st.stop()
# lấy các tên cột ra lưu lại và biến có kiểu list[str]
feature_cols: list[str] = artifact["feature_cols"] 
 # lấy coef và chuyển sang np
coef: np.ndarray = np.asarray(artifact["coef"], dtype=float)
#lấy intercepter ra
intercept: float = float(artifact["intercept"]) 

#nếu artifact.get("z_mean") tồn tại thì lấy ra không thì trả về None 
z_mean = np.asarray(artifact.get("z_mean"), dtype=float) if artifact.get("z_mean") is not None else None 
z_std = np.asarray(artifact.get("z_std"), dtype=float) if artifact.get("z_std") is not None else None 
#lấy trained_on_zscore nếu không tồn tại thì trả về false
trained_on_zscore = bool(artifact.get("trained_on_zscore", False)) 
# lấy metric đã tính toán nếu không có trả về {}
metrics = artifact.get("metrics", {}) 

#hiển thị nội dung caption dưới tiêu đề
st.caption(
    f"Model: Logistic Regression | train_zscore={trained_on_zscore} | "
    f"ROC-AUC={metrics.get('roc_auc','?')} | PR-AUC={metrics.get('pr_auc','?')}"
) 

# tiêu đề trên sidebar
st.sidebar.header(" Menu") 
#tạo button các lựa chọn 
mode = st.sidebar.radio("Chọn chức năng", [" Dự đoán 1 giao dịch", " Phân tích mô tả (11 độ đo)"]) 

# ========== 1) Prediction ==========
if mode == "🔮 Dự đoán 1 giao dịch":
    st.subheader("🔮 Dự đoán 1 giao dịch có gian lận hay không")
    st.write(" Fraud cực hiếm → bạn random 10 giao dịch THẬT mà toàn Not Fraud là **bình thường**.")

    uploaded = st.file_uploader(
        " Upload creditcard.csv (để lấy giao dịch thật / fraud thật / random theo mean-std)",
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
            #số giao dịch có label = 1
            fraud_n = int((df_up[label_col] == 1).sum()) 
            #set hiển thị banner màu xanh
            st.info(f"Dataset: {n} rows | fraud={fraud_n} ({fraud_n/n*100:.3f}%)") 

        with st.expander("Xem nhanh dataset upload (5 dòng đầu)"):
            st.dataframe(df_up.head(5), use_container_width=True)
    #nếu thuộc tính tx chưa tồn tại trong st.session_state thì khởi tạo tất =0
    if "tx" not in st.session_state: 
        st.session_state.tx = {c: 0.0 for c in feature_cols}

    #chia làm 2phaanf giao diện phần 1 chiếm 2 phần , phần 2 chiếm 1 phần
    colA, colB = st.columns([2, 1]) 

    # tạo giao diện cho phần thứ 2
    with colB: 
        st.markdown("### Hành động")
        threshold = st.slider(
            "Ngưỡng phân loại (threshold)",
            0.05, 0.95,
            float(artifact.get("default_threshold", 0.5)),
            0.01
        )

        if st.button(" Lấy ngẫu nhiên 1 giao dịch THẬT", use_container_width=True):
            if df_up is None:
                st.warning("Bạn cần upload dataset trước.")
            else:
                r = df_up.sample(1).iloc[0]
                st.session_state.tx = {c: float(r[c]) for c in feature_cols}
                st.success("Đã lấy 1 giao dịch thật!")

        if st.button(" Lấy ngẫu nhiên 1 giao dịch FRAUD (label=1)", use_container_width=True):
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

        if st.button(" Sinh ngẫu nhiên (theo mean/std)", use_container_width=True):
            if df_up is None:
                st.warning("Bạn cần upload dataset trước.")
            else:
                st.session_state.tx = random_transaction_from_meanstd(df_up, feature_cols)
                st.success("Đã sinh ngẫu nhiên theo mean/std!")

        with st.expander("Demo nhanh: 10 mẫu để bạn thấy có Fraud/Not Fraud"):
            mix = st.radio("Nguồn 10 giao dịch", ["Random mean/std", "5 fraud + 5 non-fraud (từ dataset)"], horizontal=True)
            if st.button(" Chạy demo 10 giao dịch"):
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
                            )#nối 2 df với nhau lấy mỗi bên 5, nếu chưa đủ 5 thì tiếp tục lấy cho phép lặp
                            for _, r in sample_df.iterrows():
                                tx = {c: float(r[c]) for c in feature_cols}
                                p = predict_prob(tx, feature_cols, coef, intercept, z_mean, z_std)
                                rows.append({"prob": p, "pred": int(p >= threshold), "true_label": int(r[label_col])}) #đưa ra kết quả làm tròn theo threshold
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
        st.markdown("### Form nhập giao dịch")
        with st.form("tx_form"):
            if "Time" in feature_cols or "Amount" in feature_cols:
                c1, c2 = st.columns(2)
                if "Time" in feature_cols:
                    with c1:
                        st.session_state.tx["Time"] = st.number_input("Time", value=float(st.session_state.tx.get("Time", 0.0)))
                if "Amount" in feature_cols:
                    with c2:
                        st.session_state.tx["Amount"] = st.number_input("Amount", value=float(st.session_state.tx.get("Amount", 0.0)))

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
        st.subheader("Kết quả dự đoán")

        col1, col2, col3 = st.columns(3)
        col1.metric("Fraud probability", f"{prob:.6f}")
        col2.metric("Prediction", "Fraud (1)" if pred == 1 else "Not Fraud (0)")
        col3.metric("Threshold", f"{threshold:.2f}")

        if pred == 1:
            st.error("Giao dịch có khả năng **gian lận** (Fraud).")
        else:
            st.success("Giao dịch có khả năng **hợp lệ** (Not Fraud).")

        with st.expander("Xem lại dữ liệu giao dịch đã nhập"):
            st.dataframe(pd.DataFrame([tx_clean])[feature_cols], use_container_width=True)

# ========== 2) Descriptive stats ==========
else:
    st.subheader("Phân tích mô tả (11 độ đo) — chỉ V1..V28 (bỏ Time, Amount)")
    uploaded2 = st.file_uploader("Upload dataset (creditcard.csv)", type=["csv"], key="analysis_upload")
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

    st.markdown("### Bảng thống kê (11 độ đo) — V1..V28")
    st.dataframe(stats_df, use_container_width=True)
