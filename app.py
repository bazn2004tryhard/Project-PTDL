import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ===== JAVA & SPARK =====
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-17"
os.environ["SPARK_HOME"] = r"C:\spark-4.0.1-bin-hadoop3"
os.environ["PATH"] = rf"{os.environ['JAVA_HOME']}\bin;{os.environ['SPARK_HOME']}\bin;" + os.environ["PATH"]

from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler

MODEL_DIR = "spark_lr_model"
ASSEMBLER_DIR = "spark_assembler"

# ===== UI CONFIG =====
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

st.title("🚨 Credit Card Fraud Detection")
st.caption("PySpark Logistic Regression + Streamlit")

# ===== SPARK =====
@st.cache_resource
def get_spark():
    spark = SparkSession.builder.master("local[*]").appName("fraud-app").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

@st.cache_resource
def load_model():
    model = LogisticRegressionModel.load(MODEL_DIR)
    assembler = VectorAssembler.load(ASSEMBLER_DIR)
    return model, assembler

if not (os.path.exists(MODEL_DIR) and os.path.exists(ASSEMBLER_DIR)):
    st.error("❌ Chưa có model. Hãy chạy train_spark.py trước.")
    st.stop()

spark = get_spark()
model, assembler = load_model()

# ===== UPLOAD =====
st.subheader("📤 Upload dữ liệu giao dịch (CSV)")
uploaded = st.file_uploader(
    "File CSV có các cột: Time, V1..V28, Amount, Class (hoặc label)",
    type=["csv"]
)

if uploaded is None:
    st.info("⬆️ Upload file CSV để bắt đầu")
    st.stop()

# ===== READ CSV =====
tmp_path = "uploaded.csv"
with open(tmp_path, "wb") as f:
    f.write(uploaded.getbuffer())

df = spark.read.csv(tmp_path, inferSchema=True, header=True)

if "Class" in df.columns and "label" not in df.columns:
    df = df.withColumnRenamed("Class", "label")

# ===== FEATURE VECTOR =====
feature_cols = [c for c in df.columns if c != "label"]
assembler2 = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_feat = assembler2.transform(df)

# ===== PREDICT =====
pred = model.transform(df_feat)

out = pred.select(
    *df.columns,
    "probability",
    "prediction"
)

pdf = out.toPandas()
pdf["fraud_probability"] = pdf["probability"].apply(lambda x: float(x[1]))

# ===== METRICS =====
total = len(pdf)
fraud_cnt = int(pdf["prediction"].sum())
non_fraud_cnt = total - fraud_cnt
fraud_rate = fraud_cnt / total * 100

# ===== SUMMARY =====
st.subheader("📊 Tổng quan kết quả")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Tổng giao dịch", total)
col2.metric("Fraud dự đoán", fraud_cnt)
col3.metric("Không fraud", non_fraud_cnt)
col4.metric("Tỉ lệ fraud (%)", f"{fraud_rate:.2f}%")

# ===== CHARTS =====
st.subheader("📈 Phân tích")

colA, colB = st.columns(2)

with colA:
    st.markdown("**Phân bố prediction**")
    fig1, ax1 = plt.subplots()
    pdf["prediction"].value_counts().plot(kind="bar", ax=ax1)
    ax1.set_xticklabels(["Not Fraud (0)", "Fraud (1)"], rotation=0)
    st.pyplot(fig1)

with colB:
    st.markdown("**Histogram xác suất fraud**")
    fig2, ax2 = plt.subplots()
    ax2.hist(pdf["fraud_probability"], bins=50)
    ax2.set_xlabel("Fraud probability")
    st.pyplot(fig2)

# ===== TABLE =====
st.subheader("📋 Kết quả chi tiết (50 dòng đầu)")
st.dataframe(
    pdf.head(50),
    use_container_width=True
)

st.info(
    "prediction: 0 = Không gian lận | 1 = Gian lận\n\n"
    "fraud_probability = xác suất gian lận do Logistic Regression dự đoán"
)
