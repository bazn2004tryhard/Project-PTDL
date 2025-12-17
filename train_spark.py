import os
import pickle
import numpy as np

# === Ép đường dẫn JAVA & SPARK ===
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-17"
os.environ["SPARK_HOME"] = r"C:\spark-4.0.1-bin-hadoop3"
os.environ["PATH"] = rf"{os.environ['JAVA_HOME']}\bin;{os.environ['SPARK_HOME']}\bin;" + os.environ["PATH"]

print("JAVA_HOME =", os.environ["JAVA_HOME"])
print("SPARK_HOME =", os.environ["SPARK_HOME"])

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def main():
    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("credit-train")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # 1) Đọc dữ liệu
    df = spark.read.csv("creditcard.csv", inferSchema=True, header=True)
    if "Class" not in df.columns:
        raise ValueError("Không thấy cột 'Class' trong creditcard.csv")
    df = df.withColumnRenamed("Class", "label")

    print("Total rows:", df.count())
    df.show(5)

    # 2) Chia 7/3
    train, test = df.randomSplit([0.7, 0.3], seed=7)
    print("Train rows:", train.count())
    print("Test rows :", test.count())

    # 3) VectorAssembler
    feature_cols = [c for c in df.columns if c != "label"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    train_ml = assembler.transform(train).select("features", "label")
    test_ml = assembler.transform(test).select("features", "label")

    # 4) Train Logistic Regression
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=1000)
    model = lr.fit(train_ml)

    # 5) Đánh giá ROC-AUC & PR-AUC (chuẩn fraud)
    pred = model.transform(test_ml)

    roc_auc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    ).evaluate(pred)

    pr_auc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    ).evaluate(pred)

    print("✅ ROC-AUC =", roc_auc)
    print("✅ PR-AUC  =", pr_auc)

    # 6) Lưu model kiểu sklearn (KHÔNG dùng Spark write().save() để tránh lỗi Windows)
    coef = np.array(model.coefficients)   # length = số feature
    intercept = float(model.intercept)

    artifact = {
        "feature_cols": feature_cols,
        "coef": coef,
        "intercept": intercept,
        "threshold": 0.5,
    }

    with open("model.pkl", "wb") as f:
        pickle.dump(artifact, f)

    print("✅ Đã lưu model vào model.pkl (không dùng Spark save để tránh lỗi NativeIO trên Windows)")

    spark.stop()


if __name__ == "__main__":
    main()
