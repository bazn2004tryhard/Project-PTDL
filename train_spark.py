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
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def main():
    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("credit-train-zscore")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # 1) Read data
    df = spark.read.csv("creditcard.csv", inferSchema=True, header=True)
    if "Class" not in df.columns:
        raise ValueError("Không thấy cột 'Class' trong creditcard.csv")
    df = df.withColumnRenamed("Class", "label")

    print("Total rows:", df.count())
    df.show(5)

    # 2) split
    train, test = df.randomSplit([0.7, 0.3], seed=7)
    print("Train rows:", train.count())
    print("Test rows :", test.count())

    # 3) Assembler
    feature_cols = [c for c in df.columns if c != "label"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    train_vec = assembler.transform(train).select("features", "label")
    test_vec = assembler.transform(test).select("features", "label")

    # 4) Z-score scaler (fit on train only)
    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaledFeatures",
        withMean=True,
        withStd=True
    )
    scaler_model = scaler.fit(train_vec)
    train_ml = scaler_model.transform(train_vec).select("scaledFeatures", "label")
    test_ml = scaler_model.transform(test_vec).select("scaledFeatures", "label")

    # 5) Logistic Regression on scaledFeatures
    lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="label", maxIter=200)
    model = lr.fit(train_ml)

    # 6) Evaluate (ROC-AUC + PR-AUC)
    pred = model.transform(test_ml)

    roc_auc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    ).evaluate(pred)

    pr_auc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    ).evaluate(pred)

    print("✅ ROC-AUC =", roc_auc)
    print("✅ PR-AUC  =", pr_auc)

    # 7) Export to model.pkl for numpy deploy (coef/intercept + z_mean/z_std)
    coef = np.array(model.coefficients)  # coef on scaledFeatures
    intercept = float(model.intercept)

    z_mean = np.array(scaler_model.mean.toArray(), dtype=float)
    z_std = np.array(scaler_model.std.toArray(), dtype=float)
    z_std = np.where(z_std == 0, 1.0, z_std)

    artifact = {
        "feature_cols": feature_cols,
        "coef": coef,
        "intercept": intercept,
        "z_mean": z_mean,
        "z_std": z_std,
        "trained_on_zscore": True,
        "default_threshold": 0.5,
        "metrics": {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc)},
    }

    with open("model.pkl", "wb") as f:
        pickle.dump(artifact, f)

    print("✅ Saved model.pkl (deploy numpy/pandas, có z-score).")
    spark.stop()


if __name__ == "__main__":
    main()
