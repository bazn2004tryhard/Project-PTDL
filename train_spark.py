import os
import shutil

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

    # Đổi "Class" -> "label" cho Spark ML
    if "Class" not in df.columns:
        raise ValueError("Không thấy cột 'Class' trong creditcard.csv")
    df = df.withColumnRenamed("Class", "label")

    print("Total rows:", df.count())
    df.printSchema()
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

    # 5) Đánh giá nhanh AUC
    pred = model.transform(test_ml)
    evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    auc = evaluator.evaluate(pred)
    print("✅ AUC =", auc)

    pred.select("label", "probability", "prediction").show(5, truncate=False)

    # 6) Lưu artifacts (Spark lưu theo thư mục)
    MODEL_DIR = "spark_lr_model"
    ASSEMBLER_DIR = "spark_assembler"

    for p in [MODEL_DIR, ASSEMBLER_DIR]:
        if os.path.exists(p):
            shutil.rmtree(p)

    model.write().overwrite().save(MODEL_DIR)
    assembler.write().overwrite().save(ASSEMBLER_DIR)

    print(f"✅ Saved model to: {MODEL_DIR}")
    print(f"✅ Saved assembler to: {ASSEMBLER_DIR}")

    spark.stop()


if __name__ == "__main__":
    main()
