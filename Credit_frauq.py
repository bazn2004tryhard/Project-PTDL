import os

# === Ép đường dẫn JAVA & SPARK chuẩn ở đây ===
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-17"
os.environ["SPARK_HOME"] = r"C:\spark-4.0.1-bin-hadoop3"

# Thêm vào PATH cho chắc chắn
os.environ["PATH"] = rf"{os.environ['JAVA_HOME']}\bin;{os.environ['SPARK_HOME']}\bin;" + os.environ["PATH"]

# (Khuyến mãi) In thử ra để bạn tự check
print("JAVA_HOME =", os.environ["JAVA_HOME"])
print("SPARK_HOME =", os.environ["SPARK_HOME"])

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pandas as pd

spark = SparkSession.builder.master("local[*]").appName("credit").getOrCreate()

df = spark.read.csv("creditcard.csv", inferSchema=True, header=True)
df.printSchema()
df.show(5)

# -------- chia tập dũ liệu -------
train, test = df.randomSplit([0.7, 0.3], seed=7)
print("Train set length:", train.count())
print("Test set length:", test.count())
