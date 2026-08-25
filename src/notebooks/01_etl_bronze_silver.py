# Databricks notebook source
from databricks.sdk.runtime import dbutils

dbutils.widgets.text("catalog", "pyspark_hands_on")
dbutils.widgets.text("schema", "fraud_dev")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

from fraud_analytics.data.transform import clean_raw_transactions
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("ETL_Bronze_Silver").getOrCreate()

# Create target schema
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# Mock Bronze Raw Data Ingestion
raw_data = [
    ("tx101", "cust_1", 150.0, 1.0, "2026-08-20T10:00:00Z"),
    ("tx102", "cust_1", 6200.0, 1.0, "2026-08-20T11:00:00Z"),
    ("tx103", "cust_2", 45.0, 1.0, "2026-08-20T12:00:00Z"),
    ("tx104", "cust_3", -10.0, 1.0, "2026-08-20T13:00:00Z"), # Invalid row
    ("tx105", "cust_2", 9000.0, 1.0, "2026-08-20T14:00:00Z"),
]

columns = ["transaction_id", "customer_id", "amount", "exchange_rate", "timestamp"]
raw_df = spark.createDataFrame(raw_data, columns)

# Write to Bronze
raw_df.write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{schema}.bronze_transactions")

# Transform Bronze -> Silver using packaged functions
bronze_df = spark.table(f"{catalog}.{schema}.bronze_transactions")
silver_df = clean_raw_transactions(bronze_df)

# Write to Silver
silver_df.write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{schema}.silver_transactions")
print(f"Successfully processed Silver table at {catalog}.{schema}.silver_transactions")