# Databricks notebook source
from databricks.sdk.runtime import dbutils

dbutils.widgets.text("catalog", "pyspark_hands_on")
dbutils.widgets.text("schema", "fraud_dev")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

from fraud_analytics.data.transform import build_gold_customer_features
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

silver_df = spark.table(f"{catalog}.{schema}.silver_transactions")
gold_df = build_gold_customer_features(silver_df)

gold_df.write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{schema}.gold_customer_features")
print(f"Successfully updated Gold feature table at {catalog}.{schema}.gold_customer_features")