# Databricks notebook source
from databricks.sdk.runtime import dbutils

dbutils.widgets.text("catalog", "pyspark_hands_on")
dbutils.widgets.text("schema", "fraud_dev")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

import mlflow
import pandas as pd
from fraud_analytics.ml.model import predict_anomalies
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Load Model from Unity Catalog
mlflow.set_registry_uri("databricks-uc")
model_uri = f"models:/{catalog}.{schema}.fraud_isolation_forest/latest"
loaded_model = mlflow.sklearn.load_model(model_uri)

# Load Features to Score
features_pd = spark.table(f"{catalog}.{schema}.gold_customer_features").toPandas()
feature_cols = ["total_transactions", "avg_transaction_amount", "max_transaction_amount", "high_value_transaction_count"]

# Run Batch Scoring
scored_pd = predict_anomalies(loaded_model, features_pd, feature_cols)

# Write Predictions back to Delta
scored_spark_df = spark.createDataFrame(scored_pd)
scored_spark_df.write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{schema}.fraud_predictions")
print(f"Written batch fraud predictions to {catalog}.{schema}.fraud_predictions")