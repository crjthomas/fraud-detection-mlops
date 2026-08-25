# Databricks notebook source
from databricks.sdk.runtime import dbutils

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "fraud_dev")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

import mlflow
from fraud_analytics.ml.model import train_isolation_forest
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Load Gold Features
gold_pd = spark.table(f"{catalog}.{schema}.gold_customer_features").toPandas()

# Set Unity Catalog Model Registry
mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.{schema}.fraud_isolation_forest"

with mlflow.start_run() as run:
    model, feature_cols = train_isolation_forest(gold_pd, n_estimators=100, contamination=0.1)
    
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("contamination", 0.1)
    
    # Log model & register to Unity Catalog
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=model_name
    )
    print(f"Logged & Registered ML Model to Unity Catalog: {model_name}")