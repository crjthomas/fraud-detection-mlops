from pyarrow import timestamp
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def clean_raw_transactions(transactions: DataFrame) -> DataFrame:
    """ Clean the raw transactions data."""
    return (transactions.filter (F.col("amount").isNotNull() & (F.col("amount") > 0))
            .withColumn("amount_usd", F.round(F.col("amount") * F.col("exchange_rate"), 2))
            .withColumn("transaction_timestamp", F.to_timestamp(F.col("timestamp")))
            .withColumn("is_high_value", F.when(F.col("amount_usd") > 5000, 1).otherwise(0))
            )
    
def build_gold_customer_features(silver_df: DataFrame) -> DataFrame:
    """Aggregates Silver transactions into Gold ML feature table."""
    return silver_df.groupBy("customer_id").agg(
        F.count("transaction_id").alias("total_transactions"),
        F.avg("amount_usd").alias("avg_transaction_amount"),
        F.max("amount_usd").alias("max_transaction_amount"),
        F.sum("is_high_value").alias("high_value_transaction_count")
    )
                                                