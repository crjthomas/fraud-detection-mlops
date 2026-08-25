import pandas as pd
from sklearn.ensemble import IsolationForest
import mlflow
import mlflow.sklearn

def train_isolation_forest(df_pandas: pd.DataFrame, n_estimators: int = 100, contamination: float = 0.05):
    """Train an Isolation Forest for fraud detection and logs to MLflow."""
    feature_cols = ["total_transactions", "avg_transaction_amount", "max_transaction_amount", "high_value_transaction_count"]
    X = df_pandas[feature_cols]
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
    model.fit(X)
    return model, feature_cols

def predict_anomalies(model, df_pandas: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Generates anomaly predictions (-1 is anomaly, 1 is normal)."""
    preds = model.predict(df_pandas[feature_cols])
    df_pandas["anomaly_score"] = preds
    df_pandas["is_fraud_predicted"] = (df_pandas["anomaly_score"] == -1).astype(int)
    return df_pandas[["customer_id", "is_fraud_predicted"]]