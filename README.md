# Fraud Detection MLOps

End-to-end fraud analytics pipeline on **Databricks**, packaged as a **Databricks Asset Bundle**. It moves transaction data through a medallion lakehouse (Bronze → Silver → Gold), trains an Isolation Forest model with MLflow, and runs batch inference — all deployable as versioned jobs.

## What it does

1. **ETL (Bronze → Silver → Gold)**  
   Ingests mock transaction data, cleans and enriches it, then aggregates customer-level features for ML.

2. **MLOps**  
   Trains an unsupervised Isolation Forest on Gold features, registers the model with MLflow, and writes batch fraud predictions back to Unity Catalog.

## Architecture

```text
Raw transactions
       │
       ▼
┌──────────────┐     ┌──────────────┐     ┌─────────────────────┐
│    Bronze    │ ──► │    Silver    │ ──► │ Gold (customer feats)│
└──────────────┘     └──────────────┘     └──────────┬──────────┘
                                                     │
                                                     ▼
                                          Isolation Forest (MLflow)
                                                     │
                                                     ▼
                                          fraud_predictions table
```

| Layer | Table | Purpose |
| --- | --- | --- |
| Bronze | `bronze_transactions` | Raw ingest |
| Silver | `silver_transactions` | Cleaned / enriched transactions |
| Gold | `gold_customer_features` | Customer aggregates for ML |
| Output | `fraud_predictions` | Batch anomaly scores |

## Repository layout

```text
.
├── databricks.yml              # Bundle config (targets, artifacts, variables)
├── pyproject.toml              # Python package: fraud_analytics
├── resources/
│   ├── data_pipeline_job.yml   # ETL workflow job
│   └── ml_pipeline_job.yml     # Train + batch inference job
└── src/
    ├── fraud_analytics/
    │   ├── data/transform.py   # Spark transforms (Silver / Gold)
    │   └── ml/model.py         # Isolation Forest + MLflow helpers
    └── notebooks/
        ├── 01_etl_bronze_silver.py
        ├── 02_etl_gold.py
        ├── 03_train_model.py
        └── 04_batch_inference.py
```

## Prerequisites

- [Databricks CLI](https://docs.databricks.com/dev-tools/cli/install.html) authenticated to your workspace (`databricks auth login`)
- Python 3.11+ (local packaging / IDE)
- A Unity Catalog catalog your identity can write to

## Configuration

Bundle variables (defaults in `databricks.yml`):

| Variable | Default | Description |
| --- | --- | --- |
| `catalog` | `pyspark_hands_on` | Unity Catalog name |
| `schema` | `fraud_dev` (dev) / `fraud_proud` (prod) | Schema for tables & model |

Override at deploy time if needed:

```bash
databricks bundle deploy -t dev --var="catalog=your_catalog" --var="schema=your_schema"
```

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Select `.venv/bin/python` as the IDE interpreter so `pyspark` / package imports resolve.

## Deploy & run

```bash
# Validate bundle
databricks bundle validate -t dev

# Build wheel, sync notebooks/jobs, deploy to workspace
databricks bundle deploy -t dev

# Run workflows
databricks bundle run etl_fraud_pipeline -t dev
databricks bundle run ml_fraud_pipeline -t dev
```

Jobs deployed:

- **`[${bundle.target}] Fraud Detection - ETL Workflow`** — Bronze→Silver→Gold  
- **`[${bundle.target}] Fraud Detection - MLOps Workflow`** — train → batch inference  

The Python wheel (`fraud_analytics`) is built and attached to job tasks so notebooks can import packaged transforms and model helpers.

## Tech stack

- Databricks Asset Bundles & Jobs  
- Unity Catalog (Delta tables)  
- PySpark  
- scikit-learn (Isolation Forest)  
- MLflow model registry  

## License

Add a license if you plan to open-source this repository.
