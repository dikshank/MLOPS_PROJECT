import mlflow
mlflow.set_tracking_uri('/opt/airflow/mlruns')
client = mlflow.tracking.MlflowClient()
versions = client.search_model_versions("name='melanoma_classifier'")
for v in versions:
    print(v.version, v.current_stage, v.status)
