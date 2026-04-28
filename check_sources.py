import mlflow
mlflow.set_tracking_uri('/opt/airflow/mlruns')
client = mlflow.tracking.MlflowClient()
versions = client.search_model_versions("name='melanoma_classifier'")
for v in versions:
    mv = client.get_model_version('melanoma_classifier', v.version)
    print(v.version, v.current_stage, mv.source)
