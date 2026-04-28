import mlflow

mlflow.set_tracking_uri('/opt/airflow/mlruns')
client = mlflow.tracking.MlflowClient()

versions = client.search_model_versions("name='melanoma_classifier'")

if not versions:
    print("No registered models found.")
else:
    for v in versions:
        print(f"Version: {v.version} | Stage: {v.current_stage} | Run ID: {v.run_id} | Status: {v.status}")