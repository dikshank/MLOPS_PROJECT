import mlflow
mlflow.set_tracking_uri('./mlruns')
client = mlflow.tracking.MlflowClient()
versions = client.get_latest_versions('melanoma_classifier', stages=['Production'])
if versions:
    for v in versions:
        run = client.get_run(v.run_id)
        print(f'Version: {v.version}')
        print(f'Run ID: {v.run_id}')
        print(f'Model name: {run.data.tags.get("model_name", "unknown")}')
else:
    print('No model in Production yet')