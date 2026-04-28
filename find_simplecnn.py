import mlflow
mlflow.set_tracking_uri('./mlruns')
client = mlflow.tracking.MlflowClient()
versions = client.search_model_versions('name="melanoma_classifier"')
for v in versions:
    try:
        run = client.get_run(v.run_id)
        model_name = run.data.tags.get('model_name', 'unknown')
    except:
        model_name = 'unknown'
    print(f'Version: {v.version} | Stage: {v.current_stage} | Model: {model_name}')
