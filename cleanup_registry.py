import mlflow
mlflow.set_tracking_uri('/opt/airflow/mlruns')
client = mlflow.tracking.MlflowClient()
versions = client.search_model_versions("name='melanoma_classifier'")
for v in versions:
    client.delete_model_version('melanoma_classifier', v.version)
    print(f'Deleted version {v.version}')
client.delete_registered_model('melanoma_classifier')
print('Deleted registered model melanoma_classifier')
