import mlflow
mlflow.set_tracking_uri('/opt/airflow/mlruns')
client = mlflow.tracking.MlflowClient()

for ver in ['1','2','3','7']:
    mv = client.get_model_version('melanoma_classifier', ver)
    try:
        run = client.get_run(mv.run_id)
        recall = run.data.metrics.get('val_recall', 'N/A')
        f1 = run.data.metrics.get('val_f1', 'N/A')
        print(f'Version {ver} | stage={mv.current_stage} | val_recall={recall} | val_f1={f1} | source={mv.source}')
    except Exception as e:
        print(f'Version {ver} | error={e}')
