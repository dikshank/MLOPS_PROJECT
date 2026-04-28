import mlflow
mlflow.set_tracking_uri('/opt/airflow/mlruns')
client = mlflow.tracking.MlflowClient()

# These are the run IDs with Windows paths that need fixing
runs_to_fix = {
    '4': 'f77d961c89d0406aaaa17aa625b14862',
    '5': '21af62fae7b14528aa1326c1f8fbe624',
    '6': 'c737411fd358449ea9fb862dab5d1b6a',
}

# Check which artifacts actually exist on disk
import os
for ver, run_id in runs_to_fix.items():
    artifact_path = f'/opt/airflow/mlruns/418733204694717083/{run_id}/artifacts/pytorch_model'
    exists = os.path.exists(artifact_path)
    print(f'Version {ver} | run={run_id} | artifacts exist: {exists} | path: {artifact_path}')
