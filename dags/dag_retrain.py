from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

with DAG(
    dag_id="coffee_model_retraining",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    retrain_model = DockerOperator(
        task_id="retrain_coffee_model",
        image="diegoecalb/basice2e_docker_prod:v1.0.2",  # Reemplaza con tu imagen en Docker Hub
        command=["python", "/app/coffee_modeling/train.py"],
        mounts=[Mount(source="/home/dcalb/basic-e2e/src", target="/app", type="bind")],
        docker_url="unix://var/run/docker.sock",
        # Conectar el contenedor a la misma red que los otros servicios
        # para que pueda comunicarse con MLflow.
        network_mode="basic-e2e_mlflow-network",
    )
