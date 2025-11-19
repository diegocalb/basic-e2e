from datetime import datetime

from airflow.decorators import dag, task
from docker.types import Mount


@dag(
    dag_id="ml_model_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@weekly",
    catchup=False,
    tags=["ml", "end-to-end"],
    doc_md="""
    ### Pipeline Completo de Modelo de ML

    Este DAG orquesta el ciclo de vida completo de un modelo de ML para pronóstico de ventas.
    1. **retrain_model**: Reentrena el modelo con los datos más recientes y lo registra en MLflow.
    2. **evaluate_model**: Evalúa el rendimiento del modelo recién entrenado en un conjunto de holdout.
    3. **promote_model**: Compara el modelo en 'Staging' con el de 'Production' y lo promueve si cumple los criterios.
    """,
)
def ml_model_pipeline_dag():
    COMMON_ENV = {
        "MLFLOW_TRACKING_URI": "{{ var.value.mlflow_tracking_uri }}",
        "POSTGRES_CONN_ID": "{{ var.value.postgres_conn_id }}",
        "POSTGRES_TABLE_NAME": "{{ var.value.postgres_table_name }}",
        "HOLDOUT_DAYS": "{{ var.value.holdout_days }}",
        "RMSE_IMPROVEMENT_THRESHOLD": "{{ var.value.rmse_improvement_threshold }}",
        "MODEL_OBSOLESCENCE_DAYS": "{{ var.value.model_obsolescence_days }}",
    }

    @task()
    def log_start():
        print("Iniciando pipeline de ML: Reentrenamiento, Evaluación y Promoción.")

    @task.docker(
        task_id="retrain_model",
        image="diegoecalb/basice2e_docker_prod:v1.0.2",
        docker_url="unix://var/run/docker.sock",
        command=["python", "/app/coffee_modeling/train.py"],
        environment=COMMON_ENV,
        mounts=[Mount(source="/home/dcalb/basic-e2e/src", target="/app", type="bind")],
        network_mode="basic-e2e_mlflow-network",
    )
    def retrain_model_docker():
        """Ejecuta el script de entrenamiento del modelo en un contenedor Docker."""
        pass

    @task.docker(
        task_id="evaluate_model",
        image="diegoecalb/basice2e_docker_prod:v1.0.2",
        docker_url="unix://var/run/docker.sock",
        command=["python", "/app/coffee_modeling/evaluate.py"],
        environment=COMMON_ENV,
        mounts=[Mount(source="/home/dcalb/basic-e2e/src", target="/app", type="bind")],
        network_mode="basic-e2e_mlflow-network",
    )
    def evaluate_model_docker():
        """Ejecuta el script de evaluación del modelo en un contenedor Docker."""
        pass

    @task.docker(
        task_id="promote_model",
        image="diegoecalb/basice2e_docker_prod:v1.0.2",
        docker_url="unix://var/run/docker.sock",
        command=["python", "/app/coffee_modeling/promote.py"],
        environment=COMMON_ENV,
        mounts=[Mount(source="/home/dcalb/basic-e2e/src", target="/app", type="bind")],
        network_mode="basic-e2e_mlflow-network",
    )
    def promote_model_docker():
        """Ejecuta el script de promoción del modelo en un contenedor Docker."""
        pass

    @task()
    def log_end():
        print("Pipeline de ML finalizado.")

    # Definir el flujo de tareas
    start_task = log_start()
    retrain_task = retrain_model_docker()
    evaluate_task = evaluate_model_docker()
    promote_task = promote_model_docker()
    end_task = log_end()

    start_task >> retrain_task >> evaluate_task >> promote_task >> end_task


ml_model_pipeline_dag()
