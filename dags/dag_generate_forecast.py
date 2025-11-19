from datetime import datetime

from airflow.decorators import dag, task
from docker.types import Mount


@dag(
    dag_id="generate_forecast_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@weekly",
    catchup=False,
    tags=["ml", "forecast"],
    doc_md="""
    ### Pipeline de Generación de Pronósticos

    Este DAG utiliza el modelo de ML en producción para generar un pronóstico de ventas para los próximos días.
    1. **generate_forecast**: Ejecuta el script `forecast.py`.
    2. El script carga el modelo etiquetado como 'Production' desde MLflow.
    3. Genera un pronóstico para los próximos 7 días.
    4. Guarda el resultado en una tabla de la base de datos `postgres_data`.
    """,
)
def generate_forecast_dag():
    # Configuración centralizada para la tarea Docker
    COMMON_ENV = {
        "MLFLOW_TRACKING_URI": "{{ var.value.mlflow_tracking_uri }}",
        "POSTGRES_CONN_ID": "{{ var.value.postgres_conn_id }}",
        "POSTGRES_TABLE_NAME": "{{ var.value.postgres_table_name }}",
        "FORECAST_DAYS": "7",
        "FORECAST_OUTPUT_TABLE": "sales_forecast",
    }

    @task()
    def log_start():
        print("Iniciando la generación de pronósticos de ventas.")

    @task.docker(
        task_id="generate_forecast",
        image="diegoecalb/basice2e_docker_prod:v1.0.2",
        docker_url="unix://var/run/docker.sock",
        command=["python", "/app/coffee_modeling/forecast.py"],
        environment=COMMON_ENV,
        mounts=[Mount(source="/home/dcalb/basic-e2e/src", target="/app", type="bind")],
        network_mode="basic-e2e_mlflow-network",
    )
    def generate_forecast_docker():
        """Ejecuta el script de pronóstico en un contenedor Docker."""
        pass

    @task()
    def log_end():
        print("Pronóstico generado y guardado en la base de datos.")

    start_task = log_start()
    forecast_task = generate_forecast_docker()
    end_task = log_end()

    start_task >> forecast_task >> end_task


generate_forecast_dag()
