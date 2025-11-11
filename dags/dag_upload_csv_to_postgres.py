from datetime import datetime

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook


def _upload_csv_to_postgres():
    """
    Lee un archivo CSV y lo carga en una tabla de PostgreSQL.
    """
    csv_filepath = "/opt/airflow/data/coffee_sales_full.csv"
    pg_hook = PostgresHook(postgres_conn_id="postgres_data_conn")

    df = pd.read_csv(csv_filepath)
    df.columns = [c.lower().replace(".", "").replace(" ", "_") for c in df.columns]

    engine = pg_hook.get_sqlalchemy_engine()
    df.to_sql("coffee_sales", engine, if_exists="replace", index=False, chunksize=1000)
    print(f"Se cargaron {len(df)} filas en la tabla 'coffee_sales'.")


with DAG(
    dag_id="upload_csv_to_postgres_dag",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@once",
    catchup=False,
    tags=["data-pipeline"],
) as dag:
    upload_task = PythonOperator(
        task_id="upload_csv_to_postgres",
        python_callable=_upload_csv_to_postgres,
    )
