from datetime import datetime

from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook


@dag(
    dag_id="upload_csv_to_postgres",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@once",
    catchup=False,
    tags=["data-pipeline"],
    doc_md="""
    ### Upload csv to postgres DAG

    Este DAG toma el csv con las ventas de café y las carga en una tabla de PostgreSQL por única vez.
    """,
)
def upload_csv_to_postgres_dag():
    """
    DAG para cargar un archivo CSV en una tabla de PostgreSQL.
    """

    @task
    def upload_csv_to_postgres():
        """
        Lee un archivo CSV y lo carga en una tabla de PostgreSQL.
        """
        import pandas as pd

        csv_filepath = "/opt/airflow/data/coffee_sales_full.csv"
        pg_hook = PostgresHook(postgres_conn_id="postgres_data_conn")

        df = pd.read_csv(csv_filepath)
        df.columns = [c.lower().replace(".", "").replace(" ", "_") for c in df.columns]
        df["transaction_date"] = pd.to_datetime(
            df["transaction_date"], format="%m/%d/%Y"
        ).dt.date

        engine = pg_hook.get_sqlalchemy_engine()
        df.to_sql(
            "coffee_sales", engine, if_exists="replace", index=False, chunksize=1000
        )
        print(f"Se cargaron {len(df)} filas en la tabla 'coffee_sales'.")

    upload_csv_to_postgres()


upload_csv_to_postgres_dag()
