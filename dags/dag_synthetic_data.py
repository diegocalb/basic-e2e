from io import StringIO

import pendulum
from airflow.decorators import dag, task
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook


@dag(
    dag_id="synthetic_data_loader",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    schedule_interval="@daily",
    catchup=False,
    tags=["data-pipeline"],
    doc_md="""
    ### Synthetic Data Loader DAG

    Este DAG genera datos sintéticos de ventas de café y los carga en una tabla de PostgreSQL.
    - **generate_synthetic_data**: Crea un DataFrame de pandas con datos falsos basados en el esquema de `coffee_sales_full.csv`. La fecha de la transacción se basa en la fecha de ejecución del DAG.
    - **load_data_to_postgres**: Carga los datos generados en la tabla `coffee_sales` de la base de datos `postgres_data`.
    """,
)
def synthetic_data_loader_dag():
    """
    DAG para generar y cargar datos sintéticos de ventas de café.
    """
    BUCKET_NAME = "airflow-data"

    @task
    def ensure_bucket_exists():
        """Crea el bucket de S3/MinIO si no existe."""
        hook = S3Hook(aws_conn_id="minio_conn")
        if not hook.check_for_bucket(BUCKET_NAME):
            hook.create_bucket(bucket_name=BUCKET_NAME)

    @task
    def generate_synthetic_data(**kwargs):
        """
        Genera datos sintéticos para simular transacciones diarias.
        Guarda los datos en un archivo CSV en MinIO.

        Returns:
            str: La clave S3 (ruta del archivo) del objeto guardado en MinIO.
        """
        import random

        import pandas as pd

        from dags.synthetic_data_utils import COLUMNS, generate_records

        logical_date = kwargs["ds"]
        num_records = random.randint(50, 150)

        # Obtener el último transaction_id de la base de datos
        pg_hook = PostgresHook(postgres_conn_id="postgres_data_conn")
        try:
            # Asegurarse de que la tabla exista antes de consultarla
            conn = pg_hook.get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT to_regclass('public.coffee_sales');")
            table_exists = cursor.fetchone()[0]
            if table_exists:
                last_id_df = pg_hook.get_pandas_df(
                    "SELECT MAX(transaction_id) FROM coffee_sales"
                )
                last_id = (
                    last_id_df.iloc[0, 0]
                    if not last_id_df.empty and pd.notna(last_id_df.iloc[0, 0])
                    else 0
                )
            else:
                last_id = 0
            cursor.close()
            conn.close()
        except Exception:
            last_id = 0

        start_id = last_id + 1

        # Generar los registros usando la función auxiliar
        data = generate_records(logical_date, num_records, start_id)
        df = pd.DataFrame(data, columns=COLUMNS)

        # Guardar en MinIO
        s3_hook = S3Hook(aws_conn_id="minio_conn")
        s3_key = f"synthetic_data/coffee_sales_{logical_date}.csv"

        # Convertir DataFrame a string CSV para cargarlo en memoria
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)

        s3_hook.load_string(
            string_data=csv_buffer.getvalue(),
            key=s3_key,
            bucket_name=BUCKET_NAME,
            replace=True,
        )
        return s3_key

    @task
    def load_data_to_postgres(s3_key: str):
        """
        Carga los datos desde un archivo CSV en MinIO a la tabla coffee_sales en PostgreSQL.
        """
        import pandas as pd

        s3_hook = S3Hook(aws_conn_id="minio_conn")
        pg_hook = PostgresHook(postgres_conn_id="postgres_data_conn")
        engine = pg_hook.get_sqlalchemy_engine()

        # Leer el archivo directamente desde MinIO
        csv_string = s3_hook.read_key(key=s3_key, bucket_name=BUCKET_NAME)
        df = pd.read_csv(StringIO(csv_string))

        df.to_sql("coffee_sales", engine, if_exists="append", index=False)
        s3_hook.delete_objects(
            bucket=BUCKET_NAME, keys=s3_key
        )  # Limpiar el archivo de MinIO

    ensure_bucket_exists()
    synthetic_data_path = generate_synthetic_data()
    load_data_to_postgres(synthetic_data_path)


synthetic_data_loader_dag()
