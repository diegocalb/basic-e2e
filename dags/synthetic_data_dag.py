import os
import random

import pandas as pd
import pendulum
from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from faker import Faker


@dag(
    dag_id="synthetic_data_loader",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    schedule_interval="@daily",
    catchup=False,
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

    @task
    def generate_synthetic_data(**kwargs):
        """
        Genera datos sintéticos para simular transacciones diarias.
        La fecha de las transacciones se basa en la fecha de ejecución lógica del DAG.
        """
        logical_date = kwargs["ds"]
        month = pendulum.parse(logical_date).month
        month_name = pendulum.parse(logical_date).format("MMM").lower()
        weekday = pendulum.parse(logical_date).day_of_week
        weekday_name = pendulum.parse(logical_date).format("ddd").lower()

        fake = Faker()
        num_records = random.randint(50, 150)

        # Columnas basadas en el archivo coffee_sales_full.csv
        # Se omiten las columnas de fecha/hora derivadas ('Month', 'Weekday', 'Hour', etc.) para evitar conflictos de esquema.
        columns = [
            "transaction_id",
            "transaction_date",
            "transaction_time",
            "transaction_qty",
            "store_id",
            "store_location",
            "product_id",
            "unit_price",
            "product_category",
            "product_type",
            "product_detail",
            "month",
            "month1",
            "weekday",
            "weekday1",
            "hour",
            "payment_method",
            "place_of_sale",
            "promotion",
        ]

        data = []

        # Valores de ejemplo para mantener la consistencia
        store_locations = {3: "Caballito", 5: "Palermo", 8: "Belgrano"}
        product_details = {
            22: (2.0, "Coffee", "Drip coffee", "Our Old Time Diner Blend Sm"),
            32: (3.0, "Coffee", "Gourmet brewed coffee", "Ethiopia Rg"),
            45: (3.0, "Tea", "Brewed herbal tea", "Peppermint Lg"),
            57: (3.1, "Tea", "Brewed Chai tea", "Spicy Eye Opener Chai Lg"),
            59: (4.5, "Drinking Chocolate", "Hot chocolate", "Dark chocolate Lg"),
            71: (3.75, "Bakery", "Pastry", "Chocolate Croissant"),
        }
        payment_methods = ["Tarjeta", "Efectivo"]
        places_of_sale = ["Negocio", "Take Away"]
        promotions = ["clubLN", "365", ""]

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

        for i in range(num_records):
            hour = random.randint(7, 20)
            store_id = random.choice(list(store_locations.keys()))
            product_id = random.choice(list(product_details.keys()))
            price, category, p_type, p_detail = product_details[product_id]

            data.append(
                {
                    "transaction_id": start_id + i,
                    "transaction_date": pendulum.parse(
                        logical_date
                    ).to_date_string(),  # Formato YYYY-MM-DD
                    "transaction_time": fake.time(),
                    "transaction_qty": random.randint(1, 3),
                    "store_id": store_id,
                    "store_location": store_locations[store_id],
                    "product_id": product_id,
                    "unit_price": price,
                    "product_category": category,
                    "product_type": p_type,
                    "product_detail": p_detail,
                    "month": month,
                    "month1": month_name,
                    "weekday": weekday,
                    "weekday1": weekday_name,
                    "hour": hour,
                    "payment_method": random.choice(payment_methods),
                    "place_of_sale": random.choice(places_of_sale),
                    "promotion": random.choice(promotions),
                }
            )

        df = pd.DataFrame(data, columns=columns)

        # Guardar en un archivo temporal
        temp_file_path = f"/tmp/synthetic_data_{logical_date}.csv"
        df.to_csv(temp_file_path, index=False)
        return temp_file_path

    @task
    def load_data_to_postgres(file_path: str):
        """
        Carga los datos desde un archivo CSV a la tabla coffee_sales en PostgreSQL.
        """
        pg_hook = PostgresHook(postgres_conn_id="postgres_data_conn")
        engine = pg_hook.get_sqlalchemy_engine()

        df = pd.read_csv(file_path)
        df.to_sql("coffee_sales", engine, if_exists="append", index=False)
        os.remove(file_path)  # Limpiar el archivo temporal

    synthetic_data_path = generate_synthetic_data()
    load_data_to_postgres(synthetic_data_path)


synthetic_data_loader_dag()
