import random
from typing import Dict, List

import pendulum
from faker import Faker

# Columnas basadas en el archivo coffee_sales_full.csv
# El orden debe coincidir con el DataFrame que se creará.
COLUMNS = [
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


def generate_records(
    logical_date_str: str, num_records: int, start_id: int
) -> List[Dict]:
    """
    Genera una lista de registros sintéticos de ventas de café.

    Args:
        logical_date_str (str): La fecha de ejecución lógica en formato 'YYYY-MM-DD'.
        num_records (int): El número de registros a generar.
        start_id (int): El ID de transacción inicial.

    Returns:
        List[Dict]: Una lista de diccionarios, donde cada uno es un registro de venta.
    """
    fake = Faker()
    logical_date = pendulum.parse(logical_date_str)

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

    data = []
    for i in range(num_records):
        store_id = random.choice(list(store_locations.keys()))
        product_id = random.choice(list(product_details.keys()))
        price, category, p_type, p_detail = product_details[product_id]

        data.append(
            {
                "transaction_id": start_id + i,
                "transaction_date": logical_date.to_date_string(),
                "transaction_time": fake.time(),
                "transaction_qty": random.randint(1, 3),
                "store_id": store_id,
                "store_location": store_locations[store_id],
                "product_id": product_id,
                "unit_price": price,
                "product_category": category,
                "product_type": p_type,
                "product_detail": p_detail,
                "month": logical_date.month,
                "month1": logical_date.format("MMM").lower(),
                "weekday": logical_date.day_of_week,
                "weekday1": logical_date.format("ddd").lower(),
                "hour": random.randint(7, 20),
                "payment_method": random.choice(payment_methods),
                "place_of_sale": random.choice(places_of_sale),
                "promotion": random.choice(promotions),
            }
        )
    return data
