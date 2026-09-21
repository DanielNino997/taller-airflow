"""
DAG del taller de Airflow.
Pipeline: borrar -> cargar datos crudos -> preprocesar -> entrenar modelo.
"""
import os
from datetime import datetime

import joblib
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.tree import DecisionTreeClassifier

from airflow import DAG
from airflow.operators.python import PythonOperator

CONN_STR = os.environ["MYSQL_CONN_STR"]
CSV_PATH = "/opt/airflow/data/penguins.csv"
MODEL_PATH = "/opt/airflow/models/modelo_penguins.pkl"


def borrar_datos():
    engine = create_engine(CONN_STR)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS penguins_clean"))
        conn.execute(text("DROP TABLE IF EXISTS penguins_raw"))
    print("Tablas eliminadas")


def cargar_datos():
    df = pd.read_csv(CSV_PATH)
    engine = create_engine(CONN_STR)
    df.to_sql("penguins_raw", engine, if_exists="replace", index=False)
    print(f"Filas cargadas sin preprocesamiento: {len(df)}")


def preprocesar():
    engine = create_engine(CONN_STR)
    df = pd.read_sql("SELECT * FROM penguins_raw", engine)
    df = df.dropna()
    df = df.rename(columns={
        "bill_length_mm": "culmen_length_mm",
        "bill_depth_mm": "culmen_depth_mm",
    })
    df = df[["species", "culmen_length_mm", "culmen_depth_mm",
             "flipper_length_mm", "body_mass_g"]]
    df.to_sql("penguins_clean", engine, if_exists="replace", index=False)
    print(f"Filas despues del preprocesamiento: {len(df)}")


def entrenar():
    engine = create_engine(CONN_STR)
    df = pd.read_sql("SELECT * FROM penguins_clean", engine)
    X = df[["culmen_length_mm", "culmen_depth_mm",
            "flipper_length_mm", "body_mass_g"]]
    y = df["species"]
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, MODEL_PATH)
    print(f"Modelo entrenado con {len(df)} filas")
    print(f"Precision sobre datos de entrenamiento: {clf.score(X, y):.4f}")
    print(f"Modelo guardado en {MODEL_PATH}")


with DAG(
    dag_id="penguins_pipeline",
    description="Carga, preprocesa y entrena modelo de penguins",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["taller", "mlops"],
) as dag:

    t1 = PythonOperator(task_id="borrar_datos", python_callable=borrar_datos)
    t2 = PythonOperator(task_id="cargar_datos", python_callable=cargar_datos)
    t3 = PythonOperator(task_id="preprocesar", python_callable=preprocesar)
    t4 = PythonOperator(task_id="entrenar", python_callable=entrenar)

    t1 >> t2 >> t3 >> t4
