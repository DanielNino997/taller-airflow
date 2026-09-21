"""API de inferencia para el modelo de clasificacion de penguins."""
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/modelo_penguins.pkl")

app = FastAPI(
    title="API Penguins",
    description="Predice la especie de un penguin a partir de sus medidas",
    version="1.0",
)


class Penguin(BaseModel):
    culmen_length_mm: float
    culmen_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float


def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=503,
            detail="El modelo aun no existe. Ejecute el DAG en Airflow.",
        )
    return joblib.load(MODEL_PATH)


@app.get("/")
def inicio():
    return {"mensaje": "API de penguins activa", "documentacion": "/docs"}


@app.get("/health")
def salud():
    return {"estado": "ok", "modelo_disponible": os.path.exists(MODEL_PATH)}


@app.post("/predict")
def predecir(penguin: Penguin):
    modelo = cargar_modelo()
    datos = pd.DataFrame([penguin.model_dump()])
    especie = modelo.predict(datos)[0]
    return {"especie_predicha": str(especie)}
