# Taller Airflow — Pipeline de clasificación de Penguins

Maestría en Inteligencia Artificial — Pontificia Universidad Javeriana
Asignatura: Operación de Aprendizaje de Máquinas (MLOps)
Autor: Daniel Niño

Orquestación con Apache Airflow de un pipeline completo de machine learning: carga de datos crudos a una base de datos, preprocesamiento, entrenamiento de modelo y exposición del modelo mediante una API de inferencia. Todos los servicios se despliegan con un único `docker-compose.yaml`.

## Arquitectura

```
                    docker-compose.yaml
  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐  ┌──────────┐
  │   MySQL      │  │  PostgreSQL  │  │        Airflow         │  │   API    │
  │  (datos)     │  │ (metadatos)  │  │ init / scheduler / web │  │ FastAPI  │
  │  penguins_db │  │   airflow    │  │   LocalExecutor        │  │  :8000   │
  │    :3307     │  │              │  │        :8080           │  │          │
  └──────▲───────┘  └──────▲───────┘  └───────┬────────────────┘  └────▲─────┘
         │                 │                  │                        │
         └─── lectura/escritura del DAG ──────┘                        │
                                              │                        │
                                       ./models (volumen compartido) ──┘
```

Se usan **dos motores de base de datos separados**, tal como exige el enunciado:

| Base | Motor | Contenido |
|---|---|---|
| `penguins_db` | MySQL 8.0 | Datos de penguins (tablas `penguins_raw` y `penguins_clean`) |
| `airflow` | PostgreSQL 13 | Metadatos de Airflow (DAGs, ejecuciones, usuarios) |

El modelo entrenado se comparte entre Airflow y la API a través del volumen `./models`: Airflow lo escribe, la API lo lee.

## El DAG

`dags/penguins_pipeline.py` define cuatro tareas secuenciales:

```
borrar_datos → cargar_datos → preprocesar → entrenar
```

| Tarea | Descripción |
|---|---|
| `borrar_datos` | Elimina las tablas `penguins_raw` y `penguins_clean` si existen |
| `cargar_datos` | Carga `data/penguins.csv` a `penguins_raw` **sin preprocesamiento** (344 filas) |
| `preprocesar` | Lee `penguins_raw` desde MySQL, elimina nulos, renombra `bill_*` a `culmen_*`, selecciona las variables numéricas y escribe `penguins_clean` (333 filas) |
| `entrenar` | Lee `penguins_clean` desde MySQL, entrena un `DecisionTreeClassifier(max_depth=3, random_state=42)` y serializa el modelo en `models/modelo_penguins.pkl` |

Las tareas se comunican **a través de la base de datos**, no mediante XComs, cumpliendo el requisito de entrenar "usando datos preprocesados de la base de datos".

El DAG se ejecuta bajo demanda (`schedule=None`).

## La API

FastAPI, construida con un `Dockerfile` propio en `api/`.

| Endpoint | Método | Descripción |
|---|---|---|
| `/` | GET | Mensaje de estado |
| `/health` | GET | Estado del servicio e indicador de disponibilidad del modelo |
| `/predict` | POST | Predice la especie a partir de las cuatro medidas |
| `/docs` | GET | Documentación interactiva (Swagger UI) |

Ejemplo de uso:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"culmen_length_mm": 39.1, "culmen_depth_mm": 18.7, "flipper_length_mm": 181, "body_mass_g": 3750}'
```

Respuesta:

```json
{"especie_predicha": "Adelie"}
```

El modelo se carga en cada petición, de modo que un reentrenamiento del DAG se refleja de inmediato sin reiniciar el servicio. Si el modelo aún no existe, la API responde `503` con un mensaje explicativo en lugar de fallar.

## Puesta en marcha

Requisitos: Docker y Docker Compose.

```bash
git clone git@github.com:DanielNino997/taller-airflow.git
cd taller-airflow
```

Crear el archivo de variables de entorno a partir de la plantilla:

```bash
cp .env.example .env
```

Completar los valores en `.env` y ajustar el UID del usuario:

```bash
sed -i "s/^AIRFLOW_UID=.*/AIRFLOW_UID=$(id -u)/" .env
```

Inicializar la base de metadatos y crear el usuario administrador:

```bash
docker compose up airflow-init
```

Levantar todos los servicios:

```bash
docker compose up -d --build
```

| Servicio | URL | Credenciales |
|---|---|---|
| Airflow | http://localhost:8080 | airflow / airflow |
| API | http://localhost:8000/docs | — |
| MySQL | localhost:3307 | según `.env` |

Para generar el modelo: activar el DAG `penguins_pipeline` en la interfaz de Airflow y ejecutarlo.

## Verificación

Estado de los servicios:

```bash
docker compose ps
```

Tablas y conteos en MySQL:

```bash
docker exec -it taller_mysql mysql -u penguins_user -p penguins_db
```

```sql
SELECT (SELECT COUNT(*) FROM penguins_raw) AS crudos,
       (SELECT COUNT(*) FROM penguins_clean) AS limpios;
```

La diferencia entre ambos conteos (344 frente a 333) evidencia que la carga se realizó sin preprocesamiento.

Disponibilidad del modelo:

```bash
curl http://localhost:8000/health
```

## Estructura del repositorio

```
taller-airflow/
├── docker-compose.yaml     Definición de los cinco servicios
├── .env.example            Plantilla de variables de entorno
├── .gitignore
├── dags/
│   └── penguins_pipeline.py
├── data/
│   └── penguins.csv        Dataset Palmer Penguins
├── models/                 Volumen compartido (el .pkl no se versiona)
├── api/
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── logs/                   Registros de Airflow (no se versionan)
└── plugins/
```

## Decisiones técnicas

**LocalExecutor en lugar de CeleryExecutor.** El `docker-compose.yaml` oficial de Airflow emplea CeleryExecutor, que requiere Redis y workers adicionales. Para la carga de trabajo de este taller, LocalExecutor ofrece la misma funcionalidad con menos servicios y menor consumo de recursos.

**Healthchecks y arranque ordenado.** Ambas bases de datos definen un healthcheck, y los servicios de Airflow declaran `depends_on` con `condition: service_healthy`. Esto evita los fallos en cadena que se producen cuando Airflow intenta conectarse a una base que está encendida pero todavía no acepta conexiones.

**Dependencias de Airflow vía `_PIP_ADDITIONAL_REQUIREMENTS`.** Es una solución de desarrollo: las librerías se reinstalan en cada arranque del contenedor. La alternativa recomendada, y la mejora natural de este proyecto, es construir una imagen propia de Airflow con un `Dockerfile`, tal como se hizo con la API.

**Dependencia `dill` en la API.** El entorno de Airflow incluye `dill`, y joblib lo emplea al serializar el modelo. La API debe declararlo explícitamente para poder deserializarlo. Es un ejemplo concreto de por qué conviene alinear los entornos de entrenamiento e inferencia.

## Limitaciones conocidas

- La API no valida rangos: un conjunto de medidas biológicamente imposible (por ejemplo, todas en cero) produce una predicción en lugar de un error. Un árbol de decisión siempre asigna una clase, por lo que la validación debe hacerse en el esquema de entrada.
- El modelo se entrena sobre el total de los datos, sin partición de entrenamiento y prueba, siguiendo el notebook de referencia de la asignatura. No hay, por tanto, una métrica de generalización.
- No hay versionado de modelos: cada ejecución del DAG sobrescribe el `.pkl` anterior.
