# Factores de Riesgo en Adolescentes — El Salvador (WHO GSHS 2013)

Proyecto de aprendizaje que aplica machine learning a datos reales de la Encuesta Mundial de Salud Escolar (GSHS) 2013 de El Salvador (OMS). Se construyen dos modelos dentro de un mismo pipeline:

- **Regresión** — estima (o intenta estimar) el IMC a partir de hábitos de alimentación y actividad física, sin usar peso ni estatura directamente.
- **Clasificación** — detecta riesgo grave de salud mental a partir de factores de riesgo y protección.

> Proyecto educativo. No constituye una guía médica (ni está cerca de serla)

---

## Datos

El dataset (`data/SLV2013_Public_Use.csv`) **NO SE INCLUYE** en el repositorio. Para llevar a cabo el análisis, se debe descargar desde [WHO NCD Microdata Repository](https://extranet.who.int/ncdsmicrodata/index.php/catalog/97) y luego colocarlo en `data/`. Su uso está sujeto a los términos de la OMS.

El archivo debe tener por nombre `SLV2013_Public_Use.csv` y estar en la carpeta `data/`.

---

## Requisitos

- Python 3.10+
- El dataset en `data/SLV2013_Public_Use.csv`

---

## Levantar y correr

### 1. Crear y activar el entorno virtual

```bash
python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1
# Windows (cmd)
venv\Scripts\activate.bat
# Linux / macOS
source venv/bin/activate
```

Con el entorno activo, todos los comandos siguientes instalan y corren dentro de `venv`.

### 2. Usando el Makefile

```bash
make install     # Instala el paquete y dependencias (modo editable)
make train       # Entrena los 4 modelos y los guarda en models/
make evaluate    # Genera métricas, matrices de confusión, benchmark y ablación en reports/
make streamlit   # Lanza el dashboard interactivo
```

`train` debe correr antes que `evaluate` y `streamlit`: produce los `.joblib` que ambos consumen.

### Comandos adicionales

```bash
make eda         # Ejecuta los notebooks de exploración
make clean       # Limpia caché, checkpoints y reportes generados
```

### Sin make

Con el entorno virtual ya activado (paso 1):

```bash
pip install -e ".[dev]"
python -m src.models.train
python -m src.models.evaluate
streamlit run src/visualization/app.py
```

---

## Estructura

```
data/        Dataset (no incluido)
notebooks/   EDA y prototipado (01_data_exploration, 02_feature_analysis)
src/
  config.py        Columnas, targets, constantes
  data/load.py     Carga CSV + limpieza
  features/        IMC, riesgo de salud mental, scores
  models/          train, evaluate, tune
  analysis/        benchmark, ablation
  simulation/      Simulador de escenarios
  visualization/   plots reutilizables + dashboard Streamlit
models/      Pipelines entrenados (generado)
reports/     Métricas JSON + figuras PNG (generado)
```

---

## Modelos

| Tarea | Modelos | Métrica principal |
|---|---|---|
| Regresión (IMC) | LinearRegression , RandomForestRegressor | RMSE, R^2 |
| Clasificación (riesgo de salud mental) | LogisticRegression , RandomForestClassifier + SMOTE | F1 clase minoritaria, AUC-ROC |

---

## Licencia

- **Código:** MIT — ver [LICENSE](LICENSE)
