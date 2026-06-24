# Factores de Riesgo en Adolescentes — El Salvador (WHO GSHS 2013)

Proyecto de aprendizaje que aplica técnicas de machine learning a datos reales de salud pública: la Encuesta Mundial de Salud Escolar (GSHS) 2013 de El Salvador, conducida por la Organización Mundial de la Salud.

> **Nota:** Este es un proyecto personal con fines educativos. No constituye guía médica ni de política pública oficial.

---

## Contexto

La encuesta WHO Global School-based Student Health Survey (GSHS) recopila datos de estudiantes de 13 a 17 años sobre tabaco, alcohol, hábitos alimenticios, actividad física, higiene, salud mental, exposición a violencia y factores protectores. Este proyecto aplica modelos de regresión y clasificación supervisados para identificar patrones y predecir resultados de riesgo.

El conjunto de datos de El Salvador 2013 contiene **1,914 registros** y **102 variables**.

---

## Fuente de Datos y Atribución

| | |
|---|---|
| **Dataset** | El Salvador GSHS 2013 |
| **Fuente** | WHO NCD Microdata Repository |
| **URL** | https://extranet.who.int/ncdsmicrodata/index.php/catalog/97 |
| **Incluido en este repo** | **NO** |

El dataset **no está incluido** en este repositorio. Debe descargarse directamente desde la fuente oficial de la OMS. Consulta [LICENSE](LICENSE) para ver los términos de uso.

---

## Estructura del Proyecto

```
├── LICENSE                   # MIT (código) + disclaimer dataset OMS
├── Makefile                  # Orquestador del pipeline
├── pyproject.toml            # Dependencias y configuración del proyecto
├── README.md
├── data/
│   └── SLV2013_Public_Use.csv    # ← Descargar desde la OMS (no incluido)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_prototyping.ipynb
├── src/
│   ├── config.py             # Columnas, constantes, rutas
│   ├── data/
│   │   ├── load.py           # Carga CSV + reemplaza sentinel por NaN
│   │   ├── clean.py          # Limpieza, outliers, recodificación binaria
│   │   └── impute.py         # Imputación moda/mediana
│   ├── features/
│   │   └── engineer.py       # IMC, scores compuestos de riesgo
│   ├── models/
│   │   ├── train.py          # Entrena los 4 modelos con cross-validation
│   │   ├── evaluate.py       # Métricas, matriz de confusión, importancia
│   │   └── tune.py           # GridSearchCV / RandomizedSearchCV
│   ├── analysis/
│   │   ├── ablation.py       # Experimentos leave-one-feature-group-out
│   │   └── benchmark.py      # Comparación de modelos
│   ├── simulation/
│   │   └── scenarios.py      # Simulador what-if
│   └── visualization/
│       ├── plots.py          # Figuras reutilizables (matplotlib/seaborn)
│       └── app.py            # Dashboard Streamlit (5 páginas)
├── models/                   # Pipelines entrenados (generado, no versionado)
└── reports/                  # Métricas JSON + figuras PNG (generado)
```

---

## Modelos

### Regresión

| Modelo | Variable objetivo | Notas |
|---|---|---|
| `LinearRegression` | `Q4` — altura (m) | Línea base |
| `RandomForestRegressor` | `Q5` — peso (kg) | Captura no linealidades |

Ambos con validación cruzada 5-fold.

### Clasificación

| Modelo | Variable objetivo | Manejo del desbalance |
|---|---|---|
| `LogisticRegression` | `qnowtg` — sobrepeso | `class_weight='balanced'` |
| `RandomForestClassifier` | `qnobeseg` — obesidad | SMOTE (`imblearn.pipeline`) |

> SMOTE se aplica **dentro** del pipeline de CV para evitar fuga de datos.

---

## Configuración

### Requisitos
- Python 3.10+
- Dataset descargado en `data/SLV2013_Public_Use.csv`

### Instalación

```bash
pip install -e ".[dev]"
# o
make install
```

---

## Uso

```bash
make data-check   # Verifica que el dataset existe y detecta valores sentinel
make eda          # Ejecuta y exporta los notebooks de EDA
make train        # Entrena todos los modelos, guarda en models/
make evaluate     # Genera métricas, gráficas y reportes en reports/
make streamlit    # Lanza el dashboard interactivo
make clean        # Limpia __pycache__, checkpoints y reportes generados
```

---

## Decisiones de Diseño

**Valor sentinel `1.79769313486232e+308`** — Es el máximo del tipo `float64` (IEEE 754), usado por el software WHO GSHS para marcar respuestas no aplicables. Se reemplaza por `NaN` en la carga para evitar que cualquier operación aritmética produzca resultados incorrectos silenciosamente.

**Columnas QN vs Q para clasificación** — Las columnas `QN*` son dicotomizaciones validadas por la OMS usando umbrales de salud establecidos, más robustas que aplicar umbrales ad-hoc a las columnas `Q*`. Las columnas `QN` con >70% de datos faltantes se excluyen explícitamente en `config.py`.

**SMOTE en `imblearn.pipeline`** — Al incluir SMOTE en el pipeline de `imbalanced-learn` (no de scikit-learn), el sobremuestreo ocurre únicamente en los folds de entrenamiento de la CV, evitando la fuga de datos hacia los folds de validación.

**Variables de diseño muestral excluidas** — `weight`, `stratum` y `psu` son variables del diseño muestral complejo. Se conservan en el DataFrame para posible uso futuro con modelos ponderados, pero se excluyen de todas las listas de características.

---

## Licencia

- **Código fuente:** MIT License — ver [LICENSE](LICENSE)
- **Dataset:** No incluido. Sujeto a los términos de uso de la OMS.
