from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent
DATA_PATH = ROOT_DIR / "data" / "SLV2013_Public_Use.csv"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Data constants
# ---------------------------------------------------------------------------
SENTINEL_VALUE = 1.79769313486232e+308  # IEEE 754 float max used by WHO GSHS software
HIGH_MISSINGNESS_THRESHOLD = 0.70       # drop columns with more missing than this
RANDOM_STATE = 42
CV_FOLDS = 5
TEST_SIZE = 0.2

# Survey design columns — not features, not targets
SURVEY_DESIGN_COLS = ["weight", "stratum", "psu"]

# ---------------------------------------------------------------------------
# Regression configuration
# ---------------------------------------------------------------------------
# Target: Q4 = altura (m), Q5 = peso (kg)  — variables continuas reales
REGRESSION_TARGET_HEIGHT = "Q4"
REGRESSION_TARGET_WEIGHT = "Q5"

REGRESSION_FEATURE_COLS = [
    "Q1",   # grupo de edad (1-6)
    "Q2",   # sexo (1-2)
    "Q3",   # grado escolar (1-3)
    "Q6",   # días con hambre (1-5)
    "Q7",   # días de actividad física (1-7)
    "Q8",   # horas sedentarias (1-7)
    "Q9",   # horas de sueño (1-7)
    "Q10",  # porciones de fruta (1-8)
    "Q11",  # porciones de verdura (1-6)
    "Q12",  # días con bebidas azucaradas (1-5)
    "Q13",  # días de comida rápida (1-5)
    "Q14",  # días con desayuno (1-5)
    "Q15",  # frecuencia de cepillado (1-8)
    "Q16",  # frecuencia de lavado de manos (1-8)
    "Q49",  # días de uso de tabaco (1-8)
    "Q50",  # días de consumo de alcohol (1-8)
    "Q54",  # días con ansiedad (1-5)
    "Q55",  # días de soledad (1-5)
    "Q56",  # días con dificultad para dormir (1-5)
    "Q57",  # días de tristeza/desesperanza (1-5)
]

# ---------------------------------------------------------------------------
# Classification configuration
# ---------------------------------------------------------------------------
# Targets disponibles (seleccionar uno a la vez):
#   qnowtg    — sobrepeso
#   qnobeseg  — obesidad
#   qnunwtg   — bajo peso
#   qnpa7g    — cumple recomendación de actividad física
#   qnfrvgg   — cumple recomendación de frutas/verduras
#   qnpe5g    — tiempo sedentario excesivo
CLASSIFICATION_TARGET_DEFAULT = "qnowtg"

CLASSIFICATION_ALL_TARGETS = [
    "qnowtg",
    "qnobeseg",
    "qnunwtg",
    "qnpa7g",
    "qnfrvgg",
    "qnpe5g",
]

# QN* = dicotomizaciones validadas por la OMS de las preguntas Q correspondientes.
# Se excluyen columnas QN con >70% de missingness:
#   QN18, QN19, QN21, QN34, QN36, QN37, QN40, QN44, QN45, QN47, QN48
# Se excluyen qnc1g y qnc2g (>90% missing).
CLASSIFICATION_FEATURE_COLS = [
    "Q1",    # grupo de edad
    "Q2",    # sexo
    "Q3",    # grado escolar
    "QN6",   # hambre (binario)
    "QN7",   # activo físicamente (binario)
    "QN9",   # duerme suficiente (binario)
    "QN10",  # consume suficiente fruta (binario)
    "QN11",  # consume suficiente verdura (binario)
    "QN12",  # bebidas azucaradas en riesgo (binario)
    "QN13",  # comida rápida en riesgo (binario)
    "QN14",  # desayuna diariamente (binario)
    "QN16",  # se lava las manos (binario)
    "QN22",  # involucrado en peleas (binario)
    "QN23",  # víctima de bullying (binario)
    "QN26",  # ideación suicida (binario)
    "QN35",  # fumador actual (binario)
    "QN38",  # ha consumido alcohol alguna vez (binario)
    "QN39",  # consumo actual de alcohol (binario)
    "QN46",  # uso de drogas (binario)
    "QN49",  # uso de tabaco (binario)
    "QN50",  # consumo excesivo de alcohol (binario)
    "QN54",  # ansiedad (binario)
    "QN55",  # soledad (binario)
    "QN57",  # tristeza/desesperanza (binario)
]
