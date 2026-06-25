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
SENTINEL_VALUE = 1.79769313486232e+308  # IEEE 754 float max
HIGH_MISSINGNESS_THRESHOLD = 0.70       # Quitar columnas con >70% de missing values
RANDOM_STATE = 42
CV_FOLDS = 5
TEST_SIZE = 0.2

# Survey design columns - not features, not targets
SURVEY_DESIGN_COLS = ["weight", "stratum", "psu"]

# ---------------------------------------------------------------------------
# Regression: predecir IMC (sin usar Q4/Q5 directamente como features)
# ---------------------------------------------------------------------------
# Target: "bmi" - calculado en engineer.py como Q5/(Q4^2)
# Features: hábitos de alimentación y actividad física
#
# Mapeo CSV - encuesta (Q real, no número de pregunta del cuestionario):
#   Q6  = hambre en los últimos 30 días (escala 1-5: nunca-siempre)
#   Q7  = consumo de fruta por día (1-7: ninguna-5+ veces)
#   Q8  = consumo de verdura por día (1-7: ninguna-5+ veces)
#   Q9  = consumo de bebidas azucaradas por día (1-7: ninguna-5+ veces)
#   Q10 = días comida rápida en últimos 7 días (1-8: 0-7 días)
#   Q11 = frecuencia de cepillado dental (1-6: nunca-4+ veces/día)
#   Q12 = lavado de manos antes de comer (1-5: nunca-siempre)
#   Q13 = lavado de manos después del baño (1-5: nunca-siempre)
#   Q14 = uso de jabón al lavarse las manos (1-5: nunca-siempre)
#   Q15 = veces que fue atacado físicamente (1-8: 0-12+ veces)
#   Q16 = veces en peleas físicas (1-8: 0-12+ veces)
#   Q49 = días de actividad física ≥60 min en últimos 7 días (1-8: 0-7 días)
#   Q50 = días caminando/en bicicleta al colegio (1-8: 0-7 días)
#   Q51 = días de educación física por semana (1-6: 0-5+ días)
#   Q52 = horas sentado por día (actividades sedentarias) (1-6: <1h->8h)
#   Q53 = días que faltó al colegio sin permiso (1-5: 0-10+ días)
REGRESSION_TARGET_BMI = "bmi"  # derivado en engineer.py; NO usar Q4/Q5 como features

# Q1/Q2/Q3 son categóricas; el resto son ordinales numéricas.
REGRESSION_CATEGORICAL_COLS = ["Q1", "Q2", "Q3"]

REGRESSION_FEATURE_COLS = [
    "Q1",   # edad (1=≤11, 2=12, 3=13, 4=14, 5=15, 6=≥16)
    "Q2",   # sexo (1=Masculino, 2=Femenino)
    "Q3",   # grado escolar (1=7mo, 2=8vo, 3=9no)
    # Alimentación
    "Q6",   # frecuencia de hambre por falta de comida en casa
    "Q7",   # porciones de fruta al día
    "Q8",   # porciones de verdura al día
    "Q9",   # bebidas azucaradas al día
    "Q10",  # días de comida rápida en últimos 7 días
    # Higiene
    "Q11",  # frecuencia de cepillado dental
    "Q12",  # lavado de manos antes de comer
    "Q13",  # lavado de manos después del baño
    "Q14",  # uso de jabón al lavarse las manos
    # Actividad física
    "Q49",  # días activo ≥60 min
    "Q50",  # días caminando/bicicleta al colegio
    "Q51",  # días de educación física
    "Q52",  # horas de actividades sedentarias
    "Q53",  # días que faltó al colegio
]

# ---------------------------------------------------------------------------
# Classification: predecir riesgo de salud mental
# ---------------------------------------------------------------------------
# Target: "mental_health_risk" - creado en engineer.py
#   Definición: riesgo grave de salud mental (suicidalidad). Vale 1 si el
#   estudiante presenta CUALQUIERA de los tres indicadores del continuo suicida:
#     QN24 = consideró seriamente el suicidio en los últimos 12 meses (ideación)
#     QN25 = hizo un plan sobre cómo intentar suicidarse (plan)
#     QN26 = intentó suicidarse en los últimos 12 meses (intento)
#   Estas 3 columnas forman el target - se EXCLUYEN de las features.
#   QN22 (soledad) y QN23 (insomnio por preocupación) NO forman el target;
#   se usan como predictores afectivos del riesgo.
#
# Columnas QN excluidas por altos valores missing (>65%):
#   QN34 (65.3%), QN36 (78.7%), QN37 (81.0%), QN40 (88.4%)
#   QN45 (81.7%), QN47 (82.6%), QN48 (82.5%)
#   QN18 (76.5%), QN19 (76.0%), QN21 (81.9%)
# Excluidos por ser sub-muestra condicional de otra respuesta:
#   qnc1g (90.4%), qnc2g (93.2%)
#
# Mapeo QN - tema:
#   QN6  = hambre frecuente (indicador socioeconómico)
#   QN7  = consume fruta ≥2 veces/día
#   QN8  = consume verdura ≥3 veces/día
#   QN9  = bebe bebidas azucaradas ≥1 vez/día
#   QN10 = come comida rápida ≥3 días/semana
#   QN15 = fue atacado físicamente ≥1 vez
#   QN16 = estuvo en peleas físicas ≥1 vez
#   QN20 = fue víctima de bullying ≥1 día
#   QN22 = soledad frecuente (predictor afectivo)
#   QN23 = preocupación que impide dormir (predictor afectivo)
#   QN35 = consumió alcohol ≥1 día en últimos 30 días
#   QN38 = se emborrachó alguna vez en la vida
#   QN39 = tuvo problemas por el alcohol alguna vez
#   QN44 = ha tenido relaciones sexuales alguna vez
#   QN46 = ha tenido relaciones con 2+ personas
#   QN49 = cumple recomendación de actividad física (≥5 días)
#   QN52 = tiempo sedentario excesivo (≥3 horas/día)
#   QN53 = faltó al colegio sin permiso ≥1 día
#   QN54 = compañeros de escuela amables y serviciales (factor protector)
#   QN55 = padres revisan tareas (factor protector)
#   QN56 = padres comprenden problemas (factor protector)
#   QN57 = padres saben qué hace en tiempo libre (factor protector)
CLASSIFICATION_TARGET_MENTAL_HEALTH = "mental_health_risk"  # creado en engineer.py

# Q1/Q2/Q3 son categóricas nominales/ordinales - OHE dentro del pipeline.
# El resto de features son ordinales numéricas (QN ya son 1/2; Q son escalas 1-8).
CLASSIFICATION_CATEGORICAL_COLS = ["Q1", "Q2", "Q3"]

CLASSIFICATION_FEATURE_COLS = [
    "Q1",    # edad
    "Q2",    # sexo
    "Q3",    # grado escolar
    # Factores socioeconómicos y alimentación
    "QN6",   # hambre frecuente
    "QN7",   # consume fruta ≥2 veces/día
    "QN8",   # consume verdura ≥3 veces/día
    "QN9",   # bebidas azucaradas ≥1 vez/día
    "QN10",  # comida rápida ≥3 días/semana
    # Violencia y bullying
    "QN15",  # fue atacado físicamente
    "QN16",  # estuvo en peleas físicas
    "QN20",  # víctima de bullying
    # Factores afectivos (predictores, NO forman el target de suicidalidad)
    "QN22",  # soledad frecuente
    "QN23",  # preocupación que impide dormir
    # Sustancias
    "QN35",  # consumió alcohol en últimos 30 días
    "QN38",  # se emborrachó alguna vez
    "QN39",  # tuvo problemas por el alcohol
    "QN44",  # ha tenido relaciones sexuales
    "QN46",  # relaciones con 2+ personas
    # Actividad física y sedentarismo
    "QN49",  # cumple recomendación de actividad física
    "QN52",  # tiempo sedentario excesivo
    "QN53",  # faltó al colegio sin permiso
    # Factores protectores (apoyo social/familiar)
    "QN54",  # compañeros amables (ambiente escolar positivo)
    "QN55",  # padres revisan tareas
    "QN56",  # padres comprenden problemas
    "QN57",  # padres saben qué hace en tiempo libre
]
