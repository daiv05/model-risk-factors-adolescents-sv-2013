.PHONY: install data-check eda train evaluate streamlit clean

PYTHON := python
DATA := data/SLV2013_Public_Use.csv

install:
	$(PYTHON) -m pip install -e ".[dev]"

data-check:
	@$(PYTHON) -c "\
import pandas as pd, numpy as np; \
df = pd.read_csv('$(DATA)'); \
sentinel = 1.79769313486232e+308; \
n_sentinel = (df == sentinel).sum().sum(); \
print(f'Rows: {len(df)}, Cols: {len(df.columns)}, Sentinel cells: {n_sentinel}'); \
print('Data check passed.' if n_sentinel > 0 else 'WARNING: no sentinel values found.')"

eda:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute notebooks/01_data_exploration.ipynb --inplace
	$(PYTHON) -m jupyter nbconvert --to notebook --execute notebooks/02_feature_analysis.ipynb --inplace

train:
	$(PYTHON) -m src.models.train

evaluate:
	$(PYTHON) -m src.models.evaluate
	$(PYTHON) -m src.analysis.benchmark
	$(PYTHON) -m src.analysis.ablation

streamlit:
	streamlit run src/visualization/app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f reports/*.json reports/*.csv reports/*.png 2>/dev/null || true
