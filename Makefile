.PHONY: install data-check eda train evaluate streamlit clean

PYTHON := python

install:
	$(PYTHON) -m pip install -e ".[dev]"

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
