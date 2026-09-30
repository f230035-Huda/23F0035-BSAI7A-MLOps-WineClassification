PYTHON := python
PIP := python -m pip

.PHONY: install lint test train clean

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

lint:
	$(PYTHON) -m flake8 src/ tests/ --max-line-length=100

test:
	$(PYTHON) -m pytest -v

train:
	$(PYTHON) -m src.train

clean:
	$(PYTHON) -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"
	$(PYTHON) -c "import shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__') if p.is_dir()]"
	$(PYTHON) -c "import shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.pytest_cache') if p.is_dir()]"
