# unfuse
#
# The default target is what to run before pushing. Everything here works on a
# laptop with no GPU: the tests cover the model-agnostic core, and the parts that
# need a 20 GB checkpoint live in the notebooks, whose committed outputs are the
# record that they ran.

.PHONY: help check lint fmt test notebooks install inspect clean

help:
	@echo 'Targets:'
	@echo '  check      lint + tests (run this before pushing)'
	@echo '  lint       ruff check and format --check'
	@echo '  fmt        reformat in place'
	@echo '  test       pytest'
	@echo '  notebooks  check the committed notebooks still carry their outputs'
	@echo '  install    pip install -e ".[notebook,gguf,dev]"'
	@echo '  inspect    show a checkpoint'\''s tensor layout  (make inspect CKPT=~/models/x)'
	@echo '  clean      remove caches'

check: lint test notebooks

lint:
	ruff check .
	ruff format --check .

fmt:
	ruff format .
	ruff check --fix .

test:
	pytest -q

notebooks:
	python tools/check_notebooks.py

install:
	pip install -e ".[notebook,gguf,dev]"

CKPT ?=
inspect:
	@test -n '$(CKPT)' || { echo 'usage: make inspect CKPT=/path/to/checkpoint'; exit 1; }
	python -m unfuse.cli inspect '$(CKPT)' --prefix model.visual.

clean:
	rm -rf .ruff_cache .pytest_cache dist build
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
