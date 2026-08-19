PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

.PHONY: facts test e2e eval demo-help

facts:
	$(PYTHON) scripts/verify_project_facts.py --json

test:
	$(PYTHON) -m pytest

e2e:
	@$(PYTHON) -m pytest -m e2e -q || { status=$$?; echo "E2E unavailable or failed; no success is fabricated."; exit $$status; }

eval:
	$(PYTHON) eval/scripts/validate_metrics.py --manifest eval/evaluation_manifest.json

demo-help:
	@printf '%s\n' 'demo: localhost single-user demonstration boundary' 'full: requires configured external services and credentials' 'test: default offline pytest (integration/live excluded)' 'e2e: opt-in/core E2E; missing dependencies fail or skip explicitly' 'No target installs dependencies or reads/prints .env.'
