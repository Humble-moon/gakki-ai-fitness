PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

.PHONY: facts test e2e eval demo-help demo demo-check e2e-offline

facts:
	$(PYTHON) scripts/verify_project_facts.py --json

test:
	$(PYTHON) -m pytest

e2e:
	@$(PYTHON) scripts/run_e2e.py --json || { status=$$?; echo "E2E unavailable or failed; no success is fabricated."; exit $$status; }

e2e-offline: e2e

eval:
	$(PYTHON) eval/scripts/validate_metrics.py --manifest eval/evaluation_manifest.json

demo:
	$(PYTHON) scripts/run_demo.py --mode demo

demo-check:
	$(PYTHON) scripts/run_demo.py --mode demo --check

demo-help:
	@printf '%s\n' 'demo: localhost single-user demonstration boundary (no real key required)' 'full: checks configuration only and never installs/starts external services' 'test: default offline pytest (integration/live excluded)' 'e2e: offline SSE business-flow verification with explicit terminal events' 'Browser E2E is not included: this Python project has no Playwright runtime dependency.' 'No target installs dependencies or reads/prints .env.'
