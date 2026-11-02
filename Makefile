PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

.PHONY: facts test e2e eval demo-help demo demo-check e2e-offline backup restore migrate verify-shared verify-replicas audit

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

backup:
	$(PYTHON) scripts/backup_db.py backup

restore:
	@test -n "$(FILE)" || { echo "用法: make restore FILE=backups/xxx.sql.gz"; exit 2; }
	$(PYTHON) scripts/backup_db.py restore "$(FILE)"

migrate:
	$(PYTHON) -c "from src.db_migrate import upgrade_to_head; upgrade_to_head()"

# 两个脚本都要求本机 Redis 可用；不可用时以非零码退出并说明原因。
verify-shared:
	$(PYTHON) scripts/verify_shared_state.py

verify-replicas:
	$(PYTHON) scripts/verify_two_replicas.py

audit:
	$(PYTHON) -m pip_audit -r requirements.lock --strict \
		--ignore-vuln PYSEC-2026-2447 \
		--ignore-vuln PYSEC-2026-3046
