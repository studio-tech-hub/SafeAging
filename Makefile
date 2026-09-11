# SafeAging — developer shortcuts
# Usage: make <target>

SERVICE   = safeaging-analytics
PYTHON    = docker exec -w /app/python $(SERVICE) python

# ── Docker lifecycle ──────────────────────────────────────────────────────────
up:
	docker compose up -d

down:
	docker compose down

rebuild:
	docker compose build analytics
	docker compose up -d --force-recreate analytics

rebuild-all:
	docker compose down
	docker compose build --no-cache
	docker compose up -d
	docker exec -w /app/python safeaging-analytics alembic upgrade head

logs:
	docker compose logs -f analytics

# ── Migrations ────────────────────────────────────────────────────────────────
migrate:
	docker exec -w /app/python $(SERVICE) alembic upgrade head

# ── Tests ─────────────────────────────────────────────────────────────────────

# Fast unit tests — no service needed
test-unit:
	$(PYTHON) -m pytest tests/unit/ -v --tb=short

# Integration tests — requires: make up
test-integration:
	docker exec -w /app/python -e SERVICE_URL=http://localhost:18000 \
		$(SERVICE) python -m pytest tests/integration/ -m integration -v --tb=short

# P P1.3 – CB/queue/reconnect/metadata consistency tests (subset of integration)
test-plugin-behavior:
	docker exec -w /app/python -e SERVICE_URL=http://localhost:18000 \
		$(SERVICE) python -m pytest \
		tests/integration/test_plugin_behavior.py \
		-m integration -v --tb=short

# C++ unit tests — circuit breaker + detection box normalizer (requires g++ in
# WSL or Linux; no Nx SDK, no OpenCV). Mirrors .github/workflows/ci.yml's
# build-plugin job so a local `make test-cpp` catches the same breakage a PR
# would, without needing the (licensed, non-redistributable) Nx Metadata SDK.
test-cpp:
	cd src/tests && \
		g++ -std=c++17 \
		    -I../sample_company/vms_server_plugins/opencv_object_detection \
		    test_circuit_breaker.cpp -o test_circuit_breaker -pthread && \
		./test_circuit_breaker && \
		g++ -std=c++17 \
		    -I../sample_company/vms_server_plugins/opencv_object_detection \
		    test_detection_box_normalizer.cpp -o test_detection_box_normalizer && \
		./test_detection_box_normalizer

# Compose security lint (P0-2/P0-3) — static + dynamic checks, same as
# .github/workflows/ci.yml's compose-lint job.
test-compose-lint:
	python -m pytest tools/test_check_compose_security.py tools/test_generate_secrets.py -v
	python tools/check_compose_security.py
	python tools/verify_compose_config.py

# Secret scan (P0-1 regression guard) — same as .github/workflows/ci.yml's
# secret-scan job. Requires: pip install detect-secrets==1.5.0
test-secrets:
	detect-secrets-hook --baseline .secrets.baseline $$(git ls-files)

# Load tests — requires: make up
test-load:
	docker exec -w /app/python \
		-e SERVICE_URL=http://localhost:18000 \
		-e LOAD_CAMERAS=4 \
		-e LOAD_FRAMES=10 \
		$(SERVICE) python -m pytest tests/load/ -m load -v --tb=short -s

# All tests (unit + integration)
test:
	$(PYTHON) -m pytest tests/unit/ -v --tb=short
	docker exec -w /app/python \
		-e SERVICE_URL=http://localhost:18000 \
		$(SERVICE) python -m pytest tests/integration/ -m integration -v --tb=short

.PHONY: up down rebuild rebuild-all logs migrate \
        test-unit test-integration test-plugin-behavior test-cpp test-load test \
        test-compose-lint test-secrets
