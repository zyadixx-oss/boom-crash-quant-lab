.PHONY: setup backend frontend test docker docker-down

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r backend/requirements.txt
	cd frontend && npm install

backend:
	.venv/bin/uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000 --reload

frontend:
	cd frontend && npm run dev -- --host 0.0.0.0

test:
	PYTHONPATH=backend .venv/bin/pytest -q backend/tests tests

docker:
	docker compose up --build

docker-down:
	docker compose down
