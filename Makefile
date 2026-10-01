CAMPAIGN ?= campaigns/quick.yaml
MODEL ?= models:/flroad/1
ARGS ?= -e demo

.PHONY: up down ui run sweep serve check

up:  ## Start the MLflow server (http://127.0.0.1:5000)
	docker compose up -d --build mlflow

down:  ## Stop all containers (data is kept)
	docker compose down

run:  ## One run: make run ARGS="-e demo --set seed=7"
	docker compose run --rm flroad run $(ARGS)

sweep:  ## A campaign: make sweep CAMPAIGN=campaigns/epochs.yaml
	docker compose run --rm flroad sweep $(CAMPAIGN)

serve:  ## Serve a model: make serve MODEL=models:/flroad/2
	docker compose run --rm --service-ports api serve \
		--host 0.0.0.0 --model $(MODEL)

check:  ## Local checks: format, lint, types, tests
	ruff format --check .
	ruff check .
	mypy
	pytest
