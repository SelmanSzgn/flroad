CAMPAIGN ?= campaigns/quick.yaml
MODEL ?= models:/flroad/1
ARGS ?= -e demo

.PHONY: up down run sweep serve check clean

up:
	docker compose up -d --build mlflow

down:
	docker compose down

run:
	docker compose run --rm flroad run $(ARGS)

sweep:
	docker compose run --rm flroad sweep $(CAMPAIGN)

serve:
	docker compose run --rm --service-ports api serve \
		--host 0.0.0.0 --model $(MODEL)

check:
	ruff format --check .
	ruff check .
	mypy
	pytest

clean:
	docker compose down -v
	rm -rf mlflow.db mlruns mlartifacts
	sudo rm -rf runs
