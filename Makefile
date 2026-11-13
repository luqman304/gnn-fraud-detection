.PHONY: install lint test prepare eval train serve docker

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

test:
	pytest

prepare:
	gnnfraud prepare

eval:
	gnnfraud eval --seeds 5 --device auto

train:
	gnnfraud train --model rf --features all

serve:
	gnnfraud serve

docker:
	docker compose up --build api
