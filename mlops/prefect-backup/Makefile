# Makefile
SHELL = /bin/bash

.PHONY: server
server:
	docker-compose --profile server up

.PHONY: cli
cli:
	docker-compose run cli

.PHONY: agent
agent:
	docker-compose --profile agent up