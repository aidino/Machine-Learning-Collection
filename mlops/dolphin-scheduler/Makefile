# Makefile
SHELL = /bin/bash

# Initialize the database, use profile schema
.PHONY: initdb
initdb:
	docker-compose --profile schema up -d

# start all dolphinscheduler server, use profile all
.PHONY: start
start:
	docker-compose --profile all up -d