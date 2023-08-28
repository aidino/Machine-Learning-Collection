## Airflow Official

- **Setting the right Airflow user**

```bash
mkdir -p ./dags ./logs ./plugins ./config
echo -e "AIRFLOW_UID=$(id -u)" > .env
```

- **Initialize the database**
  
```bash
docker compose up airflow-init
```

- **Clean up the environment**

```bash
docker compose down --volumes --remove-orphans
```

[Read more](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)