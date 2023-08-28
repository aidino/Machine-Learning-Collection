
# python library imports
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
import datetime as dt

# DAG arguments
default_args = {
    'owner': 'Hong Thai, Ngo',
    'start_date': dt.datetime(2023, 7, 17),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'simple_example',
    description='A simple example DAG',
    default_args=default_args,
    schedule_interval=dt.timedelta(seconds=5), # execute every 5 seconds
)

# Task definition
task1 = BashOperator(
    task_id='print_hello',
    bash_command='echo "hello world!!!"',
    dag=dag,
)

task2 = BashOperator(
    task_id='print_date',
    bash_command='date',
    dag=dag,
)

# Task pipeline

task1 >> task2