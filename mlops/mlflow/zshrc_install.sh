#!/bin/bash

source .env

echo "# MLFLOW CONFIG" >> ~/.zshrc
echo "export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID" >> ~/.zshrc
echo "export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY" >> ~/.zshrc
echo "export MLFLOW_S3_ENDPOINT_URL=$MLFLOW_S3_ENDPOINT_URL" >> ~/.zshrc
echo "export MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI" >> ~/.zshrc
echo "# END MLFLOW CONFIG" >> ~/.zshrc

echo "[ OK ] Successfully installed environment variables into your .zshrc!"