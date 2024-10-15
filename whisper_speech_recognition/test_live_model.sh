#!/bin/bash

# Variables
ACCOUNT_NAME=$1  # Account name (required)
MODEL_ID=$2      # Model ID (required)
QWAK_TOKEN=${QWAK_TOKEN}  # Qwak token (ensure it's set as an environment variable)

# Check if account name and model ID are provided
if [ -z "$ACCOUNT_NAME" ] || [ -z "$MODEL_ID" ]; then
  echo "Usage: $0 <account_name> <model_id>"
  exit 1
fi

# Ensure QWAK_TOKEN is set
if [ -z "$QWAK_TOKEN" ]; then
  echo "Error: QWAK_TOKEN is not set. Please export the token or pass it as an environment variable."
  exit 1
fi

# Test file
FILE_PATH="harvard.wav"

# Ensure the file exists
if [ ! -f "$FILE_PATH" ]; then
  echo "Error: File '$FILE_PATH' not found."
  exit 1
fi

# CURL command to send the request
curl --location --request POST "https://models.$ACCOUNT_NAME.qwak.ai/v1/$MODEL_ID/predict" \
--header 'Authorization: Bearer '$QWAK_TOKEN'' \
--form "file=@$FILE_PATH"
