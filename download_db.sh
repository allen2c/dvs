#!/bin/bash

# Verify if DOWNLOAD_DB_URL is set
if [ -z "$DOWNLOAD_DB_URL" ]; then
    echo "Error: DOWNLOAD_DB_URL environment variable is not set."
    echo "Please set it using: export DOWNLOAD_DB_URL='your_url_here'"
    exit 1
fi

FILENAME="./documents.duckdb"

# Check if file already exists
if [ -f "$FILENAME" ]; then
    echo "File already exists: $FILENAME"
    echo "Skipping download..."
    exit 0
fi

echo "Downloading from: $DOWNLOAD_DB_URL"
echo "Saving as: $FILENAME"

# Download file with curl and custom progress bar
curl -o "$FILENAME" "$DOWNLOAD_DB_URL"

echo -e "\nDownload complete: $FILENAME"
