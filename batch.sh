#!/bin/bash

# URL of the upload endpoint
URL="http://127.0.0.1:5000/image/upload"

# Directory containing images
IMAGE_DIR="image_dir"

# Iterate over all images in the directory
for IMAGE_PATH in "$IMAGE_DIR"/*; do
  # Send a POST request for each image
  curl --location "$URL" \
  --form "image=@$IMAGE_PATH"
done
