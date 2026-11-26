#!/bin/sh
# Copy generated artifacts into the frontend's static folder before `npm run build`.
cd "$(dirname "$0")" && mkdir -p frontend/public && cp reports/*.html reports/summary.json reports/retrain.json eda/eda.png frontend/public/ && cp models/$(cat models/latest.txt)/metadata.json frontend/public/model.json
