#!/bin/bash

set -e

# Run from project root so the Dockerfile can access both protein-viz-agent/ and logo/
cd "$(dirname "$0")"
mkdir -p protein-viz-agent/structures protein-viz-agent/membranes protein-viz-agent/prepared

docker rm -f $(docker ps -a -q --filter ancestor=parora) 2>/dev/null || true
docker rmi -f parora 2>/dev/null || true
docker build -t parora -f protein-viz-agent/Dockerfile .
docker run -p 8501:8501 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -v "$(pwd)/protein-viz-agent/structures:/app/structures" \
  -v "$(pwd)/protein-viz-agent/membranes:/app/membranes" \
  -v "$(pwd)/protein-viz-agent/prepared:/app/prepared" \
  parora
