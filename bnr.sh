#!/bin/bash
# Build n Run helper script
set -xe
docker build . -t gpt2-api
docker run --rm --name gpt2-api --cpus=4 -p 2666:2666 -ti gpt2-api