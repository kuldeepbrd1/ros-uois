#!/bin/bash

ORG_NAME="kuldeepbrd1"
IMAGE_NAME="uois-cuda"
VERSION="0.1.0"
IMAGE_TAG=${ORG_NAME}/${IMAGE_NAME}:${VERSION}

## Conda version
# DOCKERFILE="conda-realsense-ros-noetic-focal.Dockerfile"
# DOCKERFILE="realsense-ros-noetic-focal.Dockerfile"
DOCKERFILE="uois_cuda.Dockerfile"
DATE="$(date)"

docker build --network=host --build-arg GIT_HASH=`git rev-parse --short HEAD` --build-arg DATE="$(date -I)" -f ${DOCKERFILE} -t ${IMAGE_TAG} ./..

LATEST_TAG=${ORG_NAME}/${IMAGE_NAME}:latest
docker tag ${IMAGE_TAG} ${LATEST_TAG}
