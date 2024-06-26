""" This code is based on https://github.com/aws/amazon-sagemaker-examples/tree/main """
#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
image=$1
project=$2

if [ -z "$image" ] || [ -z "$project" ]; then
    echo "Usage: $0 <image-name> <project-name>"
    exit 1
fi

chmod +x inference/serve

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)


fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi

aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com


# Build the docker image locally with the image name and then push it to ECR
# with the full name.

docker build --platform=linux/amd64 -t ${image} .
docker tag ${image} ${fullname}

docker push ${fullname}

docker save --output ${image}.tar ${image}

wandb artifact put --name ${project}/${image} --type dockerfile Dockerfile