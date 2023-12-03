import subprocess
import sys
import wandb
import os

def run_command(command):
    """ Run shell commands """
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(command)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"Error executing {command}: {result.stderr}")
    return result.stdout.strip(), result.returncode 

# Check if the image name is provided
if len(sys.argv) < 2:
    print("Usage: python build_and_deploy.py <image-name>")
    sys.exit(1)

image = sys.argv[1]

wandb.init(project="deploy_sagemaker_custom", job_type="build_container")

wandb.config["image_name"] = image

# Make sure the 'serve' script is executable
subprocess.run("chmod +x inference/serve", shell=True)

# Get AWS account and region
account, _ = run_command("aws sts get-caller-identity --query Account --output text")
region, _ = run_command("aws configure get region")

fullname = f"{account}.dkr.ecr.{region}.amazonaws.com/{image}:latest"

# Check if the repository exists in ECR, create it if not
_, returncode = run_command(f"aws ecr describe-repositories --repository-names \"{image}\"")
if returncode != 0:
    run_command(f"aws ecr create-repository --repository-name \"{image}\"")

# Get ECR login password
run_command(f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account}.dkr.ecr.{region}.amazonaws.com")

# Build, tag, and push the Docker image
run_command(f"docker build --platform=linux/amd64 -t {image} .")
run_command(f"docker tag {image} {fullname}")
run_command(f"docker push {fullname}")

# Save the image to a tar file
run_command(f"docker save --output {image}.tar {image}")

artifact = wandb.Artifact(name=image, type="dockerfile")
artifact.add_file("Dockerfile")  
wandb.log_artifact(artifact)  
wandb.run.log_code(".")