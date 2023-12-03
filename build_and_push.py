import subprocess
import sys
import wandb
import os
import click


def run_command(command):
    """ Run shell commands """
    result = subprocess.run(command, shell=True,
                            capture_output=True, text=True)
    print(command)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        print(f"Error executing {command}: {result.stderr}")
    return result.stdout.strip(), result.returncode


@click.command()
@click.option('--image-name', required=True, help='The name to give the built image')
@click.option('--wandb-project', default='sagemaker_endpoint_deploy', required=False, help='The name of the W&B project to record the build of the run in.')
@click.option('--inference-code-dir', default='inference', required=False, help='The directory containing the inference code for the container.')
def main(image_name, wandb_project, inference_code_dir):

    config = {
        "image_name": image_name,
        "wandb_project": wandb_project,
    }
    wandb.init(project=config['wandb_project'],
               job_type="build_container", config=config, settings=wandb.Settings(disable_git=True))
    image = wandb.config.image_name


    # Make sure the 'serve' script is executable
    subprocess.run(f"chmod +x {inference_code_dir}/serve", shell=True)

    # Get AWS account and region
    account, _ = run_command(
        "aws sts get-caller-identity --query Account --output text")
    region, _ = run_command("aws configure get region")

    fullname = f"{account}.dkr.ecr.{region}.amazonaws.com/{image}:latest"

    # Check if the repository exists in ECR, create it if not
    _, returncode = run_command(
        f"aws ecr describe-repositories --repository-names \"{image}\"")
    if returncode != 0:
        run_command(f"aws ecr create-repository --repository-name \"{image}\"")

    # Get ECR login password
    run_command(
        f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account}.dkr.ecr.{region}.amazonaws.com")

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

if __name__ == '__main__':
    main()
