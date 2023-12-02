from sagemaker.model import Model
from sagemaker import Session

import shutil
import tarfile
import click
import wandb


def wandb_termlog_heading(text):
    return wandb.termlog(click.style("sagemaker deploy: ", fg="green") + text)


def err_raise(msg, e=ValueError):
    wandb.termerror(msg)
    raise e(msg)


role = 'arn:aws:iam::770934259321:role/service-role/AmazonSageMaker-ExecutionRole-20231201T125441'  # replace with your role ARN
image_uri = '770934259321.dkr.ecr.eu-west-2.amazonaws.com/custom_inf:latest' # replace with your ECR image URI
sagemaker_bucket = 'gpstec417-builder-session-eu-west-2-770934259321'  # replace with your S3 path to the model


config = {
        "role": role,
        "image_uri": image_uri,
        "sagemaker_bucket": sagemaker_bucket,
        "artifact": "wandb-smle/sagemaker-custom-deploy/clf:v0",
    }

with wandb.init(project="deploy_sagemaker_custom", job_type="deployment", config=config) as run:

    wandb_termlog_heading("Downloading artifact from wandb")

    path = run.use_artifact(run.config["artifact"]).download()
 
    wandb_termlog_heading("Creating temp directory for sagemaker model")
    name_ver = path.split("/")[-1]
    name, ver = name_ver.split(":v")
    target = f"temp/{name}/{name}/{ver}"
    shutil.copytree(path, target)


    model_str = f"{name}-{ver}"
    model_tar = f"temp/{model_str}.tar.gz"
    with tarfile.open(model_tar, mode="w:gz") as archive:
        archive.add(target, arcname="")

    wandb_termlog_heading("Uploading model to S3")
    session = Session()
    model_data = session.upload_data(
        bucket=run.config["sagemaker_bucket"], path=model_tar,
    )

    wandb_termlog_heading(
        "Deploy model to Sagemaker Endpoints (this may take a while...)")
    

    model = Model(
        image_uri=run.config["image_uri"],
        model_data=model_data,
        role=run.config["role"],
        sagemaker_session=session,
        name=name_ver.replace(":","-")
    )

    # To deploy the model (create an endpoint), you can use:
    predictor = model.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

    wandb_termlog_heading(f"Successfully deployed model {model.name} at endpoint {model.endpoint_name}")
    run.log({"sagemaker_endpoint": model.endpoint_name})
    run.log_code()
