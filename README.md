This is an example of how to use W&B Launch to deploy custom models to Sagemaker Endpoints.

## TL;DR

1. Run ``train.py`` to train a model which will log it to W&B.
2. For the remaining steps, we will be interacting with AWS, so ensure you have valid credentials in your .aws directory, or environment variables (AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN).
3. run ``python build_and_push.py   --image-name <The name to give the built image  [required]> --wandb-project  <The name of the W&B project to record the build  of the run in> --inference-code-dir <The directory containing the inference code for the container>`` which will build and push the container with the specified image-name to ECR and log the build details to W&B.
4. run ``python deploy.py   --role <The SageMaker role to be used> --image-uri <The URI of the image, from the previous step> --sagemaker-bucket <The S3 bucket for SageMaker Models> --artifact <The W&B model artifact string path to be deployed> --wandb-project <The W&B run to record the project in> --instance-type <The AWS instance type fore inference>`` which will deploy the model to SageMaker Endpoints.

## Demo Videos
Overview Video of an example workflow https://www.loom.com/share/13e3d743387f4a0cb7e568d6fdfdaaa4?sid=b59d5e9c-1a6b-4a86-b3f8-3e85a2ab3530

Behind The Scenes (more detail): https://www.loom.com/share/325200335d4149bbb12f7393495da3b6?sid=b047577e-92e8-4826-8784-c09f07f73ebb

## More details

Deploying a custom model to Sagemaker Endpoints requires a custom container.

``build_and_deploy.sh`` and `build_and_deploy.py` (the latter with W&B logging) will build and push such a container. The inference code should be in the inference directory, and it assumes the prediction is done via a script called ``inference.py``. This is where you can add your custom code on how you want to preprocess, transform, predict etc.

This can be seen as a discrete step which makes the inference code and container available for Sagemaker Endpoints to utilise.

In order to use it, We need is to deploy a model. This is handled by the `deploy.py` file. This will upload and make available a model to SageMaker endpoints, deploying it alongside the custom container. It currently relies on a model being logged as an artifact to W&B. Run the `train.py` in the training directory in order to log such a model. There is configuration within the ``deploy.py`` script that can be changed to customize which model, container and so on to use, however, the improved workflow is to connect this to W&B Launch.

With W&B Launch we can make and customize deployments from the W&B UI (or command line), as well as automatically with the W&B Model Registry. Such a workflow may involve automatically deploying any model promoted to the W&B model registry to Sagemaker Endpoints. 

You can see more details about this here: https://wandb.ai/wandb-smle/sagemaker-endpoints-custom-models/reports/Custom-Models-to-SageMaker-Endpoints--Vmlldzo2MTcwNDI1
