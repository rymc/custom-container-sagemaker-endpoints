This is an example of how to use W&B Launch to deploy custom models to Sagemaker Endpoints.

Deploying a custom model to Sagemaker Endpoints requires a custom container.

``build_and_deploy.sh`` and `build_and_deploy.py` (the latter with W&B logging) will build and push such a container. The inference code should be in the inference directory, and it assumes the prediction is done via a script called ``inference.py``. This is where you can add your custom code on how you want to preprocess, transform, predict etc.

This can be seen as a discrete step which makes the inference code and container available for Sagemaker Endpoints to utilise.

In order to use it, We need is to deploy a model. This is handled by the `deploy.py` file. This will upload and make available a model to SageMaker endpoints, deploying it alongside the custom container. It currently relies on a model being logged as an artifact to W&B. Run the `train.py` in the training directory in order to log such a model. There is configuration within the ``deploy.py`` script that can be changed to customize which model, container and so on to use, however, the improved workflow is to connect this to W&B Launch.

With W&B Launch we can make and customize deployments from the W&B UI (or command line), as well as automatically with the W&B Model Registry. Such a workflow may involve automatically deploying any model promoted to the W&B model registry to Sagemaker Endpoints. 