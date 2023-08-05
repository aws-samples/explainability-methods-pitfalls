# Explanations in AI: Methods, Stakeholders and Pitfalls (KDD 2023)

While using vast amounts of training data and sophisticated models has enhanced the predictive performance of Machine Learning (ML) and Artificial Intelligence (AI) solutions, it has also led to an increased difficulty in comprehending their predictions. The ability to explain predictions is often one of the primary desiderata for adopting AI and ML solutions.

The desire for explainability has led to a rapidly growing body of literature on explainable AI (XAI) and has also resulted in the development of hundreds of XAI methods targeting different domains (e.g., finance, healthcare), applications (e.g., model debugging, actionable recourse), data modalities (e.g., tabular data, images), models (e.g., transformers, convolutional neural networks) and stakeholders (e.g., end-users, regulators, data scientists).

The goal of this tutorial is to present a comprehensive overview of the XAI field with hands-on examples. This repository introduces state-of-the-art explanation methods that can be used for different data modalities as well as potential pitfalls, e.g., lack of robustness of explanations.

## Setup - local

Execute the following commands to set up the Jupyter environment; if working on Windows, make sure [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) is up-to-date before executing the below. The setup instructions assume that the repository was already cloned with e.g., ```git clone https://github.com/aws-samples/explainability-methods-pitfalls.git```.

```
conda create -n kdd python=3.11
conda activate kdd
pip install -r requirements.txt
python -m ipykernel install --user --name=kdd
```

Troubleshooting if installation fails, try:

```
pip install -U wheels
pip install -U setuptools
pip install -r requirements-freeze.txt
```

Troubleshooting for `FileNotFoundError`: Make sure you are inside the repository; if not use `cd explainability-methods-pitfalls`.

## Setup - provisioned

Instead of installing the environment locally, you can use the links below to open the notebooks using a pre-configured environment.

For this, you will need to provide an email and wait for SageMaker Studio Lab to approve the account. Account availability varies, so instant access cannot be guaranteed. After launching SageMaker Studio Lab the installation will take approx. 5-10'.

| Notebook Name | Studio lab |
| :---: | ---: |
| Explainability (Tabular Data)| [![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/aws-samples/explainability-methods-pitfalls/blob/main/Explainability_TabularData.ipynb)|
| Explainability (Image Data)| [![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/aws-samples/explainability-methods-pitfalls/blob/main/Explainability_ImageData.ipynb)|
| Explainability (Text Data)| [![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/aws-samples/explainability-methods-pitfalls/blob/main/Explainability_TextData.ipynb)|

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
