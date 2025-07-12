# protein-mlutils

<p align="center">
  <img src="https://img.shields.io/github/last-commit/mohitpandey92/protein-mlutils" alt="GitHub last commit">
  <img src="https://img.shields.io/github/languages/code-size/mohitpandey92/protein-mlutils" alt="GitHub code size in bytes">
  <img src="https://img.shields.io/github/languages/top/mohitpandey92/protein-mlutils" alt="GitHub top language">
  <img src="https://img.shields.io/badge/python-3.12%20-blue.svg" alt="Python package">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
</p>
<p align="center">
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit_learn">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Lightning-792DE4?style=for-the-badge&logo=lightning&logoColor=white" alt="Lightning">
  <img src="https://img.shields.io/badge/-HuggingFace-FDEE21?style=for-the-badge&logo=HuggingFace&logoColor=black" alt="HuggingFace">
</p>
A Python package for ML utility functions specifically designed for protein-related machine learning tasks. This package provides a set of tools to facilitate data loading, model training, evaluation, and prediction tasks in the field of protein machine learning.


## Features
- Protein Sequence preprocessing:
  - One-hot encoding
  - Embedding generation for a fixed dimension
- Model training and evaluation:
  - TODO: Support for various model types (e.g., Random Forest, XGBoost, PyTorch)
  - TODO: Cross-validation and hyperparameter tuning
  - TODO: Implement functionality to save trained models and their performance metrics

## Local installation
To install the package locally, you can use the `build` tool to create a source distribution and a wheel distribution. First, ensure you have the `build` package installed:
```pip install build```

Then, navigate to the root directory of your project (where `pyproject.toml` is located) and run the following commands:
```
python -m build
python -m pip install dist/*.gz
```

If you would like to reinstall your package, then type:
`python -m pip install --upgrade dist/*.gz`


Note that some folks prefer specifying the package details in `setup.py`, but I recommend using `build` which I found
<a href="https://packaging.python.org/en/latest/discussions/setup-py-deprecated/">easier to use</a>.

<blockquote>
One recommended, simple, and straightforward method of building source distributions and wheels is to use the build tool with a command like python -m build which triggers the generation of both distribution formats
</blockquote>

We are using `pyproject.toml` file, which provides instructions to `build` for how to build our Python package.


Once the package has been installed, this should work anywhere in the system:

## Usage
```python
import protein_mlutils as pml

# Example usage
data = pml.load_data("data.csv")
model = pml.train_model(data, model_type="random_forest")
predictions = pml.make_predictions(model, data)
pml.evaluate_model(model, data)
``` 

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.