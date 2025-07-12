# protein-mlutils
![Python package](https://img.shields.io/badge/python-3.12%20-blue.svg)
![PyPI](https://img.shields.io/pypi/v/protein_mlutils)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A Python package for ML utility functions



## Local installation
If you would like to convert your Python code into a Python package that can be used widely, then follow the instructions below.
After going to the folder that contains `pyproject.toml`, use the following commands:

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

## Features
- Data loading and preprocessing
- Model training and evaluation
- Prediction utilities
- Utility functions for common ML tasks 

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.