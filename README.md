# pymlutils
A Python package for ML utility functions


## Installation
You can install the package using pip:
```bash
pip install pymlutils
```     

## Usage
```python
import pymlutils as pml

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