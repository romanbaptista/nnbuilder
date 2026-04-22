# NNBuilder: A Simple Neural Network Builder for PyTorch


# Overview

NNBuilder is a lightweight, chainable utility class built on top of PyTorch’s `nn.Module`, designed to simplify the construction of sequential, fully‑connected neural networks.

It provides a builder‑style API that allows users to incrementally define a neural network architecture without manually writing repetitive constructor and forward‑pass code for each model.
This is particularly useful for:

- Learning and teaching PyTorch
- Rapid prototyping of multilayer perceptrons (MLPs)
- Tabular and vector‑based classification problems
- Reducing boilerplate while preserving full PyTorch control


# Class Description

NNBuilder encapsulates a feed‑forward neural network composed of:

- Fully connected (`nn.Linear`) layers
- Activation functions (added automatically or explicitly)
- Optional dropout regularization

Layers are stored internally in an `nn.ModuleList` and executed sequentially during the forward pass.


# Builder Functions

### *add_dense(in_features: int, out_features: int, activation: str = None)*
Adds a fully connected (dense) layer to the model and optionally appends an activation function.

**Parameters**

- `in_features` – Number of input features
- `out_features` – Number of output features
- `activation` – Activation function identifier (`'relu'` or `'sigmoid'`)

If `activation` is provided, the corresponding activation layer is automatically appended immediately after the dense layer.

**Returns**

- `self` (enables method chaining)


### *add_activation(activation: str)*
Adds an activation layer explicitly.

This allows activation functions to be inserted independently of dense layers, enabling more flexible architectural definitions while still supporting automatic activation addition via `add_dense()`.

**Parameters**

- `activation` – Activation function identifier (`'relu'` or `'sigmoid'`)

**Returns**

- `self`


### *add_dropout(p: float = 0.5)*
Adds a dropout layer using `torch.nn.Dropout`.

**Parameters**

- `p` – Dropout probability

**Returns**

- `self`


### *forward(x)*
Defines the forward pass by applying each stored layer in sequence to the input tensor.


# Using NNBuilder in Your Own Code
The NNBuilder class is defined in a standalone Python file named `nnbuilder.py`. To use it in your own scripts or notebooks, ensure that this file is accessible on Python’s import path.

## File Placement
The simplest and most common setup is to place `nnbuilder.py` in the same directory as the Python script (`.py`) or Jupyter notebook (`.ipynb`) where you want to define and train your model.

For example:
```text
project_directory/
├── nnbuilder.py
├── train_model.py
└── experiment.ipynb
```
In this arrangement, Python will automatically be able to locate and import NNBuilder.

## Importing NNBuilder

Once `nnbuilder.py` is in the same directory, you can import the class as follows:
```python
from nnbuilder import NNBuilder
```

# Example: Binary Classification with a Sklearn Dataset
The following example demonstrates how to:
- Load a real‑world dataset from scikit‑learn
- Preprocess the data
- Build a neural network using NNBuilder

## Load and process data
```python
from nnbuilder import NNBuilder

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

# Load dataset (binary classification)
X, y = load_breast_cancer(return_X_y=True)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# Standardize features (fit on training data only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert features to PyTorch tensors
X_train_scaled_tensor = torch.from_numpy(X_train_scaled).float()
X_test_scaled_tensor = torch.from_numpy(X_test_scaled).float()

# Convert labels to float tensors and add channel dimension
y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1)
y_test_tensor = torch.from_numpy(y_test).float().unsqueeze(1)

# Inspect training data dimensions
num_samples = X_train_scaled_tensor.shape[0]
num_features = X_train_scaled_tensor.shape[1]

print(f"Number of samples: {num_samples}")   # e.g. 455
print(f"Number of features: {num_features}") # e.g. 30
```

## Defining the model with NNBuilder

Activations can be added automatically via `add_dense()`:
```python
my_model = (
    NNBuilder()
    .add_dense(in_features=num_features, out_features=64, activation='relu')
    .add_dense(in_features=64, out_features=32, activation='relu')
    .add_dropout(p=0.5)
    .add_dense(in_features=32, out_features=1, activation='sigmoid')
)
```

Or explicitly using `add_activation()`:
```python
my_model = (
    NNBuilder()
    .add_dense(num_features, 64)
    .add_activation('relu')
    .add_dense(64, 32)
    .add_activation('relu')
    .add_dropout(0.5)
    .add_dense(32, 1)
    .add_activation('sigmoid')
)
```

## Comparison with standard PyTorch implementation

Below is the equivalent model definition written using traditional PyTorch class boilerplate:
```python
class CustomNN(nn.Module):

    def __init__(self):
        super(CustomNN, self).__init__()

        self.fc1 = nn.Linear(in_features=num_features, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

my_model = CustomNN()
```

NNBuilder trades some explicit verbosity for a clearer, higher‑level declaration of network architecture, while remaining fully compatible with idiomatic PyTorch workflows.

## Integrate to generic training loop

```python
# Define loss function
loss_function = nn.BCELoss()
# Define optimiser
optimizer = optim.Adam(my_model.parameters(), lr = 0.001)

# Define variables
epochs = 20

# Iterate through epochs
for epoch in range(epochs):
    
    # Enter training mode
    my_model.train()
    # Initialise tracked loss value
    running_loss = 0.0

    # Iterate through DataLoader batches
    for x_batch, y_batch in train_loader:

        # Reset gradient calculations
        optimizer.zero_grad()
        # Predict labels
        predictions = my_model(x_batch)
        # Calculate loss against actual values
        loss = loss_function(predictions, y_batch)
        
        # Backpropagate loss
        loss.backward()
        # Take step towards (local) minimum of loss function using optimizer
        optimizer.step()

        # Track loss
        running_loss += loss.item()
    
    # Print summary
    print(f'Epoch {epoch + 1}: Loss was {running_loss / len(train_loader)}')
``
```

## Evaluate Model

```python
# Use non-gradient calculation mode
with torch.no_grad():
    
    # Enter evaluation mode
    my_model.eval()

    # Predict labels on test_data
    predictions = my_model(X_test_scaled_tensor)
    # Calculate loss against actual values
    loss = loss_function(predictions, y_test_tensor).item()

    # Calculate accuracy score if predictions of >= 0.5 are taken as 1
    accuracy = ((predictions >= 0.5) == y_test_tensor).float().mean().item()

print(f'Model accuracy: {accuracy:.3f}')
```

# Summary
NNBuilder provides a compact, readable abstraction for defining sequential neural networks in PyTorch. It intentionally separates architecture definition from training logic, enabling clean experimentation and easy comparison with traditional PyTorch implementations.

The addition of explicit activation layers allows greater architectural flexibility while preserving the simplicity of automatic activation handling for common use cases.