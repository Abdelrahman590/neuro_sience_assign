# Neural Network Forward Propagation

## Overview

This project implements a **feedforward neural network** with forward propagation to compute outputs from given inputs. The network uses the hyperbolic tangent (tanh) activation function and consists of one hidden layer and one output layer.

## Network Architecture

### Structure

- **Input Layer**: 2 neurons (i₁, i₂)
- **Hidden Layer**: 2 neurons (h₁, h₂)
- **Output Layer**: 2 neurons (o₁, o₂)

### Diagram

```
    i1 ──┐
         ├─→ [Hidden Layer] ──→ [Output Layer] ──→ o1
    i2 ──┘                                    └──→ o2
```

## Mathematical Formulation

### Initialization

#### Inputs

$$i_1 = 0.05, \quad i_2 = 0.10$$

#### Biases

$$b_1 = 0.5 \quad \text{(hidden layer bias)}$$
$$b_2 = 0.7 \quad \text{(output layer bias)}$$

#### Weights

Random weights are initialized from the uniform distribution $\mathcal{U}(-0.5, 0.5)$:
$$w_1, w_2, \ldots, w_8 \sim \mathcal{U}(-0.5, 0.5)$$

where:

- $w_1, w_2$ connect inputs to $h_1$
- $w_3, w_4$ connect inputs to $h_2$
- $w_5, w_6$ connect hidden to $o_1$
- $w_7, w_8$ connect hidden to $o_2$

### Activation Function

The **hyperbolic tangent (tanh)** function is used:
$$\text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{\sinh(x)}{\cosh(x)}$$

Properties:

- Range: $(-1, 1)$
- Differentiable everywhere
- Zero-centered output

### Hidden Layer Computation

The hidden layer neurons compute weighted sums followed by activation:

$$\text{net}_{h_1} = w_1 \cdot i_1 + w_2 \cdot i_2 + b_1$$
$$h_1 = \tanh(\text{net}_{h_1})$$

$$\text{net}_{h_2} = w_3 \cdot i_1 + w_4 \cdot i_2 + b_1$$
$$h_2 = \tanh(\text{net}_{h_2})$$

### Output Layer Computation

The output layer neurons use the hidden layer outputs as inputs:

$$\text{net}_{o_1} = w_5 \cdot h_1 + w_6 \cdot h_2 + b_2$$
$$o_1 = \tanh(\text{net}_{o_1})$$

$$\text{net}_{o_2} = w_7 \cdot h_1 + w_8 \cdot h_2 + b_2$$
$$o_2 = \tanh(\text{net}_{o_2})$$

## Forward Propagation Algorithm

1. **Compute hidden layer net input**: Apply weights and bias to input layer
2. **Apply activation**: Pass through tanh function to get hidden neuron outputs
3. **Compute output layer net input**: Apply weights and bias to hidden layer outputs
4. **Apply activation**: Pass through tanh function to get final network outputs
5. **Return outputs**: $o_1$ and $o_2$ are the network predictions

## Code Structure

### Dependencies

```python
import numpy as np
```

### Key Components

| Component       | Purpose                                |
| --------------- | -------------------------------------- |
| Input Variables | i1, i2 - network inputs                |
| Bias Terms      | b1, b2 - learnable bias parameters     |
| Weights         | w1-w8 - learnable weight parameters    |
| tanh()          | Activation function                    |
| net_h1, net_h2  | Pre-activation outputs (hidden layer)  |
| h1, h2          | Post-activation outputs (hidden layer) |
| net_o1, net_o2  | Pre-activation outputs (output layer)  |
| o1, o2          | Final network outputs                  |

## Usage

Run the script directly:

```bash
python forwarde.py
```

### Output

The script prints the final network outputs:

```
Output o1 = [value]
Output o2 = [value]
```

Both outputs will be in the range $(-1, 1)$ due to the tanh activation function.

## Parameters

### Configurable Inputs

Modify the input values:

```python
i1 = 0.05  # Input 1
i2 = 0.10  # Input 2
```

### Configurable Biases

```python
b1 = 0.5   # Hidden layer bias
b2 = 0.7   # Output layer bias
```

### Random Seed

For reproducible weight initialization:

```python
np.random.seed(42)  # Change seed for different initial weights
```

## Key Features

✓ **Simple & Interpretable**: Easy-to-understand feed-forward network  
✓ **Reproducible**: Configurable random seed for consistent results  
✓ **Vector Operations**: Uses NumPy for efficient computation  
✓ **Tanh Activation**: Non-linear activation suitable for bounded outputs

## Applications

This network can be used for:

- **Classification**: Binary or multi-class classification tasks
- **Regression**: Function approximation with bounded outputs
- **Pattern Recognition**: Learning non-linear relationships
- **Educational Purpose**: Understanding neural network fundamentals

## Mathematical Derivatives

For training purposes, the gradient of tanh is:
$$\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x) = 1 - h^2$$

where $h = \tanh(x)$.

## Future Enhancements

- [ ] Implement backpropagation for training
- [ ] Add loss function computation (MSE)
- [ ] Add batch processing capability
- [ ] Implement different activation functions (ReLU, Sigmoid)
- [ ] Add weight regularization (L1/L2)
- [ ] Visualize network architecture and outputs

## Author

Neural Network Implementation - Neuro Assignment

## License

Educational Use

---

**Last Updated**: February 2026
