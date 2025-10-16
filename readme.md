# Neural Network from Scratch: Breast Cancer Classification

## Project Overview

This project implements a neural network from scratch using only Python and NumPy to classify breast cancer tumors as malignant or benign. The implementation is built without using any machine learning libraries, allowing for a deep understanding of how neural networks function at a fundamental level.

## Table of Contents

1. [Dataset and Feature Selection](#dataset-and-feature-selection)
2. [Neural Network Architecture](#neural-network-architecture)
3. [Implementation Details](#implementation-details)
4. [Training Process](#training-process)
5. [Results and Evaluation](#results-and-evaluation)
6. [How to Run](#how-to-run)

## Dataset and Feature Selection

### Dataset Information

We used the Breast Cancer Wisconsin (Diagnostic) Dataset, which contains 569 samples of tumor measurements. Each sample is labeled as either malignant (M) or benign (B). The dataset includes 30 numerical features computed from digitized images of fine needle aspirate (FNA) of breast masses.

### Feature Selection Process

Since our neural network is constrained to use only 2 input features, we performed a correlation analysis to identify the features most strongly associated with the diagnosis. Our approach involved:

1. **Calculating Correlations**: We computed the absolute correlation coefficient between each feature and the target variable (malignant or benign diagnosis).

2. **Ranking Features**: Features were ranked by their correlation strength with the target.

3. **Selection**: We selected the top 2 features with the highest correlation values.

### Selected Features

Based on our correlation analysis, we chose:

- **perimeter3**
- **concave_points3**

These features showed the highest correlation with malignancy. The perimeter gives us information about the overall size of the tumor, while concave points indicate irregularity in the tumor boundary. Malignant tumors tend to have larger perimeters and more irregular boundaries with multiple concave portions, making these features particularly discriminative.

## Neural Network Architecture

### Layer Structure

Our neural network consists of three layers:

1. **Input Layer**: 2 neurons (one for each selected feature)
2. **Hidden Layer**: 3 neurons with ReLU activation
3. **Output Layer**: 1 neuron with Sigmoid activation

### Why This Architecture?

**Input Layer**: The two neurons directly correspond to our two selected features (perimeter3 and concave_points3). Each neuron receives the normalized value of its respective feature.

**Hidden Layer**: We chose 3 neurons for the hidden layer as a balance between model capacity and simplicity. With too few neurons (like 2), the network might not capture the complexity of the decision boundary. With too many neurons (like 5 or more), we risk overfitting on our relatively small dataset. Three neurons provide enough representational power to learn non-linear patterns while remaining computationally efficient.

**Output Layer**: A single neuron is sufficient for binary classification. The output value ranges from 0 to 1, representing the probability that a tumor is malignant.

### How Neurons Are Modeled

Each neuron performs two operations:

1. **Weighted Sum**: The neuron computes a weighted sum of its inputs plus a bias term.

   ```
   z = (w1 * x1) + (w2 * x2) + ... + (wn * xn) + b
   ```

   where w values are weights, x values are inputs, and b is the bias.

2. **Activation**: The weighted sum passes through an activation function to introduce non-linearity.
   ```
   a = activation_function(z)
   ```

This two-step process happens at every neuron in every layer. The weights and biases are the learnable parameters that the network adjusts during training.

### Layer Connections

**Input to Hidden Layer**: Each of the 2 input neurons connects to all 3 hidden neurons, creating 6 weighted connections. Each hidden neuron also has its own bias term, adding 3 more parameters. This gives us 9 trainable parameters in this layer.

**Hidden to Output Layer**: Each of the 3 hidden neurons connects to the single output neuron, creating 3 weighted connections plus 1 bias term. This adds 4 more trainable parameters.

In total, our network has 13 trainable parameters (6 + 3 + 3 + 1 = 13).

## Implementation Details

### Weight Initialization

We use He initialization for our weights, which is particularly effective for networks using ReLU activation. The weights are drawn from a normal distribution and scaled by the square root of 2 divided by the number of input neurons:

```
W = random_normal * sqrt(2 / n_inputs)
```

This initialization helps prevent vanishing or exploding gradients during training. Biases are initialized to zero.

### Activation Functions

#### ReLU (Hidden Layer)

We chose the Rectified Linear Unit (ReLU) activation function for the hidden layer. ReLU is defined as:

```
ReLU(z) = max(0, z)
```

**Why ReLU?**

1. **Computational Efficiency**: ReLU is extremely fast to compute compared to sigmoid or tanh, requiring only a simple threshold operation.

2. **Solves Vanishing Gradient Problem**: Unlike sigmoid or tanh, ReLU does not saturate for positive values. This means gradients flow more effectively during backpropagation, leading to faster training.

3. **Sparse Activation**: ReLU naturally produces sparse representations (many neurons output zero), which can lead to more efficient and interpretable models.

4. **Better Performance**: In practice, ReLU often leads to better performance on classification tasks compared to traditional activation functions.

The derivative of ReLU is straightforward:

```
ReLU'(z) = 1 if z > 0, else 0
```

#### Sigmoid (Output Layer)

For the output layer, we use the Sigmoid activation function:

```
Sigmoid(z) = 1 / (1 + e^(-z))
```

**Why Sigmoid for Output?**

1. **Probability Interpretation**: Sigmoid squashes the output to the range [0, 1], which can be interpreted as the probability of the tumor being malignant.

2. **Binary Classification**: Sigmoid is the natural choice for binary classification problems. An output greater than 0.5 predicts malignant, while less than 0.5 predicts benign.

3. **Smooth Gradients**: Unlike ReLU, sigmoid provides smooth gradients everywhere, which is beneficial for the final prediction layer.

The derivative of sigmoid is:

```
Sigmoid'(a) = a * (1 - a)
```

### Forward Propagation

Forward propagation is the process of passing input data through the network to generate predictions. Here is how it works step by step:

**Step 1: Input to Hidden Layer**

```
Z1 = X * W1 + b1
A1 = ReLU(Z1)
```

- We multiply the input matrix X by the weights W1 and add bias b1
- Apply ReLU activation to get the hidden layer activations A1

**Step 2: Hidden to Output Layer**

```
Z2 = A1 * W2 + b2
A2 = Sigmoid(Z2)
```

- Multiply hidden activations A1 by weights W2 and add bias b2
- Apply Sigmoid activation to get the final prediction A2

The final output A2 represents the predicted probability of malignancy.

### Loss Function

We use Mean Squared Error (MSE) as our loss function:

```
MSE = (1 / 2m) * sum((y_predicted - y_actual)^2)
```

where m is the number of training samples.

**How MSE is Calculated:**

1. For each training example, we compute the difference between the predicted value and the actual label (0 or 1)
2. We square this difference to penalize larger errors more heavily and ensure all errors are positive
3. We sum all squared errors across all training examples
4. We divide by 2m to get the average error (the factor of 2 is for mathematical convenience during differentiation)

**Why MSE?**

While binary cross-entropy is more commonly used for classification, MSE is perfectly valid and offers some advantages for educational purposes:

1. **Simpler Mathematics**: The derivative of MSE is straightforward, making it easier to understand backpropagation
2. **Clear Interpretation**: The loss directly represents how far predictions are from actual values
3. **Smooth Gradients**: MSE provides smooth, continuous gradients throughout the training process

### Backward Propagation

Backpropagation computes the gradients of the loss function with respect to all weights and biases. These gradients tell us how to adjust each parameter to reduce the loss.

**Step 1: Output Layer Gradients**

```
dZ2 = (A2 - y_true) * Sigmoid'(A2)
dW2 = (1/m) * A1^T * dZ2
db2 = (1/m) * sum(dZ2)
```

We start from the output and compute how much each weight and bias contributed to the error.

**Step 2: Hidden Layer Gradients**

```
dA1 = dZ2 * W2^T
dZ1 = dA1 * ReLU'(Z1)
dW1 = (1/m) * X^T * dZ1
db1 = (1/m) * sum(dZ1)
```

We propagate the error backward through the network, accounting for the ReLU activation in the hidden layer.

### Parameter Updates

Once we have all gradients, we update the weights and biases using gradient descent:

```
W = W - learning_rate * dW
b = b - learning_rate * db
```

The learning rate controls how large of a step we take in the direction of the negative gradient. We use a learning rate of 0.5 in our implementation, which provides stable and relatively fast convergence.

## Training Process

### Data Preprocessing

Before training, we perform two critical preprocessing steps:

1. **Normalization**: We standardize our features to have mean 0 and standard deviation 1. This is crucial because:

   - Features have different scales (perimeter is much larger than concave points)
   - Neural networks train more effectively when inputs are normalized
   - It prevents features with larger magnitudes from dominating the learning process

2. **Train-Test Split**: We split the data into 80% training and 20% testing sets. The training set is used to learn the parameters, while the test set evaluates how well the model generalizes to unseen data.

### Training Loop

The training process runs for 1000 iterations. In each iteration:

1. Forward propagation computes predictions for all training samples
2. Loss is calculated using MSE
3. Backward propagation computes gradients
4. Parameters are updated using gradient descent
5. Every 100 iterations, we print the current loss and training accuracy

This iterative process gradually adjusts the weights and biases to minimize the loss function, improving the model's predictions.

## Results and Evaluation

### Performance Metrics

We evaluate our model using several metrics:

1. **Training Accuracy**: Measures how well the model fits the training data
2. **Test Accuracy**: Measures how well the model generalizes to unseen data
3. **Confusion Matrix**: Shows the breakdown of true positives, true negatives, false positives, and false negatives

### Visualizations

Our implementation generates three key visualizations:

1. **Training Loss Plot**: Shows how the loss decreases over iterations, demonstrating that the model is learning

2. **Decision Boundary**: Visualizes the learned decision boundary in the 2D feature space, showing how the model separates malignant from benign tumors

3. **Test Set Performance**: Shows the decision boundary with test data points, highlighting any misclassifications

## How to Run

### Requirements

```
Python 3.x
NumPy
Pandas
Matplotlib
scikit-learn (only for loading the dataset)
```

### Installation

```bash
pip install numpy pandas matplotlib scikit-learn
```

### Running the Notebook

1. Clone this repository
2. Place your dataset file in the same directory (or let it download automatically)
3. Open the Jupyter notebook: `neural_network_from_scratch.ipynb`
4. Run all cells sequentially

### Experimentation

You can modify the following parameters to experiment with different configurations:

- `hidden_size`: Number of neurons in the hidden layer (try 2, 3, or 4)
- `activation`: Activation function for hidden layer ('relu', 'sigmoid', or 'tanh')
- `learning_rate`: Step size for gradient descent (try 0.01, 0.1, 0.5, or 1.0)
- `iterations`: Number of training iterations (try 500, 1000, or 1500)

## Technical Notes

### Why Build from Scratch?

Building a neural network from scratch provides deep insights into:

- How data flows through the network
- How gradients are computed and propagated
- How learning actually happens through parameter updates
- The mathematical foundations of deep learning

These limitations are intentional for educational purposes, keeping the implementation clear and understandable.

## Conclusion

This project demonstrates the fundamental principles of neural networks through a practical implementation. By building each component from scratch, we gain a thorough understanding of forward propagation, activation functions, loss computation, backpropagation, and parameter optimization. The choice of ReLU activation in the hidden layer provides efficient training, while the careful selection of perimeter3 and concave_points3 as input features gives our simple network the discriminative power needed for effective breast cancer classification.

## Authors

Ian Gabriel Paulmino
Ibrahim
Corpuz

## References

- Breast Cancer Wisconsin (Diagnostic) Dataset: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- UCI Machine Learning Repository
- scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
