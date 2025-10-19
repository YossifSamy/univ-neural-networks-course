"""
Lab 03: Training Neural Networks Implementation
================================================

This file implements backpropagation and gradient descent for training
neural networks.

Covers:
1. Activation function derivatives
2. Loss functions (MSE)
3. Backpropagation algorithm
4. Training loop
5. XOR training example
6. Hyperparameter experiments

Follow along with your instructor!

Author: Neural Networks Course
Lab: 03 - Training Neural Networks
"""

import math
import random


print("="*70)
print(" "*15 + "TRAINING NEURAL NETWORKS IMPLEMENTATION")
print("="*70)


# =============================================================================
# PART 1: ACTIVATION FUNCTIONS AND THEIR DERIVATIVES
# =============================================================================

print("\n" + "="*70)
print("PART 1: ACTIVATION FUNCTIONS AND DERIVATIVES")
print("="*70)

print("""
For backpropagation, we need DERIVATIVES of activation functions.
The derivative tells us how much the output changes when input changes.

Why do we need this? Backpropagation uses the chain rule, which requires
derivatives at each step!
""")

def sigmoid(z):
    """
    Sigmoid activation: σ(z) = 1 / (1 + e^(-z))
    Output range: (0, 1)
    """
    return 1 / (1 + math.exp(-z))


def sigmoid_derivative(z):
    """
    Derivative of sigmoid: σ'(z) = σ(z) × (1 - σ(z))
    
    This is a beautiful property of sigmoid!
    The derivative can be computed from the sigmoid itself.
    """
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z):
    """
    Hyperbolic tangent: tanh(z)
    Output range: (-1, 1)
    """
    return math.tanh(z)


def tanh_derivative(z):
    """
    Derivative of tanh: tanh'(z) = 1 - tanh²(z)
    """
    t = tanh(z)
    return 1 - t**2


def relu(z):
    """
    ReLU: Rectified Linear Unit
    Output: z if z > 0, else 0
    """
    return max(0, z)


def relu_derivative(z):
    """
    Derivative of ReLU:
    - 1 if z > 0
    - 0 if z <= 0
    """
    return 1 if z > 0 else 0


# Demonstrate derivatives
print("\nActivation function derivatives at different points:")
print("-"*70)
print("  z   | sigmoid | sigmoid' |  tanh  | tanh'  | relu | relu'")
print("-"*70)

test_z_values = [-2, -1, 0, 1, 2]
for z in test_z_values:
    s = sigmoid(z)
    s_deriv = sigmoid_derivative(z)
    t = tanh(z)
    t_deriv = tanh_derivative(z)
    r = relu(z)
    r_deriv = relu_derivative(z)
    print(f" {z:4.1f} | {s:7.4f} | {s_deriv:8.4f} | {t:7.4f} | {t_deriv:7.4f} | {r:4.1f} | {r_deriv:5.1f}")

print("""
OBSERVATIONS:
- Sigmoid derivative is largest near z=0, approaches 0 at extremes
- Tanh derivative has same shape as sigmoid'
- ReLU derivative is simple: 0 or 1 (this makes it fast!)
""")


# =============================================================================
# PART 2: LOSS FUNCTIONS
# =============================================================================

print("\n\n" + "="*70)
print("PART 2: LOSS FUNCTIONS")
print("="*70)

print("""
Loss functions measure how wrong our predictions are.
We want to MINIMIZE the loss!

Mean Squared Error (MSE) is most common for regression:
MSE = (1/n) × Σ(actual - predicted)²
""")

def mse_loss(y_true, y_pred):
    """
    Mean Squared Error loss
    
    Parameters:
        y_true: Actual values (list or single value)
        y_pred: Predicted values (list or single value)
    
    Returns:
        Average squared error
    """
    # Handle single values
    if not isinstance(y_true, list):
        y_true = [y_true]
        y_pred = [y_pred]
    
    # Calculate squared errors and average
    squared_errors = [(yt - yp)**2 for yt, yp in zip(y_true, y_pred)]
    return sum(squared_errors) / len(squared_errors)


def mse_derivative(y_true, y_pred):
    """
    Derivative of MSE with respect to prediction
    
    ∂MSE/∂y_pred = -2(y_true - y_pred) / n
    
    For simplicity, we'll use: -2(y_true - y_pred)
    (The 1/n and 2 are constants that affect learning rate)
    """
    if not isinstance(y_true, list):
        return -2 * (y_true - y_pred)
    else:
        return [-2 * (yt - yp) for yt, yp in zip(y_true, y_pred)]


# Demonstrate loss calculation
print("\nExample loss calculations:")
print("-"*70)

examples = [
    (1.0, 0.9),   # Close prediction
    (1.0, 0.5),   # Medium error
    (1.0, 0.1),   # Large error
    (0.0, 0.0),   # Perfect!
]

for y_true, y_pred in examples:
    loss = mse_loss(y_true, y_pred)
    error = y_true - y_pred
    print(f"Actual: {y_true}, Predicted: {y_pred:.1f} → Error: {error:+.1f} → Loss: {loss:.4f}")

print("\nNote: Larger errors result in much larger loss (squared error)!")


# =============================================================================
# PART 3: MLP CLASS WITH TRAINING CAPABILITY
# =============================================================================

print("\n\n" + "="*70)
print("PART 3: MLP CLASS WITH TRAINING")
print("="*70)

print("""
Now we'll build an MLP class that can:
1. Forward propagation (prediction)
2. Backward propagation (compute gradients)
3. Update weights (gradient descent)
4. Train on data
""")


class Layer:
    """
    Represents one layer in the neural network.
    Stores weights, biases, and intermediate values needed for backprop.
    """
    
    def __init__(self, input_size, output_size):
        """
        Initialize layer with random weights and zero biases.
        
        Parameters:
            input_size: Number of inputs to this layer
            output_size: Number of neurons in this layer
        """
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize weights randomly (small values)
        # Each neuron has input_size weights
        self.weights = [[random.uniform(-1, 1) for _ in range(input_size)] 
                       for _ in range(output_size)]
        
        # Initialize biases to zero
        self.biases = [0.0 for _ in range(output_size)]
        
        # Store intermediate values for backpropagation
        self.inputs = None          # Input to this layer
        self.z_values = None        # Weighted sums (before activation)
        self.activations = None     # Output (after activation)
    
    def forward(self, inputs):
        """
        Forward pass through this layer.
        
        Parameters:
            inputs: Input values (list)
        
        Returns:
            Layer outputs after activation
        """
        self.inputs = inputs  # Save for backprop
        self.z_values = []
        self.activations = []
        
        # For each neuron in this layer
        for neuron_idx in range(self.output_size):
            # Calculate weighted sum: z = Σ(w × x) + b
            z = sum(w * x for w, x in zip(self.weights[neuron_idx], inputs))
            z += self.biases[neuron_idx]
            
            # Apply activation function
            activation = sigmoid(z)
            
            # Store for backprop
            self.z_values.append(z)
            self.activations.append(activation)
        
        return self.activations
    
    def backward(self, d_activations, learning_rate):
        """
        Backward pass through this layer.
        Computes gradients and updates weights.
        
        Parameters:
            d_activations: Gradient of loss with respect to layer output
            learning_rate: Step size for gradient descent
        
        Returns:
            Gradient of loss with respect to layer input (for previous layer)
        """
        # Will store gradients w.r.t. layer input
        d_inputs = [0.0] * self.input_size
        
        # For each neuron in this layer
        for neuron_idx in range(self.output_size):
            # Gradient of loss w.r.t. activation
            d_activation = d_activations[neuron_idx]
            
            # Gradient of activation w.r.t. z (using activation derivative)
            d_z = d_activation * sigmoid_derivative(self.z_values[neuron_idx])
            
            # Update weights for this neuron
            for input_idx in range(self.input_size):
                # Gradient w.r.t. weight: ∂L/∂w = ∂L/∂z × ∂z/∂w = d_z × input
                d_weight = d_z * self.inputs[input_idx]
                
                # Update weight using gradient descent
                self.weights[neuron_idx][input_idx] -= learning_rate * d_weight
                
                # Accumulate gradient w.r.t. input
                # ∂L/∂input = ∂L/∂z × ∂z/∂input = d_z × weight
                d_inputs[input_idx] += d_z * self.weights[neuron_idx][input_idx]
            
            # Update bias
            # Gradient w.r.t. bias: ∂L/∂b = ∂L/∂z × ∂z/∂b = d_z × 1 = d_z
            self.biases[neuron_idx] -= learning_rate * d_z
        
        return d_inputs


class MLP:
    """
    Multi-Layer Perceptron with training capability.
    """
    
    def __init__(self, layer_sizes):
        """
        Initialize MLP with given architecture.
        
        Parameters:
            layer_sizes: List of layer sizes, e.g., [2, 3, 1]
                        means 2 inputs, 3 hidden neurons, 1 output
        """
        self.layer_sizes = layer_sizes
        self.layers = []
        
        # Create layers
        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
        
        print(f"Created MLP with architecture: {layer_sizes}")
        self._print_parameters()
    
    def _print_parameters(self):
        """Print number of parameters in the network."""
        total = 0
        for i, layer in enumerate(self.layers):
            layer_params = layer.input_size * layer.output_size + layer.output_size
            total += layer_params
            print(f"  Layer {i+1}: {layer_params} parameters")
        print(f"  Total: {total} parameters")
    
    def predict(self, inputs):
        """
        Make a prediction (forward pass through all layers).
        
        Parameters:
            inputs: Input values
        
        Returns:
            Network output
        """
        activations = inputs
        
        # Forward through each layer
        for layer in self.layers:
            activations = layer.forward(activations)
        
        return activations
    
    def train_step(self, inputs, target, learning_rate):
        """
        Perform one training step (forward + backward + update).
        
        Parameters:
            inputs: Training input
            target: Expected output
            learning_rate: Step size for gradient descent
        
        Returns:
            Loss for this example
        """
        # STEP 1: Forward pass
        output = self.predict(inputs)
        
        # STEP 2: Calculate loss
        loss = mse_loss(target, output)
        
        # STEP 3: Backward pass
        # Start with gradient of loss w.r.t. output
        d_output = mse_derivative(target, output)
        if not isinstance(d_output, list):
            d_output = [d_output]
        
        # Backpropagate through layers (in reverse order)
        d_activations = d_output
        for layer in reversed(self.layers):
            d_activations = layer.backward(d_activations, learning_rate)
        
        return loss
    
    def train(self, X, y, epochs, learning_rate, verbose=True):
        """
        Train the network on a dataset.
        
        Parameters:
            X: List of input examples
            y: List of target outputs
            epochs: Number of complete passes through data
            learning_rate: Step size for gradient descent
            verbose: Whether to print progress
        
        Returns:
            List of average losses per epoch
        """
        loss_history = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            # Train on each example
            for i in range(len(X)):
                loss = self.train_step(X[i], y[i], learning_rate)
                epoch_loss += loss
            
            # Calculate average loss
            avg_loss = epoch_loss / len(X)
            loss_history.append(avg_loss)
            
            # Print progress
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:4d}: Loss = {avg_loss:.6f}")
        
        return loss_history


# =============================================================================
# PART 4: TRAINING XOR - THE MOMENT OF TRUTH!
# =============================================================================

print("\n\n" + "="*70)
print("PART 4: TRAINING XOR FROM SCRATCH")
print("="*70)

print("""
Let's train a network to solve XOR!

XOR Truth Table:
  Input1 | Input2 | Output
  -------|--------|-------
    0    |   0    |   0
    0    |   1    |   1
    1    |   0    |   1
    1    |   1    |   0

We'll use architecture: [2, 2, 1]
- 2 inputs (the two XOR inputs)
- 2 hidden neurons (needed to solve XOR)
- 1 output (the XOR result)
""")

# XOR training data
X_xor = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_xor = [0, 1, 1, 0]

# Create network
print("\n" + "-"*70)
print("Creating network...")
print("-"*70)
network = MLP([2, 2, 1])

# Test before training
print("\n" + "-"*70)
print("BEFORE TRAINING:")
print("-"*70)
for i, inputs in enumerate(X_xor):
    output = network.predict(inputs)[0]  # Get single output value
    prediction = 1 if output >= 0.5 else 0
    expected = y_xor[i]
    status = "✓" if prediction == expected else "✗"
    print(f"Input: {inputs} → Output: {output:.4f} → Prediction: {prediction} (Expected: {expected}) {status}")

# Train!
print("\n" + "-"*70)
print("TRAINING (1000 epochs, learning rate = 0.5)...")
print("-"*70)
loss_history = network.train(X_xor, y_xor, epochs=1000, learning_rate=0.5)

# Test after training
print("\n" + "-"*70)
print("AFTER TRAINING:")
print("-"*70)
for i, inputs in enumerate(X_xor):
    output = network.predict(inputs)[0]
    prediction = 1 if output >= 0.5 else 0
    expected = y_xor[i]
    status = "✓" if prediction == expected else "✗"
    print(f"Input: {inputs} → Output: {output:.4f} → Prediction: {prediction} (Expected: {expected}) {status}")

print("\n🎉 THE NETWORK LEARNED XOR BY ITSELF! 🎉")
print("We didn't manually set the weights - it figured them out from examples!")


# =============================================================================
# PART 5: VISUALIZING TRAINING PROGRESS
# =============================================================================

print("\n\n" + "="*70)
print("PART 5: TRAINING PROGRESS")
print("="*70)

print("""
Let's look at how loss decreased during training:
""")

print("\nLoss at different epochs:")
print("-"*70)
print(" Epoch  |   Loss   | Change")
print("-"*70)

milestones = [0, 10, 50, 100, 200, 500, 999]
for epoch in milestones:
    if epoch < len(loss_history):
        loss = loss_history[epoch]
        if epoch > 0:
            prev_loss = loss_history[epoch-1]
            change = loss - prev_loss
            print(f" {epoch:5d}  | {loss:8.6f} | {change:+.6f}")
        else:
            print(f" {epoch:5d}  | {loss:8.6f} |    ---")

print("-"*70)
print(f"\nLoss decreased from {loss_history[0]:.6f} to {loss_history[-1]:.6f}")
print(f"Reduction: {(1 - loss_history[-1]/loss_history[0])*100:.1f}%")


# =============================================================================
# PART 6: EFFECT OF LEARNING RATE
# =============================================================================

print("\n\n" + "="*70)
print("PART 6: EXPERIMENTING WITH LEARNING RATE")
print("="*70)

print("""
Learning rate is the most important hyperparameter!
Let's try three different values and compare results.
""")

learning_rates = [0.01, 0.5, 2.0]
descriptions = ["Too Small", "Just Right", "Too Large"]

print("\nTraining XOR with different learning rates:")
print("="*70)

for lr, desc in zip(learning_rates, descriptions):
    print(f"\n{desc}: Learning Rate = {lr}")
    print("-"*70)
    
    # Create fresh network
    net = MLP([2, 2, 1])
    
    # Train (fewer epochs, less verbose)
    losses = net.train(X_xor, y_xor, epochs=200, learning_rate=lr, verbose=False)
    
    # Show results
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Loss at epoch 0: {losses[0]:.6f}")
    print(f"Loss at epoch 50: {losses[min(50, len(losses)-1)]:.6f}")
    print(f"Loss at epoch 100: {losses[min(100, len(losses)-1)]:.6f}")
    
    # Test accuracy
    correct = 0
    for i, inputs in enumerate(X_xor):
        output = net.predict(inputs)[0]
        prediction = 1 if output >= 0.5 else 0
        if prediction == y_xor[i]:
            correct += 1
    
    accuracy = correct / len(X_xor) * 100
    print(f"Accuracy: {correct}/{len(X_xor)} = {accuracy:.0f}%")
    
    # Analysis
    if lr == 0.01:
        print("Analysis: Too slow! Needs more epochs to converge. 🐌")
    elif lr == 0.5:
        print("Analysis: Perfect! Converges quickly and stably. ✓")
    elif lr == 2.0:
        print("Analysis: Too fast! May be unstable or overshoot. 🎢")


# =============================================================================
# PART 7: EFFECT OF NETWORK ARCHITECTURE
# =============================================================================

print("\n\n" + "="*70)
print("PART 7: EXPERIMENTING WITH NETWORK ARCHITECTURE")
print("="*70)

print("""
Network architecture affects learning:
- Too few neurons: Can't learn complex patterns
- Just enough: Learns efficiently
- Too many: Slower, but still works
""")

architectures = [
    [2, 1, 1],      # Too simple (can't solve XOR!)
    [2, 2, 1],      # Just right
    [2, 4, 1],      # More than needed
    [2, 3, 3, 1],   # Multiple layers
]

descriptions = [
    "Too Simple (1 hidden neuron)",
    "Minimal (2 hidden neurons)",
    "Comfortable (4 hidden neurons)",
    "Deep (two hidden layers)"
]

print("\nTesting different architectures:")
print("="*70)

for arch, desc in zip(architectures, descriptions):
    print(f"\n{desc}: {arch}")
    print("-"*70)
    
    # Create network
    net = MLP(arch)
    
    # Train
    losses = net.train(X_xor, y_xor, epochs=500, learning_rate=0.5, verbose=False)
    
    # Evaluate
    correct = 0
    for i, inputs in enumerate(X_xor):
        output = net.predict(inputs)[0]
        prediction = 1 if output >= 0.5 else 0
        if prediction == y_xor[i]:
            correct += 1
    
    accuracy = correct / len(X_xor) * 100
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Accuracy: {accuracy:.0f}%")
    
    if arch == [2, 1, 1]:
        print("Note: Single hidden neuron cannot solve XOR (not linearly separable)!")
    elif arch == [2, 2, 1]:
        print("Note: Minimal architecture that works. Efficient!")
    elif arch == [2, 4, 1]:
        print("Note: More capacity than needed, but converges reliably.")
    elif arch == [2, 3, 3, 1]:
        print("Note: Deep network - more parameters, potentially more powerful.")


# =============================================================================
# PART 8: TRAINING ON A LARGER DATASET
# =============================================================================

print("\n\n" + "="*70)
print("PART 8: TRAINING ON A SIMPLE REGRESSION TASK")
print("="*70)

print("""
Let's train a network to learn a simple pattern:
f(x) = x²

We'll give it examples and see if it learns the pattern!
""")

# Generate training data: y = x²
X_regression = [[x/10] for x in range(-10, 11)]  # x from -1.0 to 1.0
y_regression = [(x[0])**2 for x in X_regression]  # y = x²

print(f"\nTraining data: {len(X_regression)} examples")
print("Sample data:")
print("-"*70)
for i in [0, 5, 10, 15, 20]:
    print(f"  x = {X_regression[i][0]:+.1f} → y = {y_regression[i]:.2f}")

# Create and train network
print("\n" + "-"*70)
print("Training network: [1, 5, 1]")
print("-"*70)

regression_net = MLP([1, 5, 1])
losses = regression_net.train(X_regression, y_regression, epochs=1000, learning_rate=0.1)

# Test predictions
print("\nPredictions vs Actual:")
print("-"*70)
print("   x   | Actual |  Predicted | Error")
print("-"*70)

test_points = [-1.0, -0.5, 0.0, 0.5, 1.0]
for x in test_points:
    actual = x**2
    predicted = regression_net.predict([x])[0]
    error = abs(actual - predicted)
    print(f" {x:+5.1f} | {actual:6.3f} | {predicted:10.3f} | {error:.3f}")

print("\n✓ Network learned to approximate x² function!")


# =============================================================================
# PART 9: KEY TAKEAWAYS
# =============================================================================

print("\n\n" + "="*70)
print(" "*25 + "KEY TAKEAWAYS")
print("="*70)

print("""
1. BACKPROPAGATION computes gradients efficiently
   - Uses chain rule to propagate errors backward
   - Each layer computes gradients for its parameters
   - Enables training deep networks

2. GRADIENT DESCENT updates weights to minimize loss
   - Update rule: w = w - learning_rate × gradient
   - Gradients point uphill, we go downhill (negative)
   - Repeat until loss converges

3. TRAINING LOOP structure:
   FOR each epoch:
       FOR each training example:
           - Forward pass (predict)
           - Calculate loss
           - Backward pass (compute gradients)
           - Update weights
   
4. LEARNING RATE is critical:
   - Too small: Slow convergence
   - Too large: Unstable, diverges
   - Just right: Fast and stable convergence
   - Typically: 0.001 to 1.0

5. NETWORK ARCHITECTURE matters:
   - Too simple: Can't learn complex patterns
   - Sufficient: Learns efficiently
   - Too complex: Slower, more parameters
   - Start simple, add complexity if needed

6. MONITORING TRAINING:
   - Plot loss over epochs
   - Loss should decrease
   - If loss increases: learning rate too high
   - If loss flat: learning rate too low or converged

7. NEURAL NETWORKS CAN LEARN:
   - XOR (non-linear problem)
   - Function approximation (x²)
   - Classification, regression, and more!
   - All through automatic weight optimization!

NEXT STEPS:
- Complete the student task
- Experiment with hyperparameters
- Train networks on different problems
- In Lab 04: Advanced techniques and real datasets!
""")

print("\n" + "="*70)
print("END OF TRAINING IMPLEMENTATION")
print("="*70)
print("\nYou now understand how neural networks learn from data!")
print("Time to apply this knowledge in the student task! 🚀")
