"""
Lab 02: Multi-Layer Perceptron Implementation
==============================================

This file shows how to implement an MLP in two ways:
1. WITHOUT OOP (procedural) - to show why OOP is needed
2. WITH OOP (professional) - the right way

Author: Neural Networks Course
Lab: 02 - Multi-Layer Perceptron
"""

import math
import random


print("=" * 70)
print(" " * 15 + "MULTI-LAYER PERCEPTRON IMPLEMENTATION")
print("=" * 70)


# =============================================================================
# SECTION 1: WITHOUT OOP (PROCEDURAL APPROACH)
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 1: WITHOUT OOP - The Messy Way")
print("=" * 70)

print("""
Let's build a simple [2, 2, 1] network (solves XOR) WITHOUT OOP.
Watch how quickly it becomes messy!
""")

# Define network weights manually - Very messy!
weights_hidden_layer = [
    [1.0, 1.0],  # Weights for first hidden neuron
    [1.0, 1.0]   # Weights for second hidden neuron
]
biases_hidden_layer = [-0.5, -1.5]

weights_output_layer = [1.0, -2.0]
bias_output_layer = -0.5


def sigmoid_old(z):
    """Sigmoid activation function."""
    return 1 / (1 + math.exp(-z))


def forward_hidden_layer_old(inputs, weights, biases):
    """Calculate hidden layer output - procedural way."""
    outputs = []
    for i in range(len(weights)):
        # Calculate weighted sum for this neuron
        z = sum(w * x for w, x in zip(weights[i], inputs)) + biases[i]
        # Apply activation
        a = sigmoid_old(z)
        outputs.append(a)
    return outputs


def forward_output_layer_old(inputs, weights, bias):
    """Calculate output layer - procedural way."""
    z = sum(w * x for w, x in zip(weights, inputs)) + bias
    return sigmoid_old(z)


def predict_old(inputs):
    """Full network prediction - procedural way."""
    # Forward through hidden layer
    hidden = forward_hidden_layer_old(inputs, weights_hidden_layer, biases_hidden_layer)
    # Forward through output layer
    output = forward_output_layer_old(hidden, weights_output_layer, bias_output_layer)
    return output


# Test XOR
print("\nTesting XOR Problem (Procedural Approach):")
xor_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
xor_expected = [0, 1, 1, 0]

for i, inputs in enumerate(xor_inputs):
    output = predict_old(inputs)
    prediction = 1 if output >= 0.5 else 0
    status = "✓" if prediction == xor_expected[i] else "✗"
    print(f"  Input: {inputs} → Output: {output:.4f} → Prediction: {prediction} (Expected: {xor_expected[i]}) {status}")

print("\n❌ PROBLEMS:")
print("  - Hard to add more layers")
print("  - Weights are scattered everywhere")
print("  - Functions need many parameters")
print("  - Difficult to maintain and debug")
print("\n💡 Let's see the OOP way...")


# =============================================================================
# SECTION 2: WITH OOP (PROFESSIONAL APPROACH)
# =============================================================================

print("\n\n" + "=" * 70)
print("SECTION 2: WITH OOP - The Professional Way")
print("=" * 70)

class Layer:
    """
    Represents one layer in the neural network.
    """
    
    def __init__(self, num_inputs, num_neurons, activation='sigmoid'):
        """
        Initialize a layer.
        
        Parameters:
            num_inputs (int): Number of inputs to this layer
            num_neurons (int): Number of neurons in this layer
            activation (str): Activation function ('sigmoid', 'relu', 'tanh')
        """
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.activation = activation
        
        # Initialize weights and biases with small random values
        self.weights = [[random.uniform(-1, 1) for _ in range(num_inputs)] 
                       for _ in range(num_neurons)]
        self.biases = [random.uniform(-1, 1) for _ in range(num_neurons)]
    
    def set_weights(self, weights, biases):
        """Manually set weights and biases (for testing)."""
        self.weights = weights
        self.biases = biases
    
    def forward(self, inputs):
        """
        Forward propagation through this layer.
        
        Parameters:
            inputs (list): Input values from previous layer
        
        Returns:
            list: Activated outputs from this layer
        """
        outputs = []
        
        for i in range(self.num_neurons):
            # Calculate weighted sum
            z = sum(w * x for w, x in zip(self.weights[i], inputs)) + self.biases[i]
            
            # Apply activation function
            if self.activation == 'sigmoid':
                a = 1 / (1 + math.exp(-z))
            elif self.activation == 'relu':
                a = max(0, z)
            elif self.activation == 'tanh':
                a = math.tanh(z)
            else:  # linear
                a = z
            
            outputs.append(a)
        
        return outputs
    
    def __repr__(self):
        return f"Layer({self.num_inputs}→{self.num_neurons}, {self.activation})"


class MLP:
    """
    Multi-Layer Perceptron - A feedforward neural network.
    """
    
    def __init__(self, architecture, activation='sigmoid'):
        """
        Initialize an MLP.
        
        Parameters:
            architecture (list): List of layer sizes [inputs, hidden1, hidden2, ..., output]
            activation (str): Default activation function for all layers
        
        Example:
            MLP([2, 4, 3, 1]) creates:
            - 2 inputs
            - Hidden layer 1 with 4 neurons
            - Hidden layer 2 with 3 neurons
            - Output layer with 1 neuron
        """
        self.architecture = architecture
        self.layers = []
        
        # Create layers
        for i in range(len(architecture) - 1):
            num_inputs = architecture[i]
            num_neurons = architecture[i + 1]
            layer = Layer(num_inputs, num_neurons, activation)
            self.layers.append(layer)
        
        print(f"✓ MLP created with architecture {architecture}")
        print(f"  Total layers: {len(self.layers)}")
        for i, layer in enumerate(self.layers):
            print(f"  Layer {i+1}: {layer}")
    
    def forward(self, inputs):
        """
        Forward propagation through entire network.
        
        Parameters:
            inputs (list): Input features
        
        Returns:
            list: Output predictions
        """
        activation = inputs
        
        # Pass through each layer
        for layer in self.layers:
            activation = layer.forward(activation)
        
        return activation
    
    def predict(self, inputs):
        """
        Make a prediction (same as forward for now).
        
        Parameters:
            inputs (list): Input features
        
        Returns:
            list: Output predictions
        """
        return self.forward(inputs)
    
    def predict_binary(self, inputs, threshold=0.5):
        """
        Make binary prediction (for classification).
        
        Parameters:
            inputs (list): Input features
            threshold (float): Decision threshold
        
        Returns:
            int: 0 or 1
        """
        output = self.predict(inputs)[0]  # Get first output
        return 1 if output >= threshold else 0
    
    def predict_detailed(self, inputs):
        """
        Make prediction and show all intermediate values.
        """
        print(f"\n{'=' * 60}")
        print(f"DETAILED FORWARD PROPAGATION")
        print(f"{'=' * 60}")
        print(f"Input: {inputs}")
        
        activation = inputs
        
        for i, layer in enumerate(self.layers):
            print(f"\n--- Layer {i+1} ({layer}) ---")
            print(f"Input to layer: {[f'{x:.4f}' for x in activation]}")
            
            activation = layer.forward(activation)
            
            print(f"Output from layer: {[f'{x:.4f}' for x in activation]}")
        
        print(f"\n{'=' * 60}")
        print(f"Final Output: {[f'{x:.4f}' for x in activation]}")
        print(f"{'=' * 60}")
        
        return activation


# =============================================================================
# SECTION 3: TESTING XOR WITH OOP MLP
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3: SOLVING XOR WITH OOP MLP")
print("=" * 70)

# Create network
xor_network = MLP([2, 2, 1], activation='sigmoid')

# Set the same weights as before (for fair comparison)
xor_network.layers[0].set_weights(
    [[1.0, 1.0], [1.0, 1.0]],
    [-0.5, -1.5]
)
xor_network.layers[1].set_weights(
    [[1.0, -2.0]],
    [-0.5]
)

print("\nTesting XOR Problem (OOP Approach):")
for i, inputs in enumerate(xor_inputs):
    output = xor_network.predict(inputs)[0]
    prediction = 1 if output >= 0.5 else 0
    status = "✓" if prediction == xor_expected[i] else "✗"
    print(f"  Input: {inputs} → Output: {output:.4f} → Prediction: {prediction} (Expected: {xor_expected[i]}) {status}")

# Show detailed prediction for one case
print("\n" + "-" * 70)
print("Detailed prediction for input [1, 0]:")
xor_network.predict_detailed([1, 0])

print("\n✓ BENEFITS OF OOP:")
print("  - Clean, organized code")
print("  - Easy to change architecture")
print("  - Simple to add/remove layers")
print("  - Professional structure")
print("  - Reusable components")


# =============================================================================
# SECTION 4: PRACTICAL EXAMPLE - IRIS FLOWER CLASSIFICATION
# =============================================================================

print("\n\n" + "=" * 70)
print("SECTION 4: PRACTICAL APPLICATION - IRIS CLASSIFICATION")
print("=" * 70)

print("""
The Iris dataset is a classic machine learning dataset:
- 150 samples of iris flowers
- 3 species: Setosa, Versicolor, Virginica
- 4 features: Sepal length, sepal width, petal length, petal width

We'll create a network to classify iris flowers!
""")

# Sample iris data (simplified)
# Format: [sepal_length, sepal_width, petal_length, petal_width]
iris_samples = [
    # Setosa (class 0)
    {"features": [5.1, 3.5, 1.4, 0.2], "species": "Setosa", "class": 0},
    {"features": [4.9, 3.0, 1.4, 0.2], "species": "Setosa", "class": 0},
    
    # Versicolor (class 1)
    {"features": [7.0, 3.2, 4.7, 1.4], "species": "Versicolor", "class": 1},
    {"features": [6.4, 3.2, 4.5, 1.5], "species": "Versicolor", "class": 1},
    
    # Virginica (class 2)
    {"features": [6.3, 3.3, 6.0, 2.5], "species": "Virginica", "class": 2},
    {"features": [5.8, 2.7, 5.1, 1.9], "species": "Virginica", "class": 2},
]

# Create network for iris classification
# Architecture: [4, 8, 3]
# - 4 inputs (4 features)
# - 8 hidden neurons
# - 3 outputs (3 classes)
print("\nCreating Iris Classification Network...")
iris_network = MLP([4, 8, 3], activation='sigmoid')

print("\nNote: This network has RANDOM weights (untrained).")
print("In Lab 03, you'll learn how to TRAIN networks to learn correct weights!")
print("\nMaking predictions with random weights (just for demonstration):")

for sample in iris_samples:
    features = sample['features']
    true_species = sample['species']
    outputs = iris_network.predict(features)
    
    # Find predicted class (highest output)
    predicted_class = outputs.index(max(outputs))
    species_names = ["Setosa", "Versicolor", "Virginica"]
    predicted_species = species_names[predicted_class]
    
    print(f"\nFeatures: {features}")
    print(f"  Outputs: [{outputs[0]:.3f}, {outputs[1]:.3f}, {outputs[2]:.3f}]")
    print(f"  Predicted: {predicted_species} (class {predicted_class})")
    print(f"  Actual: {true_species} (class {sample['class']})")


# =============================================================================
# SECTION 5: COMPARING DIFFERENT ARCHITECTURES
# =============================================================================

print("\n\n" + "=" * 70)
print("SECTION 5: EXPERIMENTING WITH ARCHITECTURES")
print("=" * 70)

print("""
Let's see how different architectures affect the network structure.
All use random weights (not trained).
""")

architectures = [
    [4, 3],           # Shallow: direct from input to output
    [4, 8, 3],        # Medium: one hidden layer
    [4, 16, 8, 3],    # Deep: two hidden layers
    [4, 32, 16, 8, 3] # Very deep: three hidden layers
]

for arch in architectures:
    print(f"\n{'=' * 60}")
    print(f"Architecture: {arch}")
    
    # Count parameters
    total_params = 0
    for i in range(len(arch) - 1):
        weights = arch[i] * arch[i + 1]
        biases = arch[i + 1]
        layer_params = weights + biases
        total_params += layer_params
        print(f"  Layer {i+1}: {arch[i]}→{arch[i+1]} = {layer_params} parameters")
    
    print(f"  TOTAL: {total_params} parameters")
    
    # Create network
    net = MLP(arch, activation='relu')


# =============================================================================
# SECTION 6: ACTIVATION FUNCTION COMPARISON
# =============================================================================

print("\n\n" + "=" * 70)
print("SECTION 6: COMPARING ACTIVATION FUNCTIONS")
print("=" * 70)

activations = ['sigmoid', 'relu', 'tanh']
test_input = [0.5, -0.3, 0.8]

print(f"\nTest input: {test_input}")
print("\nOutputs with different activation functions:")

for act in activations:
    net = MLP([3, 4, 2], activation=act)
    output = net.predict(test_input)
    print(f"  {act:10s}: {[f'{x:.4f}' for x in output]}")


# =============================================================================
# SUMMARY AND KEY TAKEAWAYS
# =============================================================================

print("\n\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("""
1. MLP STRUCTURE:
   - Input layer: Holds input features
   - Hidden layer(s): Extract patterns and features
   - Output layer: Produces final predictions

2. FORWARD PROPAGATION:
   - Data flows from input to output
   - Each layer transforms the data
   - Formula: output = activation(weights × inputs + bias)

3. OOP FOR NEURAL NETWORKS:
   - Layer class: Encapsulates one layer
   - MLP class: Manages multiple layers
   - Clean, maintainable, scalable code

4. ARCHITECTURE DESIGN:
   - Notation: [inputs, hidden1, hidden2, ..., output]
   - More layers = deeper network (can learn more complex patterns)
   - More neurons = wider network (more representational capacity)

5. FULLY CONNECTED:
   - Every neuron connects to all neurons in next layer
   - Lots of parameters to learn!
   - Parameters = (num_inputs × num_neurons) + num_neurons per layer

6. CURRENT LIMITATION:
   - We're using RANDOM weights
   - Network doesn't "know" anything yet
   - Need TRAINING to learn correct weights
   - Training comes in Lab 03!

COMPARISON:
-----------
Without OOP: Messy, hard to scale, error-prone
With OOP: Clean, easy to modify, professional

Next Steps:
-----------
1. Complete student-task.py
2. Experiment with different architectures
3. Prepare for Lab 03: Training Neural Networks!
""")

print("\n" + "=" * 70)
print("END OF MLP IMPLEMENTATION")
print("=" * 70)
