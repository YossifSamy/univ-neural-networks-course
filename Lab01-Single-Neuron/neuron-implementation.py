"""
Lab 01: Neuron Implementation
==============================

This file demonstrates how to implement an artificial neuron in Python
using FUNCTIONS (procedural programming approach).

In Lab 02, you'll learn Object-Oriented Programming (OOP) to organize
code better. For now, we use simple functions!

Follow along with your instructor as they explain each section.

Author: Neural Networks Course
Lab: 01 - Single Neuron
"""

import math


# =============================================================================
# PART 1: ACTIVATION FUNCTIONS
# =============================================================================

print("="*70)
print(" "*20 + "ACTIVATION FUNCTIONS")
print("="*70)

def step_function(z):
    """
    Step activation function (Heaviside function).
    Returns 1 if z >= 0, else 0.
    
    Use case: Binary classification (yes/no decisions)
    """
    if z >= 0:
        return 1
    else:
        return 0


def sigmoid_function(z):
    """
    Sigmoid activation function (Logistic function).
    Returns value between 0 and 1.
    
    Formula: σ(z) = 1 / (1 + e^(-z))
    
    Use case: Binary classification with probability output
    """
    return 1 / (1 + math.exp(-z))


def tanh_function(z):
    """
    Hyperbolic tangent activation function.
    Returns value between -1 and 1.
    
    Formula: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
    
    Use case: Hidden layers, zero-centered output
    """
    return math.tanh(z)


def relu_function(z):
    """
    Rectified Linear Unit (ReLU) activation function.
    Returns z if z > 0, else 0.
    
    Formula: ReLU(z) = max(0, z)
    
    Use case: Hidden layers in deep networks (most popular)
    """
    return max(0, z)


def leaky_relu_function(z, alpha=0.01):
    """
    Leaky ReLU activation function.
    Returns z if z > 0, else α*z.
    
    Formula: LeakyReLU(z) = max(α*z, z)
    
    Use case: Alternative to ReLU to avoid "dying neurons"
    """
    if z > 0:
        return z
    else:
        return alpha * z


# =============================================================================
# PART 2: NEURON CALCULATION FUNCTIONS
# =============================================================================

print("\n" + "="*70)
print(" "*20 + "NEURON CALCULATION FUNCTIONS")
print("="*70)

def calculate_weighted_sum(inputs, weights, bias):
    """
    Calculate the weighted sum of inputs plus bias.
    
    Formula: z = w₁*x₁ + w₂*x₂ + ... + wₙ*xₙ + b
    
    Parameters:
        inputs (list): List of input values
        weights (list): List of weights
        bias (float): Bias term
    
    Returns:
        float: The weighted sum (net input)
    """
    # Check if number of inputs matches number of weights
    if len(inputs) != len(weights):
        raise ValueError(f"Expected {len(weights)} inputs, got {len(inputs)}")
    
    # Calculate weighted sum
    z = 0
    for i in range(len(inputs)):
        z += weights[i] * inputs[i]
    
    # Add bias
    z += bias
    
    return z


def apply_activation(z, activation='step'):
    """
    Apply the activation function to the net input.
    
    Parameters:
        z (float): Net input (weighted sum)
        activation (str): Activation function name
            Options: 'step', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'linear'
    
    Returns:
        float: Activated output
    """
    if activation == 'step':
        return step_function(z)
    elif activation == 'sigmoid':
        return sigmoid_function(z)
    elif activation == 'tanh':
        return tanh_function(z)
    elif activation == 'relu':
        return relu_function(z)
    elif activation == 'leaky_relu':
        return leaky_relu_function(z)
    else:  # linear
        return z


def neuron_predict(inputs, weights, bias, activation='step'):
    """
    Make a prediction using a neuron.
    
    This is the complete forward pass:
    1. Calculate weighted sum
    2. Apply activation function
    
    Parameters:
        inputs (list): List of input values
        weights (list): List of weights
        bias (float): Bias term
        activation (str): Activation function name
    
    Returns:
        float: The neuron's output
    """
    # Step 1: Calculate net input
    z = calculate_weighted_sum(inputs, weights, bias)
    
    # Step 2: Apply activation function
    output = apply_activation(z, activation)
    
    return output


def neuron_predict_detailed(inputs, weights, bias, activation='step'):
    """
    Make a prediction with detailed output (for learning purposes).
    Shows each step of the calculation.
    
    Parameters:
        inputs (list): List of input values
        weights (list): List of weights
        bias (float): Bias term
        activation (str): Activation function name
    
    Returns:
        dict: Dictionary with 'z', 'output', and other details
    """
    print("\n" + "="*50)
    print("DETAILED PREDICTION")
    print("="*50)
    
    # Show inputs
    print(f"\nInputs: {inputs}")
    print(f"Weights: {weights}")
    print(f"Bias: {bias}")
    
    # Calculate each weighted input
    print("\nWeighted inputs:")
    z = 0
    for i in range(len(inputs)):
        weighted = weights[i] * inputs[i]
        z += weighted
        print(f"  w[{i}] * x[{i}] = {weights[i]} * {inputs[i]} = {weighted:.4f}")
    
    # Add bias
    print(f"\nAdding bias: {bias}")
    z += bias
    print(f"Net input (z) = weighted_sum + bias = {z:.4f}")
    
    # Apply activation
    output = apply_activation(z, activation)
    print(f"\nActivation function: {activation}")
    print(f"Output = {activation}({z:.4f}) = {output:.4f}")
    
    print("="*50)
    
    return {
        'z': z,
        'output': output,
        'inputs': inputs,
        'weights': weights,
        'bias': bias
    }


# =============================================================================
# PART 3: WATER BOTTLE CLASSIFIER EXAMPLE
# =============================================================================

print("\n\n" + "#"*70)
print("# WATER BOTTLE TEMPERATURE CLASSIFIER")
print("#"*70)

print("""
We'll create a neuron to classify whether a water bottle is HOT or COLD.

Inputs:
1. Temperature from touch (x₁): -10 (very cold) to +10 (very hot)
2. Visual cues (x₂): -5 (condensation) to +5 (steam)
3. Context (x₃): -3 (from fridge) to +3 (from sun)

Weights (importance):
- w₁ = 0.7 (Touch is most reliable - 70%)
- w₂ = 0.2 (Visual cues - 20%)
- w₃ = 0.1 (Context - 10%)

Bias: -2.0 (Conservative threshold for "HOT")

Activation: Step function (0 = COLD, 1 = HOT)
""")

# Define neuron parameters
water_weights = [0.7, 0.2, 0.1]  # Touch, Visual, Context importance
water_bias = -2.0                 # Conservative threshold
water_activation = 'step'

print("\n--- Test Case 1: Hot Water Bottle ---")
hot_inputs = [8, 3, 1]  # Hot touch, steam visible, warm room
print(f"Inputs: Touch={hot_inputs[0]}, Visual={hot_inputs[1]}, Context={hot_inputs[2]}")

result = neuron_predict_detailed(hot_inputs, water_weights, water_bias, water_activation)
classification = "HOT" if result['output'] == 1 else "COLD"
print(f"\n>>> CLASSIFICATION: {classification}")

print("\n--- Test Case 2: Cold Water Bottle ---")
cold_inputs = [-7, -4, -1]  # Cold touch, condensation, cold room
print(f"Inputs: Touch={cold_inputs[0]}, Visual={cold_inputs[1]}, Context={cold_inputs[2]}")

result = neuron_predict_detailed(cold_inputs, water_weights, water_bias, water_activation)
classification = "HOT" if result['output'] == 1 else "COLD"
print(f"\n>>> CLASSIFICATION: {classification}")

print("\n--- Test Case 3: Lukewarm Water Bottle ---")
lukewarm_inputs = [2, 0, 0]  # Slightly warm, no visual cues, neutral context
print(f"Inputs: Touch={lukewarm_inputs[0]}, Visual={lukewarm_inputs[1]}, Context={lukewarm_inputs[2]}")

result = neuron_predict_detailed(lukewarm_inputs, water_weights, water_bias, water_activation)
classification = "HOT" if result['output'] == 1 else "COLD"
print(f"\n>>> CLASSIFICATION: {classification}")


# =============================================================================
# PART 4: COMPARING ACTIVATION FUNCTIONS
# =============================================================================

print("\n\n" + "#"*70)
print("# COMPARING DIFFERENT ACTIVATION FUNCTIONS")
print("#"*70)

print("""
Let's see how different activation functions affect the output
for the same inputs, weights, and bias.
""")

# Test the same inputs with different activation functions
test_inputs = [8, 3, 1]
test_weights = [0.7, 0.2, 0.1]
test_bias = -2.0

activations = ['step', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'linear']

print(f"\nTest inputs: {test_inputs}")
print(f"Weights: {test_weights}")
print(f"Bias: {test_bias}")
print("\nResults with different activation functions:")
print("-" * 70)

for act in activations:
    output = neuron_predict(test_inputs, test_weights, test_bias, act)
    print(f"{act:12s}: {output:.6f}")

print("\nObservation:")
print("- Step: Sharp binary decision (0 or 1)")
print("- Sigmoid: Smooth probability-like output (0 to 1)")
print("- Tanh: Zero-centered output (-1 to 1)")
print("- ReLU: Keeps positive values as-is")
print("- Leaky ReLU: Small slope for negative values")
print("- Linear: No transformation (just the net input)")


# =============================================================================
# PART 5: VISUALIZING ACTIVATION FUNCTIONS
# =============================================================================

print("\n\n" + "#"*70)
print("# ACTIVATION FUNCTIONS ACROSS A RANGE OF VALUES")
print("#"*70)

print("""
Let's see how each activation function transforms different z values.
This helps understand their behavior!
""")

z_values = [-5, -3, -1, 0, 1, 3, 5]

print("\nz value |  Step  | Sigmoid |  Tanh  |  ReLU  | Leaky  | Linear")
print("-" * 70)

for z in z_values:
    step_out = step_function(z)
    sig_out = sigmoid_function(z)
    tanh_out = tanh_function(z)
    relu_out = relu_function(z)
    leaky_out = leaky_relu_function(z)
    linear_out = z
    
    print(f"{z:7.1f} | {step_out:6.3f} | {sig_out:7.4f} | {tanh_out:7.4f} | {relu_out:6.2f} | {leaky_out:6.3f} | {linear_out:6.2f}")


# =============================================================================
# PART 6: SENSITIVITY ANALYSIS - EFFECT OF WEIGHTS
# =============================================================================

print("\n\n" + "#"*70)
print("# SENSITIVITY ANALYSIS: HOW WEIGHTS AFFECT OUTPUT")
print("#"*70)

print("""
Let's see how changing weights affects the neuron's output.
This shows the importance of weight selection!
""")

print("\nTest inputs: [5, 5, 5] (All moderate values)")
print("Bias: -2.0")
print("Activation: sigmoid")

test_inputs = [5, 5, 5]
test_bias = -2.0

print("\n1. Original weights: [0.7, 0.2, 0.1] (Touch dominates)")
weights1 = [0.7, 0.2, 0.1]
output1 = neuron_predict(test_inputs, weights1, test_bias, 'sigmoid')
print(f"   Output: {output1:.4f}")

print("\n2. Equal weights: [0.33, 0.33, 0.33] (All inputs equally important)")
weights2 = [0.33, 0.33, 0.33]
output2 = neuron_predict(test_inputs, weights2, test_bias, 'sigmoid')
print(f"   Output: {output2:.4f}")

print("\n3. Visual dominant: [0.2, 0.7, 0.1] (Visual dominates)")
weights3 = [0.2, 0.7, 0.1]
output3 = neuron_predict(test_inputs, weights3, test_bias, 'sigmoid')
print(f"   Output: {output3:.4f}")

print("\nConclusion:")
print("- Weights control which inputs have more influence")
print("- Higher weight = more impact on final decision")
print("- Choosing right weights is crucial (usually learned through training)")


# =============================================================================
# PART 7: THE ROLE OF BIAS
# =============================================================================

print("\n\n" + "#"*70)
print("# THE ROLE OF BIAS: ADJUSTING THE THRESHOLD")
print("#"*70)

print("""
Bias shifts the decision boundary without depending on inputs.
Let's see how different bias values affect classification!
""")

print("\nTest inputs: [3, 2, 1] (Mildly warm)")
print("Weights: [0.7, 0.2, 0.1]")
print("Activation: step")
print("\nTrying different bias values:")
print("-" * 70)

test_inputs = [3, 2, 1]
test_weights = [0.7, 0.2, 0.1]
biases = [-5, -2, 0, 2, 5]

for b in biases:
    z = calculate_weighted_sum(test_inputs, test_weights, b)
    output = neuron_predict(test_inputs, test_weights, b, 'step')
    classification = "HOT" if output == 1 else "COLD"
    print(f"Bias = {b:2d}  →  z = {z:.2f}  →  Output = {output}  →  {classification}")

print("\nObservation:")
print("- Negative bias: Harder to activate (stricter 'HOT' classification)")
print("- Positive bias: Easier to activate (more lenient 'HOT' classification)")
print("- Bias = 0: Neutral threshold at z = 0")
print("- Bias is like adjusting a thermostat's sensitivity!")


# =============================================================================
# PART 8: PRACTICAL EXAMPLE - AND GATE (Logic Gate)
# =============================================================================

print("\n\n" + "#"*70)
print("# BONUS: IMPLEMENTING AN AND LOGIC GATE")
print("#"*70)

print("""
A single neuron can implement logic gates!
Let's create an AND gate that outputs 1 only if BOTH inputs are 1.
""")

print("\nAND Gate Truth Table:")
print("  Input1 | Input2 | Output")
print("  -------|--------|-------")
print("    0    |   0    |   0")
print("    0    |   1    |   0")
print("    1    |   0    |   0")
print("    1    |   1    |   1")

# AND gate neuron parameters
and_weights = [1, 1]      # Both inputs equally important
and_bias = -1.5           # Threshold requires sum > 1.5 (so both must be 1)
and_activation = 'step'

print("\nAND Gate Neuron:")
print(f"  Weights: {and_weights}")
print(f"  Bias: {and_bias}")
print(f"  Activation: {and_activation}")

print("\nTesting AND gate:")
test_cases = [[0, 0], [0, 1], [1, 0], [1, 1]]

for inputs in test_cases:
    output = neuron_predict(inputs, and_weights, and_bias, and_activation)
    print(f"  Input: {inputs} → Output: {output}")


# =============================================================================
# PART 9: PRACTICAL EXAMPLE - OR GATE
# =============================================================================

print("\n\n" + "#"*70)
print("# BONUS: IMPLEMENTING AN OR LOGIC GATE")
print("#"*70)

print("\nAn OR gate outputs 1 if EITHER input is 1")
print("\nOR Gate Truth Table:")
print("  Input1 | Input2 | Output")
print("  -------|--------|-------")
print("    0    |   0    |   0")
print("    0    |   1    |   1")
print("    1    |   0    |   1")
print("    1    |   1    |   1")

# OR gate neuron parameters
or_weights = [1, 1]       # Both inputs equally important
or_bias = -0.5            # Lower threshold (just one input being 1 is enough)
or_activation = 'step'

print("\nOR Gate Neuron:")
print(f"  Weights: {or_weights}")
print(f"  Bias: {or_bias}")
print(f"  Activation: {or_activation}")

print("\nTesting OR gate:")
for inputs in test_cases:
    output = neuron_predict(inputs, or_weights, or_bias, or_activation)
    print(f"  Input: {inputs} → Output: {output}")


# =============================================================================
# PART 10: CHALLENGE - CAN YOU IMPLEMENT XOR?
# =============================================================================

print("\n\n" + "#"*70)
print("# CHALLENGE: CAN A SINGLE NEURON SOLVE XOR?")
print("#"*70)

print("""
XOR (Exclusive OR) outputs 1 if inputs are DIFFERENT.

XOR Truth Table:
  Input1 | Input2 | Output
  -------|--------|-------
    0    |   0    |   0
    0    |   1    |   1
    1    |   0    |   1
    1    |   1    |   0

Challenge: Try to find weights and bias that make a single neuron solve XOR!

Spoiler: It's IMPOSSIBLE! A single neuron can only create linear decision
boundaries, but XOR is not linearly separable. You need multiple neurons
(layers) to solve it. That's what you'll learn in Lab 02!
""")

# Try with some weights (this won't work!)
xor_weights = [1, 1]
xor_bias = -0.5
xor_activation = 'step'

print(f"\nAttempting XOR with weights={xor_weights}, bias={xor_bias}:")
print("(This won't work, but let's see...)")

for inputs in test_cases:
    output = neuron_predict(inputs, xor_weights, xor_bias, xor_activation)
    expected = 1 if (inputs[0] != inputs[1]) else 0
    status = "✓" if output == expected else "✗"
    print(f"  Input: {inputs} → Output: {output}, Expected: {expected} {status}")

print("\nAs expected, a single neuron fails at XOR!")
print("This demonstrates why we need Multi-Layer Perceptrons (Lab 02)!")


# =============================================================================
# SUMMARY AND KEY TAKEAWAYS
# =============================================================================

print("\n\n" + "#"*70)
print("# KEY TAKEAWAYS")
print("#"*70)

print("""
1. NEURON STRUCTURE (using functions):
   - calculate_weighted_sum(): Computes z = Σ(wᵢ * xᵢ) + b
   - apply_activation(): Applies activation function f(z)
   - neuron_predict(): Complete forward pass

2. FORWARD PASS:
   z = Σ(wᵢ * xᵢ) + b    (weighted sum + bias)
   y = f(z)              (apply activation function)

3. ACTIVATION FUNCTIONS:
   - Step: Binary decisions (0 or 1)
   - Sigmoid: Probability-like output (0 to 1)
   - Tanh: Zero-centered output (-1 to 1)
   - ReLU: Fast, simple, popular (0 to ∞)
   - Leaky ReLU: Improved ReLU variant

4. WEIGHTS determine IMPORTANCE:
   - Higher weight = more influence
   - Negative weight = inverse relationship
   - Zero weight = input ignored

5. BIAS shifts the DECISION BOUNDARY:
   - Positive bias: Easier to activate
   - Negative bias: Harder to activate
   - Allows decision independent of inputs

6. SINGLE NEURON LIMITATIONS:
   - Can only learn linear decision boundaries
   - Cannot solve XOR problem
   - Need multiple neurons in layers for complex patterns!

7. WHY FUNCTIONS (NOT CLASSES) IN LAB 01:
   - Simpler to understand initially
   - Focus on neuron mathematics
   - In Lab 02, you'll learn OOP to organize code better!

NEXT STEPS:
- Complete the student task (student-task.py)
- Experiment with different weights and biases
- Try implementing other logic gates (NAND, NOR)
- Prepare for Lab 02: Multi-Layer Perceptron with OOP!
""")

print("\n" + "="*70)
print("END OF NEURON IMPLEMENTATION")
print("="*70)
print("\nYou've learned how neurons work using simple functions!")
print("In Lab 02, you'll learn OOP to build complex multi-layer networks!")
