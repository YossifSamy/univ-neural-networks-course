"""
Lab 01: Python Basics for Neural Networks
==========================================

This file covers the essential Python concepts you need to implement a neuron.
Work through each section carefully with your instructor.

Author: Neural Networks Course
Lab: 01 - Single Neuron
"""

# =============================================================================
# SECTION 1: VARIABLES AND DATA TYPES
# =============================================================================

print("=" * 60)
print("SECTION 1: VARIABLES AND DATA TYPES")
print("=" * 60)

# Variables are containers for storing data
# Think of them as labeled boxes

# Numbers (integers)
temperature = 8
visual_cue = 3
context = 1

print(f"Temperature from touch: {temperature}")
print(f"Visual cue: {visual_cue}")
print(f"Context: {context}")

# Numbers (floating-point - decimals)
weight_touch = 0.7
weight_visual = 0.2
weight_context = 0.1
bias = -2.0

print(f"\nWeights: {weight_touch}, {weight_visual}, {weight_context}")
print(f"Bias: {bias}")

# Strings (text)
classification = "HOT"
neuron_name = "Water Bottle Classifier"

print(f"\nClassification: {classification}")
print(f"Neuron name: {neuron_name}")

# Boolean (True/False)
is_hot = True
is_cold = False

print(f"\nIs it hot? {is_hot}")
print(f"Is it cold? {is_cold}")

# Type checking
print(f"\nType of temperature: {type(temperature)}")
print(f"Type of weight_touch: {type(weight_touch)}")
print(f"Type of classification: {type(classification)}")
print(f"Type of is_hot: {type(is_hot)}")

# =============================================================================
# SECTION 2: LISTS (ARRAYS)
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 2: LISTS")
print("=" * 60)

# Lists store multiple values in a single variable
# Perfect for storing inputs and weights!

# Creating a list of inputs
inputs = [8, 3, 1]
print(f"Inputs list: {inputs}")

# Creating a list of weights
weights = [0.7, 0.2, 0.1]
print(f"Weights list: {weights}")

# Accessing list elements (indexing starts at 0!)
print(f"\nFirst input (index 0): {inputs[0]}")
print(f"Second input (index 1): {inputs[1]}")
print(f"Third input (index 2): {inputs[2]}")

print(f"\nFirst weight (index 0): {weights[0]}")
print(f"Second weight (index 1): {weights[1]}")
print(f"Third weight (index 2): {weights[2]}")

# Length of a list
num_inputs = len(inputs)
num_weights = len(weights)
print(f"\nNumber of inputs: {num_inputs}")
print(f"Number of weights: {num_weights}")

# Adding elements to a list
inputs.append(2)  # Add room humidity as 4th input
print(f"After appending: {inputs}")

# Removing the last element
inputs.pop()
print(f"After removing last: {inputs}")

# List slicing (getting a portion)
first_two_inputs = inputs[0:2]  # Get elements at index 0 and 1
print(f"First two inputs: {first_two_inputs}")

# =============================================================================
# SECTION 3: MATHEMATICAL OPERATIONS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 3: MATHEMATICAL OPERATIONS")
print("=" * 60)

# Basic arithmetic
a = 10
b = 3

print(f"Addition: {a} + {b} = {a + b}")
print(f"Subtraction: {a} - {b} = {a - b}")
print(f"Multiplication: {a} * {b} = {a * b}")
print(f"Division: {a} / {b} = {a / b}")
print(f"Integer Division: {a} // {b} = {a // b}")
print(f"Modulus (remainder): {a} % {b} = {a % b}")
print(f"Power: {a} ** {b} = {a ** b}")

# Operations with variables from our neuron example
weighted_input_1 = weights[0] * inputs[0]
weighted_input_2 = weights[1] * inputs[1]
weighted_input_3 = weights[2] * inputs[2]

print(f"\nWeighted input 1: {weights[0]} * {inputs[0]} = {weighted_input_1}")
print(f"Weighted input 2: {weights[1]} * {inputs[1]} = {weighted_input_2}")
print(f"Weighted input 3: {weights[2]} * {inputs[2]} = {weighted_input_3}")

# Sum of weighted inputs
weighted_sum = weighted_input_1 + weighted_input_2 + weighted_input_3
print(f"\nWeighted sum: {weighted_sum}")

# Add bias
net_input = weighted_sum + bias
print(f"Net input (z): {net_input}")

# =============================================================================
# SECTION 4: CONDITIONAL STATEMENTS (IF/ELSE)
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 4: CONDITIONAL STATEMENTS")
print("=" * 60)

# If statements allow making decisions in code
# Essential for implementing activation functions!

# Simple if statement
z = 4.3

if z >= 0:
    print(f"z = {z} is non-negative")

# If-else statement
print("\nStep activation function:")
if z >= 0:
    output = 1
    print(f"z = {z} → output = {output} (HOT)")
else:
    output = 0
    print(f"z = {z} → output = {output} (COLD)")

# If-elif-else (multiple conditions)
temperature_value = 8

print("\nTemperature classification:")
if temperature_value < 0:
    print("Very Cold")
elif temperature_value < 3:
    print("Cold")
elif temperature_value < 7:
    print("Warm")
else:
    print("Hot")

# Comparison operators
x = 5
y = 10

print(f"\nComparison operators:")
print(f"{x} == {y}: {x == y}")  # Equal to
print(f"{x} != {y}: {x != y}")  # Not equal to
print(f"{x} < {y}: {x < y}")    # Less than
print(f"{x} > {y}: {x > y}")    # Greater than
print(f"{x} <= {y}: {x <= y}")  # Less than or equal
print(f"{x} >= {y}: {x >= y}")  # Greater than or equal

# Logical operators (and, or, not)
temp = 8
visual = 3

print(f"\nLogical operators:")
print(f"temp > 5 AND visual > 2: {temp > 5 and visual > 2}")
print(f"temp > 5 OR visual > 10: {temp > 5 or visual > 10}")
print(f"NOT (temp > 5): {not (temp > 5)}")

# =============================================================================
# SECTION 5: LOOPS (FOR LOOP)
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 5: LOOPS")
print("=" * 60)

# For loops repeat code multiple times
# Perfect for processing multiple inputs!

# Simple for loop with range
print("Counting from 0 to 4:")
for i in range(5):
    print(f"  i = {i}")

# Looping through a list
print("\nLooping through inputs:")
for input_value in inputs:
    print(f"  Input: {input_value}")

# Looping with index and value
print("\nLooping with both index and value:")
for i in range(len(inputs)):
    print(f"  Input[{i}] = {inputs[i]}")

# Better way: enumerate
print("\nUsing enumerate (better way):")
for index, value in enumerate(inputs):
    print(f"  Input[{index}] = {value}")

# Calculate weighted sum using a loop
print("\nCalculating weighted sum with a loop:")
weighted_sum = 0
for i in range(len(inputs)):
    weighted_value = weights[i] * inputs[i]
    weighted_sum += weighted_value  # Same as: weighted_sum = weighted_sum + weighted_value
    print(f"  weights[{i}] * inputs[{i}] = {weights[i]} * {inputs[i]} = {weighted_value}")
    print(f"  Running sum: {weighted_sum}")

print(f"\nFinal weighted sum: {weighted_sum}")

# More efficient way using zip
print("\nUsing zip (pairs up two lists):")
weighted_sum = 0
for w, x in zip(weights, inputs):
    weighted_sum += w * x
    print(f"  {w} * {x} = {w * x}, running sum = {weighted_sum}")

# =============================================================================
# SECTION 6: FUNCTIONS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 6: FUNCTIONS")
print("=" * 60)

# Functions organize code into reusable blocks
# They take inputs (parameters) and return outputs

# Simple function with no parameters
def greet():
    """This function prints a greeting."""
    print("Hello, Neural Networks student!")

greet()

# Function with parameters
def calculate_weighted_value(weight, input_value):
    """Calculate a single weighted input."""
    return weight * input_value

result = calculate_weighted_value(0.7, 8)
print(f"\nWeighted value: {result}")

# Function with multiple parameters and return value
def calculate_weighted_sum(weights_list, inputs_list):
    """
    Calculate the weighted sum of inputs.
    
    Parameters:
        weights_list: List of weights
        inputs_list: List of inputs
    
    Returns:
        The weighted sum (float)
    """
    total = 0
    for w, x in zip(weights_list, inputs_list):
        total += w * x
    return total

sum_result = calculate_weighted_sum(weights, inputs)
print(f"\nWeighted sum from function: {sum_result}")

# Function with default parameters
def calculate_net_input(weights_list, inputs_list, bias_value=0):
    """
    Calculate net input (weighted sum + bias).
    
    Parameters:
        weights_list: List of weights
        inputs_list: List of inputs
        bias_value: Bias term (default is 0)
    
    Returns:
        Net input z
    """
    weighted_sum = calculate_weighted_sum(weights_list, inputs_list)
    return weighted_sum + bias_value

z1 = calculate_net_input(weights, inputs, -2)  # With bias
z2 = calculate_net_input(weights, inputs)       # Without bias (uses default 0)

print(f"\nNet input with bias -2: {z1}")
print(f"Net input with default bias 0: {z2}")

# Activation functions
def step_activation(z):
    """Step activation function."""
    if z >= 0:
        return 1
    else:
        return 0

def sigmoid_activation(z):
    """Sigmoid activation function."""
    import math
    return 1 / (1 + math.exp(-z))

def relu_activation(z):
    """ReLU activation function."""
    return max(0, z)

# Test activation functions
test_z = 4.3
print(f"\nTesting activation functions with z = {test_z}:")
print(f"  Step: {step_activation(test_z)}")
print(f"  Sigmoid: {sigmoid_activation(test_z):.4f}")
print(f"  ReLU: {relu_activation(test_z)}")

# =============================================================================
# SECTION 7: IMPORTING MODULES
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 7: IMPORTING MODULES")
print("=" * 60)

# Python has many built-in modules (libraries)
# We import them to use additional functionality

# Import the entire math module
import math

print(f"Pi: {math.pi}")
print(f"e (Euler's number): {math.e}")
print(f"Square root of 16: {math.sqrt(16)}")
print(f"e^2: {math.exp(2)}")

# Import specific functions
from math import exp, tanh

z_value = 2.5
sigmoid_result = 1 / (1 + exp(-z_value))
tanh_result = tanh(z_value)

print(f"\nFor z = {z_value}:")
print(f"  Sigmoid: {sigmoid_result:.4f}")
print(f"  Tanh: {tanh_result:.4f}")

# Import with alias (shorter name)
import math as m

print(f"\nUsing alias: cos(0) = {m.cos(0)}")

# =============================================================================
# SECTION 8: PUTTING IT ALL TOGETHER
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 8: COMPLETE NEURON CALCULATION")
print("=" * 60)

def neuron_predict(inputs_list, weights_list, bias_value, activation='step'):
    """
    Complete neuron prediction.
    
    Parameters:
        inputs_list: List of input values
        weights_list: List of weights
        bias_value: Bias term
        activation: Activation function name ('step', 'sigmoid', 'relu')
    
    Returns:
        Output of the neuron
    """
    # Step 1: Calculate weighted sum
    z = 0
    for w, x in zip(weights_list, inputs_list):
        z += w * x
    
    # Step 2: Add bias
    z += bias_value
    
    print(f"  Net input (z): {z:.4f}")
    
    # Step 3: Apply activation function
    if activation == 'step':
        output = 1 if z >= 0 else 0
    elif activation == 'sigmoid':
        output = 1 / (1 + math.exp(-z))
    elif activation == 'relu':
        output = max(0, z)
    else:
        output = z  # Linear (no activation)
    
    return output

# Test the complete neuron
print("\nWater Bottle Classification:")
print(f"Inputs: {inputs}")
print(f"Weights: {weights}")
print(f"Bias: {bias}")

print("\nUsing Step activation:")
output_step = neuron_predict(inputs, weights, bias, 'step')
print(f"  Output: {output_step} → {'HOT' if output_step == 1 else 'COLD'}")

print("\nUsing Sigmoid activation:")
output_sigmoid = neuron_predict(inputs, weights, bias, 'sigmoid')
print(f"  Output: {output_sigmoid:.4f} → {output_sigmoid*100:.2f}% confidence HOT")

print("\nUsing ReLU activation:")
output_relu = neuron_predict(inputs, weights, bias, 'relu')
print(f"  Output: {output_relu:.4f}")

# =============================================================================
# PRACTICE EXERCISES
# =============================================================================

print("\n" + "=" * 60)
print("PRACTICE EXERCISES")
print("=" * 60)

print("""
Try these exercises to practice:

1. Create a list of 5 different temperature readings.
   Calculate their average using a for loop.

2. Write a function that takes a temperature value and returns
   "Cold", "Warm", or "Hot" based on thresholds you define.

3. Modify the neuron_predict function to add a new activation
   function called 'tanh'.

4. Create a neuron with 4 inputs instead of 3. Add a weight
   for "humidity" as the 4th input.

5. Test the neuron with different input combinations:
   - Very cold bottle: inputs = [-8, -4, -2]
   - Lukewarm bottle: inputs = [2, 0, 0]
   - Very hot bottle: inputs = [10, 5, 3]

Try writing these in a new Python file!
""")

print("\n" + "=" * 60)
print("END OF PYTHON BASICS TUTORIAL")
print("=" * 60)
print("\nYou're now ready to implement a complete neuron class!")
print("Proceed to: neuron-implementation.py")
