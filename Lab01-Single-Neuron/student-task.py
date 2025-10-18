"""
Lab 01: Student Task - Single Neuron Implementation
====================================================

Complete the following tasks to practice what you've learned about neurons.

SUBMISSION INSTRUCTIONS:
1. Complete all three tasks below
2. Test your code with the provided test cases
3. Add your own test cases
4. Submit this file before the end of lab

Your Name: _________________________
Student ID: _________________________
Date: _________________________

"""

import math


# =============================================================================
# TASK 1: FRUIT RIPENESS CLASSIFIER
# =============================================================================

print("="*60)
print("TASK 1: FRUIT RIPENESS CLASSIFIER")
print("="*60)

"""
SCENARIO:
You're building a smart system to determine if a banana is RIPE or UNRIPE.

INPUTS (scale -10 to +10):
1. Color: -10 (green) to +10 (brown), 0 (yellow)
2. Smell: -10 (no smell) to +10 (very sweet smell)
3. Firmness: -10 (very hard) to +10 (very soft/mushy)

OUTPUT:
- 1 = RIPE (ready to eat)
- 0 = UNRIPE (not ready)

YOUR TASK:
1. Create a Neuron class (copy from neuron-implementation.py or write your own)
2. Choose appropriate weights for each input
3. Choose an appropriate bias
4. Choose an activation function
5. Test with the provided test cases

HINT: Think about which features are most important for ripeness!
- A perfectly ripe banana is yellow (0), sweet smell (7-9), and slightly soft (3-5)
- An unripe banana is green (-7 to -3), little smell (0-2), and hard (-5 to -2)
"""

# =============================================================================
# HELPER FUNCTIONS (Copy from neuron-implementation.py or write your own)
# =============================================================================

# TODO: Implement these activation functions
def step_function(z):
    """Step activation: returns 1 if z >= 0, else 0"""
    # TODO: Your code here
    pass


def sigmoid_function(z):
    """Sigmoid activation: returns value between 0 and 1"""
    # TODO: Your code here
    # Formula: 1 / (1 + e^(-z))
    pass


def tanh_function(z):
    """Tanh activation: returns value between -1 and 1"""
    # TODO: Your code here
    pass


def relu_function(z):
    """ReLU activation: returns z if z > 0, else 0"""
    # TODO: Your code here
    pass


def calculate_weighted_sum(inputs, weights, bias):
    """
    Calculate weighted sum: z = w₁*x₁ + w₂*x₂ + ... + wₙ*xₙ + b
    
    Parameters:
        inputs (list): Input values
        weights (list): Weight values
        bias (float): Bias value
    
    Returns:
        float: The weighted sum
    """
    # TODO: Your code here
    pass


def apply_activation(z, activation='step'):
    """
    Apply activation function to z
    
    Parameters:
        z (float): Net input (weighted sum)
        activation (str): Name of activation function ('step', 'sigmoid', 'tanh', 'relu')
    
    Returns:
        float: Activated output
    """
    # TODO: Your code here
    pass


def neuron_predict(inputs, weights, bias, activation='step'):
    """
    Complete neuron prediction using functions
    
    Parameters:
        inputs (list): Input values
        weights (list): Weight values
        bias (float): Bias value
        activation (str): Activation function name
    
    Returns:
        float: Neuron's output
    """
    # TODO: Your code here
    # Step 1: Calculate weighted sum
    # Step 2: Apply activation function
    pass


# TODO: Define your fruit ripeness classifier parameters
# Choose your weights, bias, and activation function
fruit_weights = None  # Replace with: [?, ?, ?]
fruit_bias = None     # Replace with: ?
fruit_activation = None  # Replace with: 'step' or 'sigmoid' etc.

# TODO: Uncomment and complete this section after implementing the functions above
"""
print("\nFruit Ripeness Classifier Configuration:")
print(f"Weights: {fruit_weights}")
print(f"Bias: {fruit_bias}")
print(f"Activation: {fruit_activation}")

# Test Case 1: Ripe banana
print("\n--- Test Case 1: Perfect Ripe Banana ---")
ripe_inputs = [0, 8, 4]  # Yellow, sweet smell, slightly soft
print(f"Inputs - Color: {ripe_inputs[0]}, Smell: {ripe_inputs[1]}, Firmness: {ripe_inputs[2]}")
output = neuron_predict(ripe_inputs, fruit_weights, fruit_bias, fruit_activation)
print(f"Output: {output} → {'RIPE ✓' if output == 1 else 'UNRIPE ✗'}")

# Test Case 2: Unripe banana
print("\n--- Test Case 2: Unripe Banana ---")
unripe_inputs = [-5, 1, -3]  # Green, little smell, hard
print(f"Inputs - Color: {unripe_inputs[0]}, Smell: {unripe_inputs[1]}, Firmness: {unripe_inputs[2]}")
output = neuron_predict(unripe_inputs, fruit_weights, fruit_bias, fruit_activation)
print(f"Output: {output} → {'RIPE ✗' if output == 1 else 'UNRIPE ✓'}")

# Test Case 3: Overripe banana
print("\n--- Test Case 3: Overripe Banana ---")
overripe_inputs = [7, 10, 8]  # Brown, very sweet, very soft
print(f"Inputs - Color: {overripe_inputs[0]}, Smell: {overripe_inputs[1]}, Firmness: {overripe_inputs[2]}")
output = neuron_predict(overripe_inputs, fruit_weights, fruit_bias, fruit_activation)
print(f"Output: {output} → {'RIPE' if output == 1 else 'UNRIPE'}")
print("(Note: This might be overripe, but still technically ripe)")

# TODO: Add your own test case
print("\n--- Your Test Case ---")
# your_inputs = [?, ?, ?]
# output = neuron_predict(your_inputs, fruit_weights, fruit_bias, fruit_activation)
# print(f"Output: {output}")
"""

# QUESTIONS TO ANSWER:
print("\n" + "="*60)
print("QUESTIONS FOR TASK 1:")
print("="*60)
print("""
1. What weights did you choose and why?
   Answer: 


2. What bias did you choose and why?
   Answer: 


3. Which activation function did you use and why?
   Answer: 


4. Did your classifier correctly identify all test cases?
   If not, what adjustments did you make?
   Answer: 


""")


# =============================================================================
# TASK 2: ACTIVATION FUNCTION COMPARISON
# =============================================================================

print("\n" + "="*60)
print("TASK 2: ACTIVATION FUNCTION COMPARISON")
print("="*60)

"""
TASK:
Use the SAME fruit classifier configuration (same weights and bias),
but test with DIFFERENT activation functions.

Compare the outputs and understand how each activation function behaves.
"""

# TODO: Test with different activation functions
"""
print("\nComparing Activation Functions")
print("Weights: [your weights]")
print("Bias: your bias")
print("Test inputs: [2, 5, 2] (slightly yellow, moderate smell, slightly soft)")

test_inputs = [2, 5, 2]
test_weights = [?, ?, ?]  # Use same weights as Task 1
test_bias = ?              # Use same bias as Task 1

activations = ['step', 'sigmoid', 'tanh', 'relu']

for act in activations:
    # Use neuron_predict function with different activations
    output = neuron_predict(test_inputs, test_weights, test_bias, act)
    print(f"{act:10s}: {output:.4f}")
"""

# QUESTIONS TO ANSWER:
print("\n" + "="*60)
print("QUESTIONS FOR TASK 2:")
print("="*60)
print("""
1. Which activation function gives the most interpretable output?
   Answer: 


2. How does sigmoid output differ from step output?
   Answer: 


3. When might you prefer sigmoid over step function?
   Answer: 


4. What happened with ReLU? Is it suitable for this problem?
   Answer: 


""")


# =============================================================================
# TASK 3: SMART LIGHT SWITCH CONTROLLER
# =============================================================================

print("\n" + "="*60)
print("TASK 3: SMART LIGHT SWITCH CONTROLLER")
print("="*60)

"""
SCENARIO:
Design a neuron that controls a smart light switch. It should decide
whether to turn the light ON (1) or OFF (0) based on environmental factors.

INPUTS (scale 0 to 10):
1. Ambient Light Level: 0 (complete darkness) to 10 (very bright)
2. Time of Day: 0 (midnight) to 10 (noon), then back to 0 (midnight)
3. Motion Detected: 0 (no motion) to 10 (lots of motion)
4. User Preference Setting: 0 (likes dark) to 10 (likes bright)

OUTPUT:
- 1 = Turn light ON
- 0 = Keep light OFF

YOUR TASK:
1. Design weights that make sense for a smart light
2. Choose an appropriate bias
3. Test with various scenarios

HINT: 
- Low ambient light + motion detected → should turn ON
- High ambient light → probably don't need light
- User preference should have some influence
"""

# TODO: Implement your smart light controller
"""
# Define your light controller parameters
light_weights = [?, ?, ?, ?]  # ambient, time, motion, preference
light_bias = ?
light_activation = 'step'

print("Smart Light Controller Configuration:")
print(f"Weights: {light_weights}")
print(f"Bias: {light_bias}")

# Test Scenario 1: Dark room, evening, person enters
print("\n--- Scenario 1: Dark Room, Evening, Person Enters ---")
inputs1 = [1, 3, 8, 5]  # Dark, evening, motion detected, neutral preference
print(f"Inputs - Light:{inputs1[0]}, Time:{inputs1[1]}, Motion:{inputs1[2]}, Pref:{inputs1[3]}")
output1 = neuron_predict(inputs1, light_weights, light_bias, light_activation)
print(f"Decision: {'Turn ON ✓' if output1 == 1 else 'Keep OFF'}")

# Test Scenario 2: Bright room, midday, no motion
print("\n--- Scenario 2: Bright Room, Midday, No Motion ---")
inputs2 = [9, 10, 0, 5]  # Bright, noon, no motion, neutral preference
print(f"Inputs - Light:{inputs2[0]}, Time:{inputs2[1]}, Motion:{inputs2[2]}, Pref:{inputs2[3]}")
output2 = neuron_predict(inputs2, light_weights, light_bias, light_activation)
print(f"Decision: {'Turn ON' if output2 == 1 else 'Keep OFF ✓'}")

# Test Scenario 3: Medium light, evening, some motion, user likes bright
print("\n--- Scenario 3: Medium Light, Evening, Some Motion, Likes Bright ---")
inputs3 = [5, 3, 4, 9]  # Medium light, evening, some motion, likes bright
print(f"Inputs - Light:{inputs3[0]}, Time:{inputs3[1]}, Motion:{inputs3[2]}, Pref:{inputs3[3]}")
output3 = neuron_predict(inputs3, light_weights, light_bias, light_activation)
print(f"Decision: {'Turn ON' if output3 == 1 else 'Keep OFF'}")

# TODO: Add 2 more test scenarios of your choice
print("\n--- Your Scenario 1 ---")
# your_inputs1 = [?, ?, ?, ?]
# output = neuron_predict(your_inputs1, light_weights, light_bias, light_activation)

print("\n--- Your Scenario 2 ---")
# your_inputs2 = [?, ?, ?, ?]
# output = neuron_predict(your_inputs2, light_weights, light_bias, light_activation)
"""

# QUESTIONS TO ANSWER:
print("\n" + "="*60)
print("QUESTIONS FOR TASK 3:")
print("="*60)
print("""
1. Which input do you think should have the highest weight and why?
   Answer: 


2. Should ambient light have a positive or negative weight? Why?
   Answer: 


3. How did you handle the user preference in your design?
   Answer: 


4. What bias value makes sense for this problem?
   Answer: 


5. Test your controller with edge cases (all zeros, all tens).
   Does it behave as expected?
   Answer: 


""")


# =============================================================================
# BONUS CHALLENGE (OPTIONAL)
# =============================================================================

print("\n" + "="*60)
print("BONUS CHALLENGE: XOR PROBLEM")
print("="*60)

"""
CHALLENGE:
Can you create a SINGLE neuron that implements the XOR (exclusive OR) function?

XOR Truth Table:
  Input1 | Input2 | Output
  -------|--------|-------
    0    |   0    |   0
    0    |   1    |   1
    1    |   0    |   1
    1    |   1    |   0

Try it! Can you find weights and bias that work?

SPOILER ALERT: You can't! A single neuron can only learn linearly separable
patterns. XOR is not linearly separable. You need multiple neurons (layers)
to solve this. You'll learn this in Lab 02!
"""

# TODO: Try to implement XOR (you'll discover it's impossible with one neuron!)
"""
xor_weights = [?, ?]
xor_bias = ?
xor_activation = 'step'

test_cases = [[0, 0], [0, 1], [1, 0], [1, 1]]
expected = [0, 1, 1, 0]

print("\nAttempting XOR with single neuron:")
for i, inputs in enumerate(test_cases):
    output = neuron_predict(inputs, xor_weights, xor_bias, xor_activation)
    status = "✓" if output == expected[i] else "✗"
    print(f"Input: {inputs} → Output: {output}, Expected: {expected[i]} {status}")
"""

print("""
If you couldn't solve XOR, don't worry! This demonstrates why we need
multi-layer neural networks. Looking forward to Lab 02!
""")


# =============================================================================
# REFLECTION AND SELF-ASSESSMENT
# =============================================================================

print("\n" + "="*60)
print("REFLECTION")
print("="*60)

print("""
Answer these reflection questions:

1. What was the most challenging part of this lab?
   Answer: 


2. What concept do you understand better now?
   Answer: 


3. What concept is still unclear?
   Answer: 


4. How would you explain what a neuron does to a friend who hasn't
   taken this course?
   Answer: 


5. What questions do you have for the next lab?
   Answer: 


""")


# =============================================================================
# GRADING CHECKLIST (For Instructor)
# =============================================================================

"""
GRADING RUBRIC (Total: 100 points)

Task 1: Fruit Classifier (40 points)
  - Correct function implementations (15 points)
    * calculate_weighted_sum, apply_activation, neuron_predict
  - Reasonable weight choices (10 points)
  - Appropriate bias and activation (5 points)
  - Working test cases (5 points)
  - Thoughtful answers to questions (5 points)

Task 2: Activation Comparison (20 points)
  - Implemented comparison correctly using functions (10 points)
  - Thoughtful analysis of differences (10 points)

Task 3: Light Controller (30 points)
  - Correct implementation with 4 inputs using functions (10 points)
  - Logical weight and bias choices (10 points)
  - Working test scenarios (5 points)
  - Thoughtful answers to questions (5 points)

Reflection (10 points)
  - Completed reflection questions (10 points)

Bonus: XOR Challenge (+5 extra credit)
  - Attempted XOR implementation using functions (5 points)

TOTAL: _____ / 100 points

Comments:


"""

print("\n" + "="*60)
print("END OF STUDENT TASKS")
print("="*60)
print("\nDon't forget to save and submit this file!")
print("Good luck! 🚀")
