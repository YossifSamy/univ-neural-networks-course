"""
Lab 02: Student Task - Multi-Layer Perceptron
==============================================

Complete the following tasks to practice MLPs and OOP.

SUBMISSION INSTRUCTIONS:
1. Complete all three tasks below
2. Test your code thoroughly
3. Answer all questions
4. Submit this file before the end of lab

Your Name: _________________________
Student ID: _________________________
Date: _________________________

"""

import math
import random


# =============================================================================
# STARTER CODE - COPY FROM mlp-implementation.py
# =============================================================================

class Layer:
    """
    Represents one layer in the neural network.
    (Copy your implementation from mlp-implementation.py or implement from scratch)
    """
    
    def __init__(self, num_inputs, num_neurons, activation='sigmoid'):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.activation = activation
        
        # TODO: Initialize weights and biases
        self.weights = []  # Replace with your implementation
        self.biases = []   # Replace with your implementation
    
    def forward(self, inputs):
        """Forward propagation through this layer."""
        # TODO: Implement forward propagation
        pass


class MLP:
    """
    Multi-Layer Perceptron.
    (Copy your implementation from mlp-implementation.py or implement from scratch)
    """
    
    def __init__(self, architecture, activation='sigmoid'):
        # TODO: Implement __init__
        pass
    
    def forward(self, inputs):
        """Forward propagation through entire network."""
        # TODO: Implement forward propagation
        pass
    
    def predict(self, inputs):
        """Make a prediction."""
        # TODO: Implement predict
        pass


# =============================================================================
# TASK 1: HANDWRITTEN DIGIT RECOGNITION NETWORK
# =============================================================================

print("=" * 70)
print("TASK 1: HANDWRITTEN DIGIT RECOGNITION NETWORK")
print("=" * 70)

"""
SCENARIO:
You're building a system to recognize handwritten digits (0-9) from images.

INPUT:
- 28×28 pixel grayscale images
- Each pixel has value 0 (white) to 255 (black)
- Total: 28 × 28 = 784 input features

OUTPUT:
- 10 classes (digits 0-9)
- One output neuron per digit
- Highest output = predicted digit

YOUR TASK:
1. Design an appropriate network architecture
2. Justify your design choices
3. Implement the network
4. Calculate total parameters
5. Test with sample data
"""

# TODO: Design your architecture
print("\n--- Architecture Design ---")

# Example: architecture = [784, ?, ?, 10]
# Fill in the hidden layers!

digit_architecture = None  # TODO: Replace with your architecture list

print(f"Proposed Architecture: {digit_architecture}")

# TODO: Answer these questions
print("""
DESIGN QUESTIONS:

1. What architecture did you choose? (e.g., [784, 128, 64, 10])
   Answer: 


2. Why did you choose this number of hidden layers?
   Answer: 


3. Why did you choose this number of neurons in each hidden layer?
   Answer: 


4. Calculate the total number of parameters in your network:
   
   Layer 1: inputs × neurons + biases = ___ × ___ + ___ = ___
   Layer 2: ___ × ___ + ___ = ___
   Layer 3: ___ × ___ + ___ = ___
   
   TOTAL PARAMETERS: ___________


5. What activation function would you use for hidden layers? Why?
   Answer: 


6. What activation function would you use for output layer? Why?
   (Hint: We want probabilities for 10 classes)
   Answer: 

""")

# TODO: Implement your network
"""
digit_network = MLP(digit_architecture, activation='relu')

# Simulate a sample 28x28 image (random pixels)
sample_image = [random.uniform(0, 1) for _ in range(784)]

print("\n--- Testing Digit Recognition Network ---")
print("Input: 784 pixel values (simulated handwritten digit)")

outputs = digit_network.predict(sample_image)
predicted_digit = outputs.index(max(outputs))

print(f"Network outputs: {[f'{x:.3f}' for x in outputs]}")
print(f"Predicted digit: {predicted_digit}")
print(f"Confidence: {max(outputs):.3f}")

print("\nNote: This is with random weights. After training, accuracy would improve!")
"""


# =============================================================================
# TASK 2: ARCHITECTURE EXPERIMENTS
# =============================================================================

print("\n\n" + "=" * 70)
print("TASK 2: COMPARING DIFFERENT ARCHITECTURES")
print("=" * 70)

"""
TASK:
Experiment with different network architectures for the same problem.
Compare: shallow vs deep, narrow vs wide networks.

Problem: Binary classification with 20 input features
"""

print("\n--- Experiment Setup ---")
print("Problem: Binary classification")
print("Inputs: 20 features")
print("Output: 1 neuron (yes/no decision)")

# TODO: Design three different architectures

# Architecture 1: Shallow and Wide
arch1 = None  # Example: [20, 50, 1]

# Architecture 2: Deep and Narrow
arch2 = None  # Example: [20, 10, 10, 10, 1]

# Architecture 3: Balanced
arch3 = None  # Example: [20, 15, 10, 5, 1]

print(f"\nArchitecture 1 (Shallow & Wide): {arch1}")
print(f"Architecture 2 (Deep & Narrow): {arch2}")
print(f"Architecture 3 (Balanced): {arch3}")

# TODO: Implement and compare
"""
print("\n--- Creating Networks ---")

net1 = MLP(arch1, activation='relu')
net2 = MLP(arch2, activation='relu')
net3 = MLP(arch3, activation='relu')

# Test with sample data
test_input = [random.uniform(-1, 1) for _ in range(20)]

print("\n--- Testing All Architectures ---")
print(f"Test input: 20 random values")

output1 = net1.predict(test_input)
output2 = net2.predict(test_input)
output3 = net3.predict(test_input)

print(f"\nArchitecture 1 output: {output1[0]:.4f}")
print(f"Architecture 2 output: {output2[0]:.4f}")
print(f"Architecture 3 output: {output3[0]:.4f}")
"""

# TODO: Calculate parameters for each
print("\n--- Parameter Count ---")

# TODO: Fill in the calculations
"""
def count_parameters(architecture):
    total = 0
    for i in range(len(architecture) - 1):
        weights = architecture[i] * architecture[i + 1]
        biases = architecture[i + 1]
        total += weights + biases
    return total

params1 = count_parameters(arch1)
params2 = count_parameters(arch2)
params3 = count_parameters(arch3)

print(f"Architecture 1: {params1} parameters")
print(f"Architecture 2: {params2} parameters")
print(f"Architecture 3: {params3} parameters")
"""

# ANALYSIS QUESTIONS
print("\n" + "=" * 60)
print("ANALYSIS QUESTIONS:")
print("=" * 60)
print("""
1. Which architecture has the most parameters?
   Answer: 


2. Which architecture would likely train fastest? Why?
   Answer: 


3. Which architecture might learn more complex patterns? Why?
   Answer: 


4. What are advantages of DEEP networks (many layers)?
   Answer: 


5. What are advantages of WIDE networks (many neurons per layer)?
   Answer: 


6. What are disadvantages of having too many parameters?
   Answer: 


7. How would you decide which architecture to use in practice?
   Answer: 

""")


# =============================================================================
# TASK 3: GAME AI - TIC-TAC-TOE MOVE PREDICTOR
# =============================================================================

print("\n\n" + "=" * 70)
print("TASK 3: TIC-TAC-TOE MOVE PREDICTOR")
print("=" * 70)

"""
SCENARIO:
Build an AI to play Tic-Tac-Toe! The network predicts the best move.

INPUT:
- 9 features (one per board position)
- Values: -1 (opponent), 0 (empty), +1 (player)

OUTPUT:
- 9 outputs (one per board position)
- Highest output = best move to play

EXAMPLE BOARD:
  X | O | X     →  [+1, -1, +1,
  - | X | -          0, +1,  0,
  O | - | -         -1,  0,  0]

YOUR TASK:
1. Design appropriate architecture
2. Implement the network
3. Test with sample game states
"""

print("\n--- Task 3: Tic-Tac-Toe Architecture ---")

# TODO: Design your architecture
tictactoe_architecture = None  # Example: [9, ?, ?, 9]

print(f"Tic-Tac-Toe Architecture: {tictactoe_architecture}")

# TODO: Answer design questions
print("""
DESIGN QUESTIONS:

1. What architecture did you choose?
   Answer: 


2. Why is the output layer size 9?
   Answer: 


3. How many hidden layers did you use and why?
   Answer: 


4. What activation function for hidden layers?
   Answer: 


5. What activation function for output layer?
   (Should outputs be bounded? Should they sum to 1?)
   Answer: 

""")

# TODO: Implement and test
"""
print("\n--- Creating Tic-Tac-Toe Network ---")
tictactoe_net = MLP(tictactoe_architecture, activation='relu')

# Test Game State 1: Empty board
print("\n--- Test 1: Empty Board ---")
empty_board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
print("Board: All empty")
print("  - | - | -")
print("  - | - | -")
print("  - | - | -")

outputs = tictactoe_net.predict(empty_board)
best_move = outputs.index(max(outputs))
best_position = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)][best_move]

print(f"\nNetwork outputs: {[f'{x:.2f}' for x in outputs]}")
print(f"Best move: Position {best_move} → Row {best_position[0]}, Col {best_position[1]}")

# Test Game State 2: Mid-game
print("\n--- Test 2: Mid-Game Position ---")
mid_game = [1, -1, 1, 0, 1, 0, -1, 0, 0]
print("Board state:")
print("  X | O | X")
print("  - | X | -")
print("  O | - | -")

outputs = tictactoe_net.predict(mid_game)
best_move = outputs.index(max(outputs))

print(f"\nNetwork outputs: {[f'{x:.2f}' for x in outputs]}")
print(f"Best move: Position {best_move}")

# Filter only valid moves (empty positions)
valid_moves = [i for i, val in enumerate(mid_game) if val == 0]
valid_outputs = {i: outputs[i] for i in valid_moves}
best_valid_move = max(valid_outputs, key=valid_outputs.get)

print(f"Valid moves: {valid_moves}")
print(f"Best valid move: Position {best_valid_move}")

print("\nNote: With random weights, moves are random!")
print("After training, the network would learn strategic moves!")
"""


# =============================================================================
# BONUS CHALLENGE: ADD SOFTMAX OUTPUT LAYER
# =============================================================================

print("\n\n" + "=" * 70)
print("BONUS CHALLENGE: IMPLEMENT SOFTMAX ACTIVATION")
print("=" * 70)

"""
CHALLENGE:
Add a Softmax activation function to your Layer class.

Softmax converts outputs to probabilities:
- All outputs between 0 and 1
- All outputs sum to 1
- Useful for multi-class classification

Formula:
softmax(z_i) = e^(z_i) / sum(e^(z_j) for all j)

Example:
Inputs:  [2.0, 1.0, 0.1]
Softmax: [0.659, 0.242, 0.099]  (sum = 1.0)
"""

# TODO: Implement softmax activation
"""
def softmax(z_values):
    '''
    Compute softmax activation.
    
    Parameters:
        z_values (list): Net inputs before activation
    
    Returns:
        list: Softmax probabilities (sum to 1)
    '''
    # TODO: Implement softmax
    # Hint: Use math.exp()
    pass

# Test softmax
test_values = [2.0, 1.0, 0.1]
softmax_output = softmax(test_values)
print(f"\nInput: {test_values}")
print(f"Softmax: {[f'{x:.3f}' for x in softmax_output]}")
print(f"Sum: {sum(softmax_output):.3f} (should be 1.0)")
"""


# =============================================================================
# REFLECTION AND SELF-ASSESSMENT
# =============================================================================

print("\n\n" + "=" * 70)
print("REFLECTION")
print("=" * 70)

print("""
Answer these reflection questions:

1. What's the main advantage of using OOP for neural networks?
   Answer: 


2. How does an MLP differ from a single neuron?
   Answer: 


3. What problems can MLPs solve that single neurons cannot?
   Answer: 


4. How do you decide on network architecture (layers, neurons)?
   Answer: 


5. What's the purpose of hidden layers?
   Answer: 


6. Why do we need activation functions in hidden layers?
   Answer: 


7. What was the most challenging part of this lab?
   Answer: 


8. What concept do you understand better now?
   Answer: 


9. What are you most excited to learn in the next lab?
   Answer: 


10. Can you explain OOP to someone who hasn't taken this course?
    Answer: 

""")


# =============================================================================
# GRADING RUBRIC (For Instructor)
# =============================================================================

"""
GRADING RUBRIC (Total: 100 points)

Task 1: Digit Recognition Network (35 points)
  - Appropriate architecture design (10 points)
  - Correct parameter calculations (5 points)
  - Thoughtful justification (10 points)
  - Working implementation (10 points)

Task 2: Architecture Experiments (30 points)
  - Three distinct architectures (10 points)
  - Correct parameter counting (5 points)
  - Comparative analysis (10 points)
  - Thoughtful answers to questions (5 points)

Task 3: Tic-Tac-Toe AI (25 points)
  - Appropriate architecture (5 points)
  - Correct implementation (10 points)
  - Working test cases (5 points)
  - Thoughtful design justification (5 points)

Reflection (10 points)
  - Complete and thoughtful answers (10 points)

Bonus: Softmax Implementation (+5 extra credit)
  - Correct softmax implementation (5 points)

TOTAL: _____ / 100 points

Comments:


"""

print("\n" + "=" * 70)
print("END OF STUDENT TASKS")
print("=" * 70)
print("\nRemember:")
print("1. Complete all tasks")
print("2. Answer all questions thoughtfully")
print("3. Test your code thoroughly")
print("4. Ask questions if confused!")
print("\nGood luck! You're building the future of AI! 🚀")
