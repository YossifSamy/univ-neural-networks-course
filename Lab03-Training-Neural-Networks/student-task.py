"""
Lab 03: Student Task - Training Neural Networks
================================================

Complete the following tasks to practice training neural networks.

SUBMISSION INSTRUCTIONS:
1. Complete all three tasks below
2. Answer all reflection questions
3. Test your code thoroughly
4. Submit this file before the deadline

Your Name: _________________________
Student ID: _________________________
Date: _________________________

"""

import math
import random


# =============================================================================
# HELPER FUNCTIONS (You may need these)
# =============================================================================

def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + math.exp(-z))


def sigmoid_derivative(z):
    """Derivative of sigmoid"""
    s = sigmoid(z)
    return s * (1 - s)


def mse_loss(y_true, y_pred):
    """Mean Squared Error loss"""
    if not isinstance(y_true, list):
        y_true = [y_true]
        y_pred = [y_pred]
    return sum((yt - yp)**2 for yt, yp in zip(y_true, y_pred)) / len(y_true)


# =============================================================================
# TASK 1: IMPLEMENT GRADIENT DESCENT (30 points)
# =============================================================================

print("="*70)
print("TASK 1: GRADIENT DESCENT ON A SIMPLE FUNCTION")
print("="*70)

"""
SCENARIO:
Implement gradient descent to minimize the function:
    f(x) = (x - 5)² + 3

The minimum is at x = 5, where f(5) = 3.

YOUR TASK:
1. Implement the function f(x)
2. Implement the gradient (derivative) of f(x)
3. Implement gradient descent to find the minimum
4. Start from x = 0 and find the minimum
5. Experiment with different learning rates
"""

# TODO: Implement the function
def task1_function(x):
    """
    Function to minimize: f(x) = (x - 5)² + 3
    """
    # TODO: Your code here
    pass


# TODO: Implement the gradient (derivative)
def task1_gradient(x):
    """
    Gradient of f(x) = (x - 5)² + 3
    f'(x) = 2(x - 5)
    """
    # TODO: Your code here
    pass


# TODO: Implement gradient descent
def task1_gradient_descent(starting_x, learning_rate, iterations):
    """
    Perform gradient descent to minimize the function.
    
    Parameters:
        starting_x: Initial value of x
        learning_rate: Step size (alpha)
        iterations: Number of steps to take
    
    Returns:
        Tuple of (final_x, history) where history is list of (x, f(x), gradient)
    """
    x = starting_x
    history = []
    
    for i in range(iterations):
        # TODO: Calculate function value
        fx = None  # Replace with: task1_function(x)
        
        # TODO: Calculate gradient
        grad = None  # Replace with: task1_gradient(x)
        
        # TODO: Store history
        history.append((x, fx, grad))
        
        # TODO: Update x using gradient descent
        # x = x - learning_rate * grad
        
    return x, history


# TODO: Test your implementation
"""
print("\nTest 1: Learning rate = 0.1")
print("-"*70)
final_x, history = task1_gradient_descent(starting_x=0, learning_rate=0.1, iterations=50)

print(f"Starting x: 0")
print(f"Final x: {final_x:.6f}")
print(f"Final f(x): {task1_function(final_x):.6f}")
print(f"Expected: x = 5, f(x) = 3")

# Show first 5 steps
print("\nFirst 5 steps:")
for i, (x, fx, grad) in enumerate(history[:5]):
    print(f"  Step {i}: x = {x:.4f}, f(x) = {fx:.4f}, gradient = {grad:+.4f}")
"""

# TODO: Experiment with different learning rates
"""
print("\n\nExperimenting with different learning rates:")
print("="*70)

learning_rates = [0.01, 0.1, 0.5, 1.5]

for lr in learning_rates:
    final_x, history = task1_gradient_descent(0, lr, 50)
    print(f"Learning rate {lr:4.2f}: final x = {final_x:.4f}, f(x) = {task1_function(final_x):.4f}")
"""

# QUESTIONS FOR TASK 1:
print("\n" + "="*70)
print("QUESTIONS FOR TASK 1:")
print("="*70)
print("""
1. What is the derivative (gradient) of f(x) = (x - 5)² + 3?
   Show your work:
   Answer: 


2. Why do we subtract the gradient in the update rule (x = x - α×gradient)?
   Answer: 


3. What happened with learning rate 0.01? Why was it slow?
   Answer: 


4. What happened with learning rate 1.5? Did it converge smoothly?
   Answer: 


5. What learning rate worked best? Why?
   Answer: 


""")


# =============================================================================
# TASK 2: TRAIN A NEURAL NETWORK FOR XOR (40 points)
# =============================================================================

print("\n" + "="*70)
print("TASK 2: TRAIN XOR NETWORK WITH BACKPROPAGATION")
print("="*70)

"""
SCENARIO:
Implement training for an MLP to solve XOR.

YOUR TASK:
1. Copy the Layer and MLP classes from training-implementation.py
   (OR implement your own simplified version)
2. Train a [2, 2, 1] network on XOR data
3. Monitor the loss during training
4. Test the trained network on all XOR cases
5. Experiment with hyperparameters
"""

# TODO: Copy or implement the Layer class
class Layer:
    """
    Neural network layer.
    You can copy from training-implementation.py or implement simplified version.
    """
    
    def __init__(self, input_size, output_size):
        # TODO: Initialize weights and biases
        pass
    
    def forward(self, inputs):
        # TODO: Forward pass
        # Calculate: z = Σ(w×x) + b, then activation = sigmoid(z)
        pass
    
    def backward(self, d_activations, learning_rate):
        # TODO: Backward pass
        # Compute gradients and update weights
        # Return gradient for previous layer
        pass


# TODO: Copy or implement the MLP class
class MLP:
    """
    Multi-Layer Perceptron.
    You can copy from training-implementation.py or implement simplified version.
    """
    
    def __init__(self, layer_sizes):
        # TODO: Create layers
        pass
    
    def predict(self, inputs):
        # TODO: Forward pass through all layers
        pass
    
    def train_step(self, inputs, target, learning_rate):
        # TODO: One training step (forward + backward)
        # Return loss
        pass
    
    def train(self, X, y, epochs, learning_rate):
        # TODO: Train on dataset for multiple epochs
        # Return loss history
        pass


# XOR training data
X_xor = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_xor = [0, 1, 1, 0]

# TODO: Create and train network
"""
print("\nCreating network: [2, 2, 1]")
network = MLP([2, 2, 1])

print("\nBefore training:")
for i, inputs in enumerate(X_xor):
    output = network.predict(inputs)[0]
    prediction = 1 if output >= 0.5 else 0
    print(f"  Input: {inputs} → Output: {output:.4f} → Prediction: {prediction} (Expected: {y_xor[i]})")

print("\nTraining...")
loss_history = network.train(X_xor, y_xor, epochs=1000, learning_rate=0.5)

print("\nAfter training:")
for i, inputs in enumerate(X_xor):
    output = network.predict(inputs)[0]
    prediction = 1 if output >= 0.5 else 0
    status = "✓" if prediction == y_xor[i] else "✗"
    print(f"  Input: {inputs} → Output: {output:.4f} → Prediction: {prediction} (Expected: {y_xor[i]}) {status}")

print(f"\nFinal loss: {loss_history[-1]:.6f}")
"""

# TODO: Plot or display loss over time
"""
print("\nLoss at different epochs:")
print("-"*70)
milestones = [0, 100, 200, 500, 999]
for epoch in milestones:
    if epoch < len(loss_history):
        print(f"Epoch {epoch:4d}: Loss = {loss_history[epoch]:.6f}")
"""

# QUESTIONS FOR TASK 2:
print("\n" + "="*70)
print("QUESTIONS FOR TASK 2:")
print("="*70)
print("""
1. Did your network successfully learn XOR? How do you know?
   Answer: 


2. How did the loss change during training? Describe the pattern.
   Answer: 


3. What would happen if you used only 1 hidden neuron instead of 2?
   (Try it if you can!)
   Answer: 


4. Explain in your own words how backpropagation works.
   Answer: 


5. What role does the learning rate play in training?
   Answer: 


""")


# =============================================================================
# TASK 3: HYPERPARAMETER EXPERIMENTS (30 points)
# =============================================================================

print("\n" + "="*70)
print("TASK 3: HYPERPARAMETER EXPERIMENTS")
print("="*70)

"""
SCENARIO:
Experiment with different hyperparameters and observe their effects.

YOUR TASK:
1. Train XOR with 3 different learning rates
2. Train XOR with 3 different architectures
3. Compare training speed and final accuracy
4. Document your findings
"""

# TODO: Experiment 1 - Different Learning Rates
"""
print("\nExperiment 1: Different Learning Rates")
print("="*70)

learning_rates = [0.1, 0.5, 2.0]

for lr in learning_rates:
    print(f"\nLearning Rate = {lr}")
    print("-"*70)
    
    # Create fresh network
    net = MLP([2, 2, 1])
    
    # Train
    losses = net.train(X_xor, y_xor, epochs=500, learning_rate=lr)
    
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
    print(f"Epochs to reach loss < 0.1: ...")  # TODO: Calculate this
"""

# TODO: Experiment 2 - Different Architectures
"""
print("\n\nExperiment 2: Different Network Architectures")
print("="*70)

architectures = [
    ([2, 2, 1], "Minimal"),
    ([2, 4, 1], "More hidden neurons"),
    ([2, 3, 3, 1], "Two hidden layers"),
]

for arch, desc in architectures:
    print(f"\n{desc}: {arch}")
    print("-"*70)
    
    # Create and train network
    net = MLP(arch)
    losses = net.train(X_xor, y_xor, epochs=500, learning_rate=0.5)
    
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
"""

# TODO: Experiment 3 - Your Own Experiment!
"""
Design and run your own experiment. Some ideas:
- Different activation functions
- Different number of epochs
- Different initialization methods
- Training with noisy data

Document your experiment below:
"""

print("\n\nExperiment 3: Custom Experiment")
print("="*70)
print("""
Describe your experiment:


Results:


Insights:


""")

# QUESTIONS FOR TASK 3:
print("\n" + "="*70)
print("QUESTIONS FOR TASK 3:")
print("="*70)
print("""
1. Which learning rate worked best? Why do you think that is?
   Answer: 


2. What happened when learning rate was too large?
   Answer: 


3. Did more hidden neurons always help? What trade-offs did you observe?
   Answer: 


4. Compare training speed (epochs to converge) across different architectures:
   Answer: 


5. Based on your experiments, what advice would you give someone
   training a neural network for the first time?
   Answer: 


""")


# =============================================================================
# BONUS CHALLENGE (OPTIONAL - 10 extra credit points)
# =============================================================================

print("\n" + "="*70)
print("BONUS CHALLENGE: TRAIN ON A NEW PROBLEM")
print("="*70)

"""
CHALLENGE:
Train a network to learn a different problem of your choice!

Ideas:
1. AND, OR, NAND gates (easier than XOR)
2. Function approximation (e.g., f(x) = sin(x))
3. Simple classification (e.g., classify points above/below y=x)
4. Multi-output problem

Document your bonus work below:
"""

# TODO: Bonus challenge implementation

print("""
Bonus Problem Description:


Architecture Used:


Training Results:


Analysis:


""")


# =============================================================================
# REFLECTION AND SELF-ASSESSMENT
# =============================================================================

print("\n" + "="*70)
print("REFLECTION")
print("="*70)

print("""
Answer these reflection questions:

1. What was the most challenging part of implementing training?
   Answer: 


2. What surprised you most about how neural networks learn?
   Answer: 


3. How does backpropagation use the chain rule from calculus?
   Answer: 


4. Why is training called "learning"? What is the network learning?
   Answer: 


5. What's the difference between gradient descent and backpropagation?
   Answer: 


6. If you were to explain neural network training to a friend who
   hasn't taken this course, how would you explain it?
   Answer: 


7. What concepts from this lab do you want to explore further?
   Answer: 


8. How confident do you feel about training neural networks now?
   (1 = not confident, 10 = very confident): _____
   
   What would help you feel more confident?
   Answer: 


""")


# =============================================================================
# GRADING CHECKLIST (For Instructor)
# =============================================================================

"""
GRADING RUBRIC (Total: 100 points + 10 bonus)

Task 1: Gradient Descent (30 points)
  - Correct function implementation (5 points)
  - Correct gradient implementation (5 points)
  - Working gradient descent algorithm (10 points)
  - Successful minimization (5 points)
  - Thoughtful answers to questions (5 points)

Task 2: Train XOR Network (40 points)
  - Layer class implementation (10 points)
  - MLP class implementation (10 points)
  - Successful XOR training (10 points)
  - Loss monitoring and analysis (5 points)
  - Thoughtful answers to questions (5 points)

Task 3: Hyperparameter Experiments (30 points)
  - Learning rate experiments (10 points)
  - Architecture experiments (10 points)
  - Custom experiment (5 points)
  - Thoughtful analysis and comparison (5 points)

Reflection (10 points from overall weight distribution)
  - Completed all reflection questions (10 points)
  - Demonstrates understanding of concepts

Bonus Challenge (+10 extra credit)
  - Novel problem implementation (5 points)
  - Successful training (3 points)
  - Thoughtful analysis (2 points)

CODE QUALITY (considered in grading):
  - Clean, readable code
  - Helpful comments
  - Proper testing
  - Error handling

TOTAL: _____ / 100 points (+_____ bonus)

Comments:




Specific Strengths:


Areas for Improvement:


"""

print("\n" + "="*70)
print("END OF STUDENT TASKS")
print("="*70)
print("\nDon't forget to:")
print("  1. Test all your code")
print("  2. Answer all questions")
print("  3. Complete the reflection")
print("  4. Save and submit this file")
print("\nGood luck! You're learning the heart of AI! 🚀🧠")
