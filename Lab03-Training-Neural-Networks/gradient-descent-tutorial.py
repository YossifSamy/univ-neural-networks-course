"""
Lab 03: Gradient Descent Tutorial
===================================

This file teaches optimization concepts using simple examples
BEFORE applying them to neural networks.

Learn gradient descent by:
1. Minimizing simple math functions
2. Understanding learning rates
3. Visualizing the descent process
4. Building intuition for neural network training

Follow along with your instructor!

Author: Neural Networks Course
Lab: 03 - Training Neural Networks
"""

import math


# =============================================================================
# PART 1: THE OPTIMIZATION PROBLEM
# =============================================================================

print("="*70)
print(" "*15 + "PART 1: THE OPTIMIZATION PROBLEM")
print("="*70)

print("""
THE CHALLENGE:

You have a function f(x) and want to find the value of x that gives
the MINIMUM value of f(x).

Think of it like finding the lowest point in a valley!

Example function: f(x) = (x - 3)²

Let's try different x values and see f(x):
""")

def simple_function(x):
    """
    A simple function to minimize: f(x) = (x - 3)²
    
    The minimum is at x = 3, where f(3) = 0
    """
    return (x - 3) ** 2


print("\n" + "-"*70)
print("Testing different x values:")
print("-"*70)

test_values = [0, 1, 2, 3, 4, 5, 6]

for x in test_values:
    fx = simple_function(x)
    print(f"x = {x}  →  f(x) = {fx}")

print("""
Notice: 
- f(x) is smallest when x = 3
- f(x) gets larger as x moves away from 3
- The function forms a "bowl" shape

QUESTION: What if we couldn't try all values? What if x could be any
real number? How do we find the minimum efficiently?

ANSWER: GRADIENT DESCENT!
""")


# =============================================================================
# PART 2: UNDERSTANDING GRADIENTS (SLOPES)
# =============================================================================

print("\n" + "="*70)
print(" "*20 + "PART 2: GRADIENTS (SLOPES)")
print("="*70)

print("""
GRADIENT = SLOPE = RATE OF CHANGE

The gradient tells us how the function changes when we change x slightly.

For f(x) = (x - 3)²:
The derivative (gradient) is: f'(x) = 2(x - 3)

Let's calculate gradients at different points:
""")

def gradient_simple_function(x):
    """
    Gradient (derivative) of f(x) = (x - 3)²
    f'(x) = 2(x - 3)
    """
    return 2 * (x - 3)


print("\n" + "-"*70)
print("Gradients at different points:")
print("-"*70)

for x in test_values:
    grad = gradient_simple_function(x)
    direction = "↓ downhill to the right" if grad < 0 else "↑ downhill to the left" if grad > 0 else "→ at minimum!"
    print(f"x = {x}  →  gradient = {grad:+.1f}  {direction}")

print("""
KEY OBSERVATIONS:

1. When gradient is NEGATIVE (x < 3):
   - Function slopes downward to the right
   - We should INCREASE x to reach minimum
   
2. When gradient is POSITIVE (x > 3):
   - Function slopes downward to the left
   - We should DECREASE x to reach minimum
   
3. When gradient is ZERO (x = 3):
   - We're at the minimum!
   - Function is flat here

THE RULE: Move in the OPPOSITE direction of the gradient!
- Positive gradient → decrease x
- Negative gradient → increase x
- This is why we use: x_new = x_old - gradient
""")


# =============================================================================
# PART 3: GRADIENT DESCENT ALGORITHM
# =============================================================================

print("\n" + "="*70)
print(" "*18 + "PART 3: GRADIENT DESCENT ALGORITHM")
print("="*70)

print("""
THE GRADIENT DESCENT ALGORITHM:

1. Start at some random point x
2. Calculate the gradient (slope) at that point
3. Take a step in the opposite direction of the gradient
4. Repeat until we reach the minimum!

UPDATE RULE:
x_new = x_old - learning_rate × gradient

The learning_rate controls how big of a step we take.
""")

def gradient_descent_simple(starting_x, learning_rate, iterations):
    """
    Perform gradient descent to minimize f(x) = (x - 3)²
    
    Parameters:
        starting_x: Initial guess for x
        learning_rate: Step size (alpha)
        iterations: Number of steps to take
    
    Returns:
        List of (x, f(x), gradient) at each step
    """
    x = starting_x
    history = []
    
    for i in range(iterations):
        # Calculate function value and gradient
        fx = simple_function(x)
        grad = gradient_simple_function(x)
        
        # Store history
        history.append((x, fx, grad))
        
        # Update x using gradient descent
        x = x - learning_rate * grad
    
    return history


# =============================================================================
# PART 4: EXAMPLE 1 - GRADIENT DESCENT IN ACTION
# =============================================================================

print("\n" + "="*70)
print(" "*15 + "EXAMPLE 1: GRADIENT DESCENT IN ACTION")
print("="*70)

print("""
Let's start at x = 0 and find the minimum!

Function: f(x) = (x - 3)²
Starting point: x = 0
Learning rate: 0.1
""")

history = gradient_descent_simple(starting_x=0, learning_rate=0.1, iterations=20)

print("\n" + "-"*70)
print("Step |    x     |   f(x)   | Gradient | Change")
print("-"*70)

for i, (x, fx, grad) in enumerate(history[:15]):  # Show first 15 steps
    if i > 0:
        prev_x = history[i-1][0]
        change = x - prev_x
        print(f" {i:2d}  | {x:8.4f} | {fx:8.4f} | {grad:+8.4f} | {change:+8.4f}")
    else:
        print(f" {i:2d}  | {x:8.4f} | {fx:8.4f} | {grad:+8.4f} |    ---")

final_x = history[-1][0]
final_fx = history[-1][1]

print("-"*70)
print(f"\nFinal result after {len(history)} steps:")
print(f"  x = {final_x:.6f}")
print(f"  f(x) = {final_fx:.6f}")
print(f"  True minimum: x = 3, f(x) = 0")
print(f"\n✓ Gradient descent found the minimum!")


# =============================================================================
# PART 5: EFFECT OF LEARNING RATE
# =============================================================================

print("\n\n" + "="*70)
print(" "*18 + "PART 5: EFFECT OF LEARNING RATE")
print("="*70)

print("""
The learning rate (step size) dramatically affects gradient descent!

Let's try three different learning rates:
1. Too small (0.01)
2. Just right (0.1)  
3. Too large (1.5)
""")

learning_rates = [0.01, 0.1, 1.5]
iterations = 20

print("\n" + "-"*70)

for lr in learning_rates:
    history = gradient_descent_simple(starting_x=0, learning_rate=lr, iterations=iterations)
    final_x = history[-1][0]
    final_fx = history[-1][1]
    
    print(f"\nLearning Rate = {lr}")
    print(f"  After {iterations} steps: x = {final_x:.4f}, f(x) = {final_fx:.4f}")
    
    # Show first few steps
    print(f"  Progress: ", end="")
    for i, (x, fx, grad) in enumerate(history[:min(10, len(history))]):
        if i < 5:
            print(f"{x:.2f} → ", end="")
    print("...")
    
    # Analysis
    if lr == 0.01:
        print("  Analysis: Too slow! Barely made progress. 🐌")
    elif lr == 0.1:
        print("  Analysis: Perfect! Converged efficiently. ✓")
    elif lr == 1.5:
        print("  Analysis: Too fast! Overshot and oscillated. 🎢")

print("\n" + "-"*70)
print("""
KEY INSIGHTS:

Learning Rate TOO SMALL (0.01):
  - Very stable, won't diverge
  - But takes forever to reach minimum
  - Like walking very slowly down a mountain
  
Learning Rate JUST RIGHT (0.1):
  - Fast convergence
  - Stable progress
  - Sweet spot! ✓
  
Learning Rate TOO LARGE (1.5):
  - Takes big steps
  - Overshoots the minimum
  - Bounces back and forth
  - Like taking giant leaps - you jump over the valley!
""")


# =============================================================================
# PART 6: MORE COMPLEX FUNCTION
# =============================================================================

print("\n\n" + "="*70)
print(" "*18 + "PART 6: MORE COMPLEX FUNCTION")
print("="*70)

print("""
Let's try a more complex function with multiple terms:

f(x) = x⁴ - 3x³ + 2x

This function has multiple minima (local and global).
""")

def complex_function(x):
    """
    More complex function: f(x) = x⁴ - 3x³ + 2x
    Has multiple local minima
    """
    return x**4 - 3*x**3 + 2*x


def gradient_complex_function(x):
    """
    Gradient of f(x) = x⁴ - 3x³ + 2x
    f'(x) = 4x³ - 9x² + 2
    """
    return 4*x**3 - 9*x**2 + 2


def gradient_descent_complex(starting_x, learning_rate, iterations):
    """Gradient descent for the complex function"""
    x = starting_x
    history = []
    
    for i in range(iterations):
        fx = complex_function(x)
        grad = gradient_complex_function(x)
        history.append((x, fx, grad))
        
        # Gradient descent update
        x = x - learning_rate * grad
    
    return history


print("\nTrying different starting points:")
print("-"*70)

starting_points = [-1, 0, 1, 3]

for start in starting_points:
    history = gradient_descent_complex(starting_x=start, learning_rate=0.01, iterations=100)
    final_x = history[-1][0]
    final_fx = history[-1][1]
    
    print(f"Starting at x = {start:+.1f}  →  Ended at x = {final_x:+.4f}, f(x) = {final_fx:+.6f}")

print("""
OBSERVATION:

Different starting points can lead to different minima!
This is called "local minima" vs "global minimum" problem.

In neural networks:
- We use random initialization
- Sometimes we get stuck in local minima
- Training multiple times with different initializations can help
""")


# =============================================================================
# PART 7: GRADIENT DESCENT FOR 2D FUNCTIONS
# =============================================================================

print("\n\n" + "="*70)
print(" "*15 + "PART 7: GRADIENT DESCENT IN 2D (TWO VARIABLES)")
print("="*70)

print("""
Neural networks have MANY weights, not just one!
Let's practice with a function of TWO variables:

f(x, y) = (x - 2)² + (y + 1)²

Minimum is at x = 2, y = -1
""")

def function_2d(x, y):
    """
    2D function: f(x, y) = (x - 2)² + (y + 1)²
    Minimum at (2, -1)
    """
    return (x - 2)**2 + (y + 1)**2


def gradient_2d(x, y):
    """
    Gradients for 2D function
    ∂f/∂x = 2(x - 2)
    ∂f/∂y = 2(y + 1)
    """
    grad_x = 2 * (x - 2)
    grad_y = 2 * (y + 1)
    return grad_x, grad_y


def gradient_descent_2d(start_x, start_y, learning_rate, iterations):
    """Gradient descent for 2D function"""
    x, y = start_x, start_y
    history = []
    
    for i in range(iterations):
        f_val = function_2d(x, y)
        grad_x, grad_y = gradient_2d(x, y)
        
        history.append((x, y, f_val, grad_x, grad_y))
        
        # Update both variables
        x = x - learning_rate * grad_x
        y = y - learning_rate * grad_y
    
    return history


print("\nStarting at (x, y) = (0, 0)")
print("Learning rate = 0.1")
print("\n" + "-"*70)
print("Step |   x    |   y    |  f(x,y)  | grad_x | grad_y")
print("-"*70)

history = gradient_descent_2d(start_x=0, start_y=0, learning_rate=0.1, iterations=15)

for i, (x, y, f_val, gx, gy) in enumerate(history[:10]):
    print(f" {i:2d}  | {x:6.3f} | {y:6.3f} | {f_val:8.4f} | {gx:+7.3f} | {gy:+7.3f}")

final_x, final_y, final_f = history[-1][0], history[-1][1], history[-1][2]
print("-"*70)
print(f"\nFinal result: (x, y) = ({final_x:.4f}, {final_y:.4f})")
print(f"f(x, y) = {final_f:.6f}")
print(f"True minimum: (x, y) = (2, -1), f(x, y) = 0")
print("\n✓ Successfully minimized 2D function!")

print("""
NEURAL NETWORK CONNECTION:

In neural networks:
- Each weight is like a variable (x, y, etc.)
- We have thousands or millions of weights!
- Gradient descent updates ALL weights simultaneously
- Same principle, just many more dimensions!
""")


# =============================================================================
# PART 8: CONVERGENCE CRITERIA
# =============================================================================

print("\n\n" + "="*70)
print(" "*20 + "PART 8: WHEN TO STOP?")
print("="*70)

print("""
CONVERGENCE: When gradient descent has found the minimum (or close enough)

Common stopping criteria:

1. MAXIMUM ITERATIONS: Stop after N steps
   - Simple, predictable
   - Might stop too early or waste time

2. GRADIENT MAGNITUDE: Stop when gradient is very small
   - When gradient ≈ 0, we're at a minimum
   - Example: |gradient| < 0.001

3. FUNCTION CHANGE: Stop when f(x) stops changing
   - Example: |f(x_new) - f(x_old)| < 0.0001
   - Means we're not making progress anymore

Let's implement criterion 2 (gradient magnitude):
""")

def gradient_descent_with_convergence(starting_x, learning_rate, tolerance=0.001, max_iterations=1000):
    """
    Gradient descent with convergence criterion
    Stops when gradient magnitude < tolerance
    """
    x = starting_x
    history = []
    
    for i in range(max_iterations):
        fx = simple_function(x)
        grad = gradient_simple_function(x)
        history.append((x, fx, grad))
        
        # Check convergence
        if abs(grad) < tolerance:
            print(f"  Converged after {i+1} iterations!")
            print(f"  Gradient magnitude: {abs(grad):.6f} < {tolerance}")
            break
        
        # Update
        x = x - learning_rate * grad
    else:
        print(f"  Reached maximum iterations ({max_iterations})")
    
    return history


print("\nGradient descent with automatic stopping:")
print("-"*70)

history = gradient_descent_with_convergence(starting_x=0, learning_rate=0.1, tolerance=0.001)

final_x, final_fx, final_grad = history[-1]
print(f"  Final x: {final_x:.6f}")
print(f"  Final f(x): {final_fx:.6f}")
print(f"  Final gradient: {final_grad:.6f}")


# =============================================================================
# PART 9: KEY TAKEAWAYS
# =============================================================================

print("\n\n" + "="*70)
print(" "*25 + "KEY TAKEAWAYS")
print("="*70)

print("""
1. GRADIENT DESCENT finds minimums by following the slope downhill
   - Like hiking down a mountain in fog
   - Feel the slope (gradient) and step downhill

2. UPDATE RULE: x_new = x_old - learning_rate × gradient
   - Gradient tells us direction (uphill/downhill)
   - Learning rate controls step size
   - We move OPPOSITE to the gradient (downhill!)

3. LEARNING RATE is critical:
   - Too small → slow convergence
   - Too large → unstable, overshooting
   - Just right → efficient convergence
   - Usually requires experimentation

4. WORKS IN MULTIPLE DIMENSIONS:
   - Same algorithm for 1, 2, or 1000 variables
   - Update each variable using its gradient
   - Neural networks: each weight is a variable!

5. CONVERGENCE CRITERIA:
   - Need to know when to stop
   - Gradient ≈ 0 means we're at minimum
   - Or function value stops changing

6. STARTING POINT MATTERS:
   - Can get stuck in local minima
   - Random initialization helps
   - In neural networks, we use special initialization

NEXT STEP: Apply this to NEURAL NETWORKS!
- Instead of f(x), minimize Loss(weights)
- Use backpropagation to compute gradients
- Update all weights using gradient descent
- This is how neural networks learn! 🚀
""")

print("\n" + "="*70)
print("END OF GRADIENT DESCENT TUTORIAL")
print("="*70)
print("\nYou now understand the optimization foundation of neural networks!")
print("Ready to train networks in training-implementation.py!")
