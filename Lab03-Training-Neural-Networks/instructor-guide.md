# Lab 03: Training Neural Networks - Instructor Guide

## Computer Engineering - Neural Networks Course

---

## 📋 Lab Overview

**Duration:** 4 hours  
**Difficulty:** Advanced  
**Prerequisites:** Lab 01 (Single Neuron), Lab 02 (Multi-Layer Perceptron)

**Learning Objectives:**

- Understand how neural networks learn from data
- Explain and implement backpropagation algorithm
- Implement gradient descent optimization
- Calculate and minimize loss functions
- Train networks to solve real problems

**Key Concepts:**

- Loss/Cost functions (MSE, Cross-Entropy)
- Gradient descent optimization
- Backpropagation algorithm
- Learning rate and epochs
- Training vs testing data

---

## 🎯 Teaching Strategy

### The Big Picture

**Up until now:** Students have manually set weights and biases.

**Lab 03's revelation:** Networks can **learn** optimal weights automatically from data!

This is the "magic" moment where students transition from:

- "I set the weights" → "The network learns the weights"
- "I solve the problem" → "The network solves the problem"

### Core Analogy: Climbing Down a Mountain (Blindfolded)

This lab's main analogy is **gradient descent as hiking down a mountain in fog**:

- **Mountain:** The error/loss surface
- **Your position:** Current weights
- **Altitude:** Loss value (how wrong the network is)
- **Goal:** Reach the valley (minimum error)
- **Strategy:** Feel the slope, step downhill
- **Step size:** Learning rate

This analogy threads through the entire lab and makes backpropagation intuitive.

---

## ⏱️ Detailed Timing Breakdown

### Part 1: Review and Motivation (20 minutes)

#### 1.1 Quick Review (10 minutes)

**What to cover:**

- Lab 01: Single neuron, forward pass, activation functions
- Lab 02: MLP architecture, forward propagation through layers

**Interactive questions:**

- "Who can explain forward propagation in their own words?"
- "What was frustrating about Lab 02?" (Answer: Manual weight setting!)

#### 1.2 The Learning Problem (10 minutes)

**What to cover:**

- In Lab 02, we manually chose weights for XOR
- Real problems: thousands/millions of weights!
- We need automatic weight optimization

**Demo:**
Show a network with random weights failing at XOR:

```python
# Random weights - bad performance
mlp = MLP([2, 2, 1])
# Test it - should fail miserably
```

**Key message:** "We need the network to **learn** from examples!"

---

### Part 2: Loss Functions (30 minutes)

#### 2.1 What is Loss? (10 minutes)

**Analogy:** Loss is like a report card - measures how wrong your network is.

**Visual:** Draw on board:

```
Prediction: 0.8
Actual: 1.0
Error: 0.2 (off by 0.2)

Loss quantifies this error!
```

**Key point:** Loss is a single number summarizing all errors.

#### 2.2 Mean Squared Error (15 minutes)

**Formula on board:**

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

**Why squared?**

1. Makes all errors positive
2. Penalizes large errors more
3. Mathematically nice (smooth, differentiable)

**Calculate by hand example:**

```
Example 1: y=1, prediction=0.8
Error = 1 - 0.8 = 0.2
Squared error = 0.04

Example 2: y=0, prediction=0.3
Error = 0 - 0.3 = -0.3
Squared error = 0.09

Average = (0.04 + 0.09) / 2 = 0.065
```

**Common student question:** "Why not absolute value?"
**Answer:** Squared error has nice mathematical properties for derivatives.

#### 2.3 Cross-Entropy Loss (5 minutes - brief mention)

**When to use:** Binary classification problems

**Quick formula:**

$$
\text{BCE} = -\frac{1}{n} \sum [y \log(\hat{y}) + (1-y) \log(1-\hat{y})]
$$

**Note:** "We'll use MSE for simplicity in this lab. Cross-entropy is better for classification but mathematically more complex."

---

### Part 3: Gradient Descent Concept (45 minutes)

#### 3.1 The Mountain Analogy (15 minutes)

**Setup the scene:**
"Imagine you're on a mountain in thick fog. You can't see where the valley is, but you can feel the slope under your feet. How do you get down?"

**Interactive discussion:**

- Feel which direction is downhill → **gradient**
- Take a step in that direction → **update**
- Repeat until you reach the valley → **optimization**

**Draw on board:**

```
     /\
    /  \
   /    \
  /  You \
 /   are  \
/   here!  \

↓ Feel the slope (gradient)
↓ Step downhill
↓ Repeat
```

#### 3.2 Mathematical Intuition (15 minutes)

**Start simple - one weight:**

Draw loss vs weight graph on board:

```
Loss
 |     /\
 |    /  \
 |   /    \
 |  /      \___
 |_______________ Weight

Current weight is here ↑
Gradient is slope
Negative gradient points downhill
```

**Update rule:**

$$
w_{\text{new}} = w_{\text{old}} - \alpha \cdot \frac{\partial L}{\partial w}
$$

**Break it down:**

- $w_{\text{old}}$: Where you are
- $\frac{\partial L}{\partial w}$: Slope at your position
- $\alpha$: Step size (learning rate)
- $w_{\text{new}}$: Where you step to

#### 3.3 Learning Rate Discussion (15 minutes)

**Demo with code - show three cases:**

**Case 1: Learning rate too small (α = 0.001)**

```python
# Takes forever, barely moves
```

**Analogy:** "Baby steps down the mountain - you'll be there next week!"

**Case 2: Learning rate too large (α = 10)**

```python
# Jumps over the valley, unstable
```

**Analogy:** "Giant leaps - you jump over the valley and land on the other mountain!"

**Case 3: Learning rate just right (α = 0.1)**

```python
# Converges nicely
```

**Analogy:** "Comfortable hiking pace - you reach the valley efficiently."

**Key teaching moment:** Let students suggest learning rates and see what happens!

---

### Part 4: Backpropagation Algorithm (60 minutes)

**⚠️ CRITICAL:** This is the hardest part of the lab. Go slowly, use multiple explanations.

#### 4.1 The Problem (10 minutes)

**Question to class:** "Gradient descent needs gradients. But our network has many layers! How do we compute ∂Loss/∂weight for each weight?"

**Answer:** Backpropagation!

**Key insight:** "Information flows forward during prediction, gradients flow backward during learning."

#### 4.2 Chain Rule Review (15 minutes)

**Most students are rusty on calculus! Quick review:**

**Simple example:**

$$
y = (x + 3)^2
$$

**Question:** "What's dy/dx?"

**Chain rule:**

- Let $u = x + 3$
- Then $y = u^2$
- $\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = 2u \cdot 1 = 2(x+3)$

**Neural network parallel:**

```
Input → Hidden → Output → Loss

Loss depends on output
Output depends on hidden
Hidden depends on input

So: ∂Loss/∂input = ∂Loss/∂output × ∂output/∂hidden × ∂hidden/∂input
```

#### 4.3 Backpropagation Intuition (20 minutes)

**Explain backward pass step by step:**

**Step 1: Forward pass (regular prediction)**

```python
# Calculate activations layer by layer
z1 = W1 @ x + b1
a1 = sigmoid(z1)
z2 = W2 @ a1 + b2
a2 = sigmoid(z2)
loss = (y - a2)^2
```

**Step 2: Output layer gradient**

```python
# How does loss change if output changes?
dL_da2 = -2(y - a2)  # Derivative of squared error
da2_dz2 = sigmoid'(z2)  # Derivative of activation
dL_dz2 = dL_da2 * da2_dz2  # Chain rule!
```

**Step 3: Hidden layer gradient (this is where backprop shines)**

```python
# How does loss change if hidden layer changes?
dL_da1 = dL_dz2 @ W2.T  # Backpropagate through weights!
da1_dz1 = sigmoid'(z1)
dL_dz1 = dL_da1 * da1_dz1
```

**Step 4: Weight gradients**

```python
# Now we can update the weights!
dL_dW2 = dL_dz2 @ a1.T
dL_dW1 = dL_dz1 @ x.T
```

**Visual on board:**

```
Forward:  Input → H1 → H2 → Output → Loss
Backward: Input ← H1 ← H2 ← Output ← Loss
               ↓    ↓    ↓
            Update Update Update
            weights weights weights
```

#### 4.4 Why "Back"propagation? (5 minutes)

**Key insight:** "We start at the output (where we know the error) and work backwards, layer by layer, using the chain rule at each step."

**Analogy:** "Like tracing back through your work to find where you made mistakes."

#### 4.5 Live Coding (10 minutes)

**Code a simple backprop example together:**

- Single hidden layer
- One training example
- Step through each calculation
- Print intermediate values

**Teaching tip:** Have students predict values before running code.

---

### Part 5: Training Loop (40 minutes)

#### 5.1 The Complete Training Algorithm (15 minutes)

**Write on board:**

```
FOR each epoch:
    FOR each training example:
        1. Forward pass (predict)
        2. Calculate loss
        3. Backward pass (backpropagation)
        4. Update weights (gradient descent)

    Print loss to monitor progress
```

**Key terms:**

- **Epoch:** One complete pass through all training data
- **Iteration:** Processing one training example
- **Batch:** Group of examples (we'll use one-at-a-time for simplicity)

#### 5.2 Training XOR from Scratch (25 minutes)

**This is the payoff moment!**

**Setup:**

```python
# XOR training data
X = [[0,0], [0,1], [1,0], [1,1]]
y = [[0], [1], [1], [0]]

# Random initial weights
mlp = MLP([2, 2, 1])  # Random initialization
```

**Before training:**

```python
# Test - should be wrong
print("Before training:", mlp.predict([0,0]))  # Random output
```

**Train:**

```python
# Train for 1000 epochs
train(mlp, X, y, epochs=1000, learning_rate=0.5)
```

**After training:**

```python
# Test - should be correct!
print("After training:", mlp.predict([0,0]))  # ~0.0
print("After training:", mlp.predict([0,1]))  # ~1.0
```

**Celebration moment:** "The network learned XOR by itself! We didn't set the weights - it figured them out from examples!"

---

### Part 6: Practical Considerations (35 minutes)

#### 6.1 Monitoring Training (10 minutes)

**Plot loss over epochs:**

```python
# Loss should decrease over time
# If it doesn't, something's wrong!
```

**What to look for:**

- Decreasing loss → Learning! ✓
- Flat loss → Stuck or already optimal
- Increasing loss → Learning rate too high or bug!
- Jumpy loss → Normal for small datasets

#### 6.2 Hyperparameter Tuning (15 minutes)

**Experiment together:**

**Learning rate experiments:**

```python
# Too small: α = 0.01
train(mlp, X, y, epochs=1000, lr=0.01)  # Slow

# Just right: α = 0.5
train(mlp, X, y, epochs=1000, lr=0.5)   # Good

# Too large: α = 5.0
train(mlp, X, y, epochs=1000, lr=5.0)   # Unstable!
```

**Hidden layer size experiments:**

```python
# Too few neurons: [2, 1, 1]
# Just enough: [2, 2, 1]
# More than needed: [2, 10, 1]
```

**Key lesson:** "There's no formula for perfect hyperparameters - experimentation is part of ML!"

#### 6.3 Overfitting Preview (10 minutes)

**Brief mention (covered more in Lab 04):**

"If you train too long or have too complex a network, you might memorize the training data without learning the pattern."

**Simple demo:**

```python
# Train for way too long
train(mlp, X, y, epochs=100000)
# Works on training data but...
# (Would fail on different but similar data)
```

---

### Part 7: Implementation Session (40 minutes)

**File:** `training-implementation.py`

**Students follow along, typing code:**

#### 7.1 Activation Derivatives (10 minutes)

```python
def sigmoid_derivative(z):
    """Derivative of sigmoid: σ(z) * (1 - σ(z))"""
    s = sigmoid(z)
    return s * (1 - s)

def relu_derivative(z):
    """Derivative of ReLU: 1 if z > 0, else 0"""
    return 1 if z > 0 else 0
```

#### 7.2 Loss Functions (10 minutes)

```python
def mse_loss(y_true, y_pred):
    """Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    """Derivative of MSE"""
    return -2 * (y_true - y_pred) / len(y_true)
```

#### 7.3 Complete Training Function (20 minutes)

```python
def train(network, X, y, epochs, learning_rate):
    """Train neural network using backpropagation"""
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0

        for i in range(len(X)):
            # Forward pass
            output = network.forward(X[i])

            # Calculate loss
            loss = mse_loss(y[i], output)
            epoch_loss += loss

            # Backward pass
            network.backward(X[i], y[i], learning_rate)

        # Record average loss
        avg_loss = epoch_loss / len(X)
        loss_history.append(avg_loss)

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    return loss_history
```

---

### Part 8: Student Task Introduction (10 minutes)

**File:** `student-task.py`

**Three tasks:**

**Task 1: Implement Gradient Descent (Simple Function)**

- Minimize $f(x) = (x-3)^2$
- Practice gradient descent before neural networks
- Visualize the descent

**Task 2: Train XOR Network**

- Implement backpropagation for MLP
- Train to solve XOR
- Monitor and plot loss

**Task 3: Hyperparameter Experiments**

- Test different learning rates
- Try different network architectures
- Compare training speeds

**Instructions:**

- "Task 1 is about understanding gradient descent conceptually"
- "Task 2 is applying it to neural networks"
- "Task 3 is developing ML intuition through experimentation"

---

## 🎓 Assessment Checklist

### Before Starting:

- [ ] Students understand forward propagation (Lab 02)
- [ ] Students remember basic calculus (chain rule)
- [ ] Python environment ready with math library

### Part 1 (Review):

- [ ] Students recall forward pass
- [ ] Students identify the problem (manual weight setting)

### Part 2 (Loss):

- [ ] Students can calculate MSE by hand
- [ ] Students understand loss as "wrongness measure"

### Part 3 (Gradient Descent):

- [ ] Students grasp the mountain analogy
- [ ] Students understand update rule
- [ ] Students see effects of different learning rates

### Part 4 (Backpropagation):

- [ ] Students understand chain rule
- [ ] Students follow backward pass logic
- [ ] Students see why it's called "back"propagation

### Part 5 (Training):

- [ ] Students understand training loop structure
- [ ] Students successfully train XOR
- [ ] Students celebrate the "it learned!" moment

### Part 6 (Practical):

- [ ] Students monitor loss during training
- [ ] Students experiment with hyperparameters
- [ ] Students understand trial-and-error nature

### Part 7 (Implementation):

- [ ] Code runs without errors
- [ ] Students understand each function
- [ ] Students can explain backprop code

### Part 8 (Tasks):

- [ ] Instructions are clear
- [ ] Students know where to start
- [ ] Time expectations set

---

## 💡 Common Student Difficulties

### Difficulty 1: "Backpropagation is too complicated"

**Symptoms:** Glazed eyes, giving up, memorizing without understanding

**Solutions:**

1. **Slow down** - This is the hardest concept
2. **Use the mountain analogy repeatedly**
3. **Show forward and backward passes side-by-side visually**
4. **Start with a tiny network** (2→1→1)
5. **Calculate one example completely by hand on board**
6. **Emphasize:** "It's just chain rule applied multiple times"

**Reassurance:** "Backprop took the AI field years to figure out. It's okay if it takes you a few hours!"

### Difficulty 2: "Why is my loss increasing?"

**Common causes:**

1. Learning rate too high
2. Gradient calculation error
3. Wrong sign in update rule

**Debugging together:**

```python
# Check gradients are reasonable
print("Gradient magnitude:", np.abs(gradient))
# Should be small numbers, not huge or NaN

# Check loss at each step
print(f"Loss: {loss:.6f}")
# Should generally decrease
```

### Difficulty 3: "My network won't learn XOR"

**Common causes:**

1. Network too small (needs hidden layer with ≥2 neurons)
2. Learning rate too small
3. Not enough epochs
4. Activation function issue (using linear activation)

**Solution:** "Let's check your architecture and hyperparameters together"

### Difficulty 4: "What's the difference between gradient descent and backpropagation?"

**Clarification:**

- **Gradient descent:** The optimization algorithm (how to update weights)
- **Backpropagation:** The method to compute gradients (what to update)
- **Together:** Backprop calculates gradients, GD uses them to update

**Analogy:**

- Backprop: Figuring out which direction is downhill
- Gradient descent: Actually taking steps downhill

### Difficulty 5: "The math is overwhelming"

**Symptoms:** Trying to memorize formulas, not understanding flow

**Solutions:**

1. **Focus on intuition first, math second**
2. **Use dimensional analysis** - check matrix shapes
3. **Provide working code** - they can learn from it
4. **Emphasize:** "You don't need to derive backprop from scratch - understanding the idea is enough"

**Reassurance:** "Even researchers use libraries that implement this. Understanding the concept matters most."

---

## 🎯 Teaching Tips

### Tip 1: Make Training Visual

**Plot loss in real-time if possible:**

```python
import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.show()
```

Students love seeing the curve go down!

### Tip 2: Use Print Statements Liberally

**During backprop explanation:**

```python
print("Forward pass:")
print(f"  Input: {x}")
print(f"  Hidden: {h}")
print(f"  Output: {output}")
print(f"  Loss: {loss}")

print("\nBackward pass:")
print(f"  Output gradient: {dL_dout}")
print(f"  Hidden gradient: {dL_dhidden}")
print(f"  Weight gradient: {dL_dW}")
```

### Tip 3: Celebrate Small Wins

**When loss first drops:**
"Look! The loss went from 0.5 to 0.4! The network is learning!"

**When XOR works:**
"Your network just solved a problem that a single neuron can't solve! That's actual AI!"

### Tip 4: Connect to Real AI

**Examples:**

- "This is exactly how GPT was trained - same backpropagation!"
- "Your smartphone's facial recognition learned using gradient descent"
- "Self-driving cars optimize using these same principles"

### Tip 5: Hands-On Experimentation

**Encourage students to:**

- Change learning rates and observe
- Try different network sizes
- Break the code and fix it
- Add print statements

**Philosophy:** "The best way to understand training is to train many networks!"

---

## 📊 Grading Rubric

**Total: 100 points**

### Task 1: Gradient Descent on Simple Function (30 points)

- Correct derivative calculation (10 points)
- Proper update rule implementation (10 points)
- Reaches minimum successfully (5 points)
- Thoughtful answers to questions (5 points)

### Task 2: Train XOR Network (40 points)

- Correct backpropagation implementation (15 points)
- Proper training loop (10 points)
- Network successfully learns XOR (10 points)
- Loss monitoring and analysis (5 points)

### Task 3: Hyperparameter Experiments (20 points)

- Tested multiple learning rates (7 points)
- Tested different architectures (7 points)
- Thoughtful analysis of results (6 points)

### Reflection Questions (10 points)

- Completed all reflection questions (10 points)

### Bonus: Creative Exploration (+10 extra credit)

- Trained on additional problem (5 points)
- Implemented momentum or other optimization (5 points)

---

## 🔗 Connections to Other Labs

### From Lab 02:

- Uses MLP architecture
- Builds on forward propagation
- OOP makes backprop implementation cleaner

### To Lab 04 (Future):

- This lab: Basic gradient descent
- Lab 04: Advanced optimizers (Adam, RMSprop)
- This lab: Small toy problems
- Lab 04: Real datasets, deep networks

---

## 📚 Additional Resources for Instructors

### Recommended Videos:

- 3Blue1Brown: "What is backpropagation really doing?"
- Andrew Ng: Coursera ML course (Week 4)

### Visualization Tools:

- TensorFlow Playground (tensorflow.org/playground)
- Neural Network Playground by Daniel Smilkov

### Papers (Optional Deep Dive):

- Rumelhart et al. (1986): "Learning representations by back-propagating errors"

---

## 🎉 Success Indicators

**You know the lab went well when:**

✅ Students' networks successfully learn XOR  
✅ Students can explain gradient descent in their own words  
✅ Students debug their own learning rate issues  
✅ Students get excited when loss decreases  
✅ Students want to try training on other problems  
✅ Students understand the "learning" part of machine learning

**Quote from successful student:**
_"Wait, the network figured out the weights by itself? That's actually magic!"_

---

## 📝 Final Notes for Instructors

**Most Important Points:**

1. **Backpropagation is hard** - Spend adequate time, use multiple explanations
2. **Visual aids are crucial** - Draw on board constantly
3. **Celebrate the magic** - Training is when neural networks become "intelligent"
4. **Hands-on experimentation** - Let students break things and learn
5. **Patience pays off** - This lab has the steepest learning curve but biggest payoff

**Timing flexibility:**

- If running over time, shorten hyperparameter experiments
- Core concept (backpropagation) cannot be rushed
- Some students may need extra office hours

**Lab culture:**

- Encourage collaboration on concepts
- Individual implementation for assessment
- Create "aha!" moments through discovery

---

**Good luck! This lab transforms students from neural network users to neural network trainers! 🚀🧠**

---

**End of Instructor Guide**
