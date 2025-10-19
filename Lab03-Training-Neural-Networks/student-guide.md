# Lab 03: Training Neural Networks

## Neural Networks Course - Computer Engineering

---

## 📋 Lab Information

**Difficulty:** Advanced  
**Prerequisites:** Lab 01 (Single Neuron), Lab 02 (Multi-Layer Perceptron)

**What You'll Learn:**

- How neural networks learn from data automatically
- The backpropagation algorithm
- Gradient descent optimization
- Loss functions and how to minimize them
- How to train networks to solve real problems

**Files You'll Need:**

- `student-guide.md` (this file)
- `gradient-descent-tutorial.py` (optimization basics)
- `training-implementation.py` (complete training code)
- `student-task.py` (your assignment)

---

## 🎯 Introduction

Welcome to the most exciting lab yet! Until now, you've been **manually setting weights** for neural networks. Today, you'll learn how networks **learn optimal weights automatically** from data.

This is where neural networks become truly "intelligent" - they figure out the solution by learning from examples!

**The Big Question:** How does a network with thousands or millions of weights learn the right values?

**The Answer:** **Backpropagation** + **Gradient Descent**

---

## 🏔️ Part 1: The Mountain Climbing Analogy

### The Problem: Finding the Valley

Imagine you're standing on a mountain in thick fog. You can't see anything beyond a few meters, but you need to reach the valley below (the lowest point). How do you get there?

**[NEED IMAGE: Person on foggy mountain with valley below]**

### Your Strategy:

1. **Feel the ground** - Which direction slopes downward?
2. **Take a step** - Move in the downhill direction
3. **Repeat** - Keep feeling and stepping until you're at the bottom

This is exactly how neural networks learn!

### The Neural Network Parallel:

| Mountain Hiking           | Neural Network Training |
| ------------------------- | ----------------------- |
| Your position on mountain | Current weights         |
| Altitude (height)         | Loss (error)            |
| Valley (lowest point)     | Optimal weights         |
| Feeling the slope         | Computing gradients     |
| Taking a step downhill    | Updating weights        |
| Step size                 | Learning rate           |
| Reaching the valley       | Training complete!      |

**Key Insight:** You don't need to see the whole mountain - you just need to know which direction is downhill at your current position!

---

## 📊 Part 2: Loss Functions - Measuring "Wrongness"

### What is Loss?

**Loss** (also called **cost** or **error**) is a single number that measures how wrong your network's predictions are.

**Think of it like a test score:**

- Low loss = Good predictions (high grade!)
- High loss = Bad predictions (need to study more!)

### Why Do We Need Loss?

In Lab 02, you visually checked if XOR worked. But for real problems with thousands of examples, you need an automatic way to measure performance.

Loss gives you a single number to optimize: **Make this number as small as possible!**

### Mean Squared Error (MSE)

The most common loss function for regression:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

**Breaking it down:**

- $y_i$ = actual (correct) value
- $\hat{y}_i$ = predicted value
- $(y_i - \hat{y}_i)$ = error for one example
- $(y_i - \hat{y}_i)^2$ = squared error (always positive)
- Sum and average over all $n$ examples

### Example Calculation:

**Dataset:**

```
Example 1: Actual = 1.0, Predicted = 0.8
Example 2: Actual = 0.0, Predicted = 0.2
Example 3: Actual = 1.0, Predicted = 0.9
```

**Calculate MSE:**

```
Example 1 error: (1.0 - 0.8)² = 0.04
Example 2 error: (0.0 - 0.2)² = 0.04
Example 3 error: (1.0 - 0.9)² = 0.01

MSE = (0.04 + 0.04 + 0.01) / 3 = 0.03
```

**Interpretation:** Average squared error is 0.03 - pretty good!

### Why Square the Error?

1. **Makes all errors positive** - Otherwise positive and negative errors cancel out
2. **Penalizes large errors more** - Being off by 0.5 is worse than being off by 0.1 twice
3. **Mathematically nice** - Has smooth derivatives (important for calculus!)

### Self-Check Question ✓

**Q:** Calculate MSE for these predictions:

- Actual: [0, 1, 1, 0]
- Predicted: [0.1, 0.9, 0.8, 0.2]

<details>
<summary>Answer (click to reveal)</summary>

```
Error 1: (0 - 0.1)² = 0.01
Error 2: (1 - 0.9)² = 0.01
Error 3: (1 - 0.8)² = 0.04
Error 4: (0 - 0.2)² = 0.04

MSE = (0.01 + 0.01 + 0.04 + 0.04) / 4 = 0.025
```

</details>

---

## ⛰️ Part 3: Gradient Descent - Climbing Down the Mountain

### The Optimization Problem

**Goal:** Find weights that minimize loss.

**Challenge:** We don't know what the loss "landscape" looks like! We can't just look at all possible weights and pick the best ones.

**Solution:** Gradient descent - feel your way down!

### The Update Rule

At each step, update your weights like this:

$$
w_{\text{new}} = w_{\text{old}} - \alpha \cdot \frac{\partial L}{\partial w}
$$

**What this means:**

- $w_{\text{old}}$: Your current position (current weights)
- $\frac{\partial L}{\partial w}$: The slope (gradient) - which direction is downhill
- $\alpha$: Learning rate - how big of a step to take
- $w_{\text{new}}$: Your new position (updated weights)

### Understanding the Gradient

**The gradient $\frac{\partial L}{\partial w}$ tells you:**

- **Direction:** Positive gradient = loss increases if you increase weight
- **Magnitude:** Large gradient = steep slope, small gradient = gentle slope

**Why the negative sign?**
We subtract the gradient because:

- Positive gradient → decrease weight (move downhill)
- Negative gradient → increase weight (move downhill)

### Visual Example:

```
Loss
 |       ^
 |      /│\
 |     / │ \
 |    /  │  \
 |   /   │   \___
 |__|_______│______ Weight
     │     │
  Negative Positive
  gradient gradient
     ←     →
```

- **Left side:** Gradient is negative → increase weight (move right)
- **Right side:** Gradient is positive → decrease weight (move left)
- **Bottom:** Gradient is zero → we're at the minimum! ✓

### The Learning Rate (α)

The learning rate controls your step size. It's like choosing how fast to walk down the mountain.

**[NEED IMAGE: Three scenarios - tiny steps, normal steps, giant leaps]**

#### Learning Rate Too Small (α = 0.001):

- **Pro:** Stable, safe
- **Con:** Takes forever! 🐌
- **Analogy:** Crawling down the mountain - you'll get there eventually...

#### Learning Rate Just Right (α = 0.1):

- **Pro:** Efficient, converges quickly
- **Con:** Requires some tuning
- **Analogy:** Normal hiking pace - arrives in reasonable time ✓

#### Learning Rate Too Large (α = 10):

- **Pro:** Fast initial progress
- **Con:** Jumps over the valley, unstable! 🎢
- **Analogy:** Giant leaps - you jump over the valley and land on the other mountain!

### Practical Guidelines:

**Typical learning rates:** 0.001 to 1.0

**Signs your learning rate is wrong:**

- Loss increases → Too high
- Loss barely changes after many epochs → Too low
- Loss jumps around wildly → Too high
- Loss decreases smoothly → Just right! ✓

### Self-Check Question ✓

**Q:** You're at weight w = 2.0, the gradient is +0.5, and learning rate is 0.1. What's the new weight?

<details>
<summary>Answer (click to reveal)</summary>

```
w_new = w_old - α × gradient
w_new = 2.0 - 0.1 × 0.5
w_new = 2.0 - 0.05
w_new = 1.95
```

The weight decreased because the gradient was positive (moving left toward lower loss).

</details>

---

## 🔙 Part 4: Backpropagation - The "Magic" Algorithm

### The Core Problem

Gradient descent needs gradients: $\frac{\partial L}{\partial w}$ for every weight.

**Challenge:** Neural networks have many layers! How do we compute the gradient for weights in early layers?

**Answer:** **Backpropagation** - a clever application of calculus's chain rule.

### The Name "Backpropagation"

Information flows in two directions:

**Forward Pass (Prediction):**

```
Input → Layer 1 → Layer 2 → Output → Loss
```

**Backward Pass (Learning):**

```
Input ← Layer 1 ← Layer 2 ← Output ← Loss
```

We start at the output (where we know the error) and **propagate backwards** through the network, calculating gradients for each layer.

### Chain Rule Review

Remember from calculus: If $y = f(g(x))$, then:

$$
\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}
$$

**Example:**

- $y = (x + 3)^2$
- Let $u = x + 3$, then $y = u^2$
- $\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = 2u \cdot 1 = 2(x+3)$

### Neural Network Chain Rule

In a neural network:

```
Input (x) → Hidden (h) → Output (y) → Loss (L)
```

To find how loss changes with input:

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial h} \cdot \frac{\partial h}{\partial x}
$$

**Key insight:** We compute gradients layer by layer, starting from the output!

### Backpropagation Step by Step

**Network structure:**

```
Input → Hidden Layer → Output Layer → Loss
  x   →      h       →      y       →  L
```

**Step 1: Forward Pass (Make predictions)**

```python
z1 = W1 @ x + b1        # Hidden layer input
h = sigmoid(z1)          # Hidden layer activation
z2 = W2 @ h + b2        # Output layer input
y = sigmoid(z2)          # Output (prediction)
L = (target - y)²        # Loss
```

**Step 2: Output Layer Gradient**

```python
# How does loss change if output changes?
dL_dy = -2 * (target - y)     # Derivative of squared error

# How does output change if z2 changes?
dy_dz2 = sigmoid'(z2)         # Derivative of sigmoid

# Chain rule!
dL_dz2 = dL_dy * dy_dz2
```

**Step 3: Hidden Layer Gradient (The "Back" Part)**

```python
# How does loss change if hidden activation changes?
dL_dh = dL_dz2 @ W2.T         # Backpropagate through weights!

# How does hidden activation change if z1 changes?
dh_dz1 = sigmoid'(z1)

# Chain rule again!
dL_dz1 = dL_dh * dh_dz1
```

**Step 4: Weight Gradients**

```python
# Now we can compute gradients for the weights!
dL_dW2 = dL_dz2 @ h.T         # Output layer weights
dL_dW1 = dL_dz1 @ x.T         # Hidden layer weights
```

**Step 5: Update Weights**

```python
W2 = W2 - learning_rate * dL_dW2
W1 = W1 - learning_rate * dL_dW1
```

### Why This Works

**The brilliant insight:**

- We know the error at the output (actual - predicted)
- We can work backwards, using the chain rule at each layer
- Each layer's gradient depends on the next layer's gradient
- This lets us calculate gradients for ALL weights efficiently!

**Without backpropagation:** We'd have to test each weight individually (millions of forward passes!)

**With backpropagation:** One forward pass + one backward pass gives us all gradients!

### Visual Flow:

```
Forward:  x → [W1,b1] → h → [W2,b2] → y → L

Backward: x ← [↓  ↓ ] ← h ← [↓  ↓ ] ← y ← L
              ∂L/∂W1        ∂L/∂W2
              ∂L/∂b1        ∂L/∂b2
```

### Self-Check Question ✓

**Q:** In which direction does gradient information flow during backpropagation?

<details>
<summary>Answer (click to reveal)</summary>

**Backwards!** From output layer to input layer.

That's why it's called "back"propagation - the gradients propagate in the opposite direction of the forward pass.

</details>

---

## 🔄 Part 5: The Training Loop

### Complete Training Algorithm

Putting it all together:

```
FOR each epoch (1 to N):
    FOR each training example (x, y):

        1. FORWARD PASS
           - Feed input through network
           - Get prediction

        2. COMPUTE LOSS
           - Compare prediction to actual value
           - Calculate loss (MSE)

        3. BACKWARD PASS (Backpropagation)
           - Compute gradients for all weights
           - Work backwards from output to input

        4. UPDATE WEIGHTS (Gradient Descent)
           - w = w - α × gradient
           - For all weights in all layers

    Print/plot loss to monitor progress
```

### Key Terms:

**Epoch:** One complete pass through ALL training examples

- Example: If you have 100 training examples, 1 epoch = 100 iterations

**Iteration:** Processing one training example (or batch)

**Convergence:** When loss stops decreasing significantly

- The network has learned as much as it can!

### Training XOR: The Payoff!

Remember from Lab 02 that XOR can't be solved by a single neuron? Now we'll **train** a multi-layer network to learn it!

**XOR Truth Table:**

```
Input 1 | Input 2 | Output
--------|---------|-------
   0    |    0    |   0
   0    |    1    |   1
   1    |    0    |   1
   1    |    1    |   0
```

**Training data:**

```python
X = [[0,0], [0,1], [1,0], [1,1]]
y = [[0],   [1],   [1],   [0]]
```

**Before training:**

```python
# Random weights - predictions are wrong
network.predict([0,0]) → 0.73  (should be 0)
network.predict([0,1]) → 0.42  (should be 1)
network.predict([1,0]) → 0.68  (should be 1)
network.predict([1,1]) → 0.55  (should be 0)
```

**After training (1000 epochs, α=0.5):**

```python
# Learned weights - predictions are correct!
network.predict([0,0]) → 0.02  (should be 0) ✓
network.predict([0,1]) → 0.97  (should be 1) ✓
network.predict([1,0]) → 0.98  (should be 1) ✓
network.predict([1,1]) → 0.03  (should be 0) ✓
```

**This is the magic moment:** The network figured out the right weights automatically just from examples! 🎉

---

## 📈 Part 6: Monitoring Training

### Loss Over Time

The most important metric during training is **loss vs epoch**.

**What you want to see:**

```
Loss
 |
4|  ●
 |
3|   ●
 |     ●
2|       ●●
 |         ●●●
1|            ●●●●●●●
 |_____________________ Epoch
   0  100  200  300
```

Loss decreases steadily → Network is learning! ✓

**Warning signs:**

**Flat Loss:**

```
Loss
 |
2|  ●●●●●●●●●●●●●●●
 |_____________________ Epoch
```

- Network isn't learning
- Possible causes: Learning rate too small, network too simple, already at optimum

**Increasing Loss:**

```
Loss
 |
 |                ●●●
 |           ●●●
 |      ●●●
2|  ●●●
 |_____________________ Epoch
```

- Network is getting worse!
- Cause: Learning rate too high

**Jumpy Loss:**

```
Loss
 |   ●
 |  ● ● ●  ●
2| ●   ●  ●  ●
 |  ●    ●
 |_____________________ Epoch
```

- Normal for small datasets
- Try reducing learning rate if extreme

### When to Stop Training?

**Option 1: Fixed number of epochs**

- Train for 1000 epochs, then stop
- Simple but might stop too early or too late

**Option 2: Convergence criteria**

- Stop when loss stops improving significantly
- Example: If loss change < 0.001 for 10 epochs, stop

**Option 3: Validation loss (covered in Lab 04)**

- Monitor performance on separate validation data
- Stop when validation loss increases (overfitting)

---

## 🎛️ Part 7: Hyperparameters

**Hyperparameters** are settings you choose before training (not learned from data):

### 1. Learning Rate (α)

**Most important hyperparameter!**

**Typical values:** 0.001, 0.01, 0.1, 0.5, 1.0

**How to choose:**

- Start with α = 0.1
- If loss increases → reduce (try 0.01)
- If loss barely changes → increase (try 0.5)
- Use trial and error!

### 2. Number of Epochs

**How many times to go through training data**

**Typical values:** 100 to 10,000+

**How to choose:**

- Train until loss stops decreasing
- More epochs isn't always better (can lead to overfitting)

### 3. Network Architecture

**How many layers? How many neurons per layer?**

**For XOR:**

- Minimum: [2, 2, 1] - 2 inputs, 2 hidden neurons, 1 output
- Works well: [2, 3, 1], [2, 4, 1], [2, 5, 1]
- Overkill: [2, 100, 50, 1] - will work but unnecessarily complex

**General rules:**

- Start simple, add complexity if needed
- More neurons = more capacity but slower training
- More layers = can learn more complex patterns

### 4. Activation Functions

**For hidden layers:** ReLU or sigmoid
**For output layer:**

- Sigmoid for binary classification (0 to 1)
- Linear for regression (any value)

### The Art of Hyperparameter Tuning

**There's no formula for "perfect" hyperparameters!**

Machine learning involves experimentation:

1. Start with reasonable defaults
2. Train and observe
3. Adjust based on results
4. Repeat until satisfied

**This is normal!** Even experts use trial and error.

---

## ⚠️ Part 8: Common Issues and Solutions

### Issue 1: "My loss is not decreasing"

**Possible causes:**

1. **Learning rate too small** → Try larger (0.1 to 1.0)
2. **Network too simple** → Add more hidden neurons
3. **Wrong activation function** → Try sigmoid or ReLU
4. **Bug in code** → Check backprop implementation

**Debugging:**

```python
# Check if gradients are being computed
print("Gradient:", gradient)
# Should be non-zero numbers

# Check if weights are updating
print("Weights before:", W1)
# train one step
print("Weights after:", W1)
# Should be different!
```

### Issue 2: "My loss is increasing"

**Cause:** Learning rate is too high!

**Solution:**

```python
# Reduce learning rate by 10x
learning_rate = 0.01  # was 0.1
```

### Issue 3: "Loss decreases then increases"

**Cause:** Might be overfitting (memorizing training data)

**Solution:**

- Stop training earlier
- Use simpler network
- Add regularization (Lab 04 topic)

### Issue 4: "Training is very slow"

**Possible causes:**

1. Learning rate too small
2. Network too large
3. Too many epochs
4. Inefficient code

**Solutions:**

- Increase learning rate
- Reduce network size
- Check for unnecessary loops in code

---

## 🎓 Key Takeaways

### 1. How Networks Learn

**Before:** We manually set weights (tedious, only works for tiny problems)

**Now:** Networks learn weights automatically from data! (scales to millions of parameters)

### 2. Loss Functions

- **Loss measures how wrong predictions are**
- **MSE = Mean Squared Error** - most common for regression
- **Goal: Minimize loss**

### 3. Gradient Descent

- **Optimization algorithm** - finds weights that minimize loss
- **Update rule:** $w = w - \alpha \cdot \frac{\partial L}{\partial w}$
- **Key parameter:** Learning rate (α) - controls step size

### 4. Backpropagation

- **Efficient way to compute gradients** for all weights
- **Uses chain rule** - works backward from output to input
- **One forward + one backward pass** gives all gradients
- **This is what makes deep learning practical!**

### 5. Training Loop

```
Repeat:
  Forward pass → Compute loss → Backward pass → Update weights
Until loss stops decreasing
```

### 6. Hyperparameters

- **Learning rate** - most important, requires tuning
- **Epochs** - how long to train
- **Architecture** - layers and neurons
- **Experimentation is normal!**

---

## 🚀 What's Next?

### In Your Student Task:

You'll implement gradient descent on a simple function, then apply it to train a neural network to solve XOR!

### In Lab 04 (Future):

- **Advanced optimizers:** Adam, RMSprop, momentum
- **Regularization:** Preventing overfitting
- **Real datasets:** Handwritten digits, images
- **Deep learning frameworks:** PyTorch, TensorFlow

---

## 💡 Tips for Success

### Understand Conceptually First

Don't memorize formulas - understand the ideas:

- Gradient descent = walking downhill
- Backpropagation = chain rule applied repeatedly
- Training = adjusting weights to minimize loss

### Implement Step by Step

1. Get forward pass working
2. Add loss calculation
3. Implement backpropagation carefully
4. Add weight updates
5. Test on simple examples

### Use Print Statements

```python
print(f"Loss: {loss:.4f}")
print(f"Gradient: {gradient}")
print(f"Weights: {weights}")
```

Helps you see what's happening!

### Experiment!

Try different:

- Learning rates
- Network architectures
- Number of epochs

**Learning through experimentation is how ML works!**

### Don't Get Discouraged

Backpropagation is the hardest concept in this course. If it doesn't click immediately:

- Review the mountain analogy
- Draw the forward and backward passes
- Work through a tiny example by hand
- Ask for help!

**Even researchers had to learn this step by step!**

---

## ✅ Self-Assessment

Before moving to the student task, make sure you can:

- [ ] Explain what loss functions measure
- [ ] Calculate MSE by hand for simple examples
- [ ] Describe gradient descent using the mountain analogy
- [ ] Explain why we use the negative gradient
- [ ] Understand what learning rate controls
- [ ] Explain the purpose of backpropagation
- [ ] Describe the forward pass vs backward pass
- [ ] List the steps in the training loop
- [ ] Identify common training issues

**If you're unsure about any of these, review that section or ask your instructor!**

---

## 📚 Additional Resources

### Videos:

- **3Blue1Brown:** "What is backpropagation really doing?"
- **StatQuest:** "Gradient Descent, Step-by-Step"

### Interactive:

- **TensorFlow Playground:** tensorflow.org/playground
  - Visual neural network training in browser
  - Experiment with parameters interactively

### Reading:

- Course textbook Chapter on Training
- Online tutorials on backpropagation

---

## 🎉 Final Thoughts

**Training is where neural networks become truly intelligent!**

You're no longer just implementing networks - you're teaching them to learn. This is the foundation of all modern AI:

- Image recognition
- Natural language processing
- Self-driving cars
- Game-playing AI

All use the same principles you're learning today: **backpropagation + gradient descent**!

**You're learning the algorithm that powers modern AI. How cool is that?! 🚀🧠**

---

**Now, let's train some networks! Head to the implementation files and student task. Good luck! 💪**

---

**End of Student Guide**
