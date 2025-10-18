# Lab 01: Single Neuron - Instructor Guide

## Neural Networks Course - Computer Engineering

---

## 📋 Lab Overview

**Duration:** 3 hours  
**Difficulty:** Beginner  
**Prerequisites:** Basic programming knowledge  
**Learning Objectives:**

- Understand the biological analogy of neurons
- Learn the mathematical model of artificial neurons
- Implement a single neuron in Python
- Apply neuron to a classification problem

---

## 🎯 Teaching Strategy

This lab uses a **progressive teaching approach**:

1. **Analogy** - Connect to real-world experience (cold/hot water bottle)
2. **Mathematics** - Build theoretical foundation
3. **Python Basics** - Provide necessary programming tools
4. **Implementation** - Apply knowledge practically
5. **Task** - Reinforce learning through practice

---

## 🧠 Part 1: The Biological Analogy (30 minutes)

### Teaching Points

#### 1.1 Real-World Scenario

**Setup:** Hold a water bottle and ask students: "How does your brain determine if this bottle is cold or hot?"

**Discussion Points:**

- Our senses provide **inputs** (touch, sight)
- Each input has different **importance/weight**
- Brain processes these inputs and makes a **decision** (classification)

**[NEED IMAGE: Human neuron diagram showing dendrites, cell body, axon, and synapses]**

#### 1.2 Human Neuron Structure

Explain the components:

- **Dendrites:** Receive signals from other neurons (inputs)
- **Cell Body (Soma):** Processes the signals (weighted sum)
- **Axon:** Transmits the output signal
- **Synapses:** Connections with varying strengths (weights)

#### 1.3 Water Bottle Classification Example

**Scenario:** Determining if a water bottle is COLD or HOT

**Inputs:**

1. **Temperature sensation from touch** (x₁)
   - Range: -10 (very cold) to +10 (very hot)
2. **Visual cues** (x₂)

   - Condensation on bottle: suggests cold
   - Steam: suggests hot
   - Range: -5 (cold indicators) to +5 (hot indicators)

3. **Context** (x₃)
   - Room temperature
   - Time since last use
   - Range: -3 to +3

**Weights (Importance):**

- w₁ = 0.7 (Touch is most reliable - 70% importance)
- w₂ = 0.2 (Visual cues - 20% importance)
- w₃ = 0.1 (Context - 10% importance)

**Decision Process:**

```
If (weighted_sum > threshold) → HOT
Else → COLD
```

**[NEED IMAGE: Diagram showing water bottle → inputs (hand, eye) → neuron → output (cold/hot)]**

---

## 📐 Part 2: Mathematical Model (45 minutes)

### Teaching Points

#### 2.1 The Mathematical Neuron

**Components:**

1. **Inputs:** x₁, x₂, x₃, ..., xₙ
2. **Weights:** w₁, w₂, w₃, ..., wₙ
3. **Bias:** b (threshold adjustment)
4. **Activation Function:** f(z)
5. **Output:** y

#### 2.2 Step-by-Step Mathematics

**Step 1: Weighted Sum (Linear Combination)**

$$
z = \sum_{i=1}^{n} w_i \cdot x_i + b
$$

Expanded form:

$$
z = w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3 + ... + w_n \cdot x_n + b
$$

**Symbol Explanation:**

- **z:** Net input (aggregated signal)
- **xᵢ:** i-th input value
- **wᵢ:** i-th weight (importance of input i)
- **n:** Number of inputs
- **b:** Bias term (adjusts decision boundary)
- **Σ (Sigma):** Summation operator (sum all terms)

**Bias Explanation:**
The bias 'b' is like a **threshold adjuster**. It shifts the decision boundary without depending on any input.

- Positive bias: Makes neuron more likely to activate
- Negative bias: Makes neuron less likely to activate
- Zero bias: Decision depends purely on weighted inputs

**Example Calculation:**

```
Given:
x₁ = 8 (touch: quite hot)
x₂ = 3 (visual: some steam)
x₃ = 1 (context: warm room)
w₁ = 0.7, w₂ = 0.2, w₃ = 0.1
b = -2 (conservative threshold)

z = (0.7 × 8) + (0.2 × 3) + (0.1 × 1) + (-2)
z = 5.6 + 0.6 + 0.1 - 2
z = 4.3
```

**Step 2: Activation Function**

The activation function **f(z)** determines the final output based on the net input z.

$$
y = f(z)
$$

Where **y** is the output of the neuron.

#### 2.3 Activation Functions

**Why do we need activation functions?**

- Introduce **non-linearity** (real-world patterns are rarely linear)
- Control the **output range**
- Enable learning of **complex patterns**

**Common Activation Functions:**

##### 2.3.1 Step Function (Threshold Function)

```
f(z) = {1  if z ≥ 0
       {0  if z < 0
```

**Characteristics:**

- **Binary output:** 0 or 1 (Cold or Hot)
- **Simple:** Easy to understand
- **Limitation:** Not differentiable (problematic for learning)

**Use Case:** Simple binary classification (our water bottle example)

**[NEED IMAGE: Step function graph showing sharp transition at z=0]**

##### 2.3.2 Sigmoid Function (Logistic)

```
f(z) = 1 / (1 + e^(-z))
```

**Characteristics:**

- **Output range:** (0, 1)
- **Smooth curve:** Differentiable everywhere
- **Interpretation:** Probability-like output
- **Formula components:**
  - e ≈ 2.71828 (Euler's number)
  - e^(-z): Exponential decay

**Example Values:**

```
z = -5  →  f(z) ≈ 0.007 (almost 0)
z = 0   →  f(z) = 0.5   (neutral)
z = 5   →  f(z) ≈ 0.993 (almost 1)
```

**Use Case:** Binary classification with probability output, hidden layers in neural networks

**[NEED IMAGE: Sigmoid function graph showing S-curve from 0 to 1]**

##### 2.3.3 Tanh Function (Hyperbolic Tangent)

```
f(z) = (e^z - e^(-z)) / (e^z + e^(-z))
```

**Characteristics:**

- **Output range:** (-1, 1)
- **Zero-centered:** Better for optimization
- **Stronger gradients:** Compared to sigmoid

**Example Values:**

```
z = -2  →  f(z) ≈ -0.96
z = 0   →  f(z) = 0
z = 2   →  f(z) ≈ 0.96
```

**Use Case:** Hidden layers, when zero-centered output is beneficial

**[NEED IMAGE: Tanh function graph showing S-curve from -1 to 1]**

##### 2.3.4 ReLU (Rectified Linear Unit)

```
f(z) = max(0, z) = {z  if z ≥ 0
                   {0  if z < 0
```

**Characteristics:**

- **Simple:** Easy to compute
- **Efficient:** Faster training
- **Sparse activation:** Many neurons output 0
- **Problem:** "Dying ReLU" (neurons stuck at 0)

**Example Values:**

```
z = -5  →  f(z) = 0
z = 0   →  f(z) = 0
z = 5   →  f(z) = 5
```

**Use Case:** Modern deep learning, hidden layers (most popular currently)

**[NEED IMAGE: ReLU function graph showing flat line at 0 for negative, linear for positive]**

##### 2.3.5 Leaky ReLU

```
f(z) = {z      if z ≥ 0
       {α·z    if z < 0     (typically α = 0.01)
```

**Characteristics:**

- **Fixes dying ReLU:** Small slope for negative values
- **α (alpha):** Small constant (e.g., 0.01)

**Use Case:** Alternative to ReLU when dying neurons are a problem

---

#### 2.4 Complete Neuron Formula

Combining everything:

$$
y = f\left(\sum_{i=1}^{n} w_i \cdot x_i + b\right)
$$

Or expanded:

$$
y = f(w_1 \cdot x_1 + w_2 \cdot x_2 + ... + w_n \cdot x_n + b)
$$

**Water Bottle Example with Step Function:**

```
z = 4.3 (from previous calculation)
f(z) = step(4.3) = 1  (since 4.3 ≥ 0)
Output: 1 → HOT
```

---

## 🐍 Part 3: Python Basics (45 minutes)

**Note to Instructor:**
Refer students to `python-basics.py` file. Go through each concept with live coding examples. Encourage students to experiment in their own Python environment.

### Topics Covered:

1. Variables and data types
2. Lists and indexing
3. Conditional statements (if/else)
4. Loops (for loop)
5. Functions
6. Mathematical operations
7. Import statements

**Teaching Tip:** Use the water bottle example throughout the Python basics to maintain context.

---

## 💻 Part 4: Implementation (45 minutes)

**Note to Instructor:**
Use the `neuron-implementation.py` file. Build the code incrementally:

### Implementation Steps:

#### 4.1 Activation Functions (15 minutes)

- Start with simple step function
- Explain if/else logic
- Add sigmoid, tanh, relu implementations
- Explain mathematical functions in Python (math.exp, math.tanh, max)

#### 4.2 Neuron Calculation Functions (15 minutes)

- Implement `calculate_weighted_sum()` function
- Implement `apply_activation()` function
- Build complete `neuron_predict()` function
- Show how functions compose together (functional programming)

**Important:** We use **functions** (not classes) in Lab 01 to keep things simple.
Students will learn Object-Oriented Programming (OOP) in Lab 02.

#### 4.3 Water Bottle Classifier (15 minutes)

- Create practical example using functions
- Define weights, bias, activation as variables
- Call `neuron_predict()` with different inputs
- Test with different inputs
- Discuss results and interpretation
- Show how changing weights affects output

**Teaching Tips:**

- **Live code** with students following along
- **Intentionally make mistakes** to debug together
- **Ask predictive questions:** "What output do you expect?"
- **Vary parameters** to show sensitivity

---

## 📝 Part 5: Student Task (15 minutes)

**Note to Instructor:**
Assign the tasks from `student-task.py`. This should be completed during lab time with your assistance.

### Task Overview:

1. **Task 1:** Implement fruit ripeness classifier
   - Inputs: color, smell, firmness
   - Output: ripe or unripe
2. **Task 2:** Experiment with different activation functions

   - Compare outputs
   - Understand function characteristics

3. **Task 3:** Light switch controller
   - Multiple environmental factors
   - Binary decision (on/off)

**Grading Rubric:**

- Correct implementation: 60%
- Code quality and comments: 20%
- Testing with multiple inputs: 10%
- Written explanation: 10%

---

## 🎓 Assessment Checklist

By the end of this lab, students should be able to:

- [ ] Explain biological neuron analogy
- [ ] Describe mathematical neuron model
- [ ] Calculate weighted sum manually
- [ ] Explain role of activation functions
- [ ] Implement a neuron in Python
- [ ] Apply neuron to classification problems
- [ ] Interpret neuron outputs

---

## 🔍 Common Student Difficulties

### Issue 1: Understanding Weights

**Problem:** Students confuse weights with inputs  
**Solution:** Use physical analogy - weights are like "volume knobs" for each input

### Issue 2: Bias Confusion

**Problem:** Not understanding why we need bias  
**Solution:** Show examples where same inputs give different outputs with different biases

### Issue 3: Activation Function Purpose

**Problem:** "Why not just use weighted sum?"  
**Solution:** Demonstrate limitation of linear functions; show how non-linearity enables complex patterns

### Issue 4: Python List Indexing

**Problem:** Off-by-one errors  
**Solution:** Emphasize Python uses 0-based indexing; practice with visual examples

### Issue 5: Mathematical Notation

**Problem:** Intimidated by Σ notation  
**Solution:** Always show both compact and expanded forms; relate to for loops

---

## 📚 Additional Resources

### For Students:

1. 3Blue1Brown - "But what is a Neural Network?" (YouTube)
2. Khan Academy - Neuron structure
3. Python documentation - Math module

### For Instructor:

1. Nielsen, Michael - "Neural Networks and Deep Learning" (free online book)
2. Goodfellow et al. - "Deep Learning" (Chapter 6)

---

## 🕐 Time Management

| Activity             | Time        | Notes                          |
| -------------------- | ----------- | ------------------------------ |
| Analogy Introduction | 30 min      | Interactive discussion         |
| Mathematics          | 45 min      | Work through examples on board |
| Python Basics        | 45 min      | Live coding session            |
| Implementation       | 45 min      | Code along with students       |
| Student Task         | 15 min      | Independent work with support  |
| **Total**            | **3 hours** |                                |

---

## 🎯 Next Lab Preview

In **Lab 02 - Multi-Layer Perceptron**, students will:

- Connect multiple neurons in layers
- Learn Object-Oriented Programming (OOP)
- Build a complete neural network
- Solve real AI problems

**Preparation:** Ensure students complete this lab's tasks and understand single neuron thoroughly.

---

**Version:** 1.0  
**Last Updated:** October 2025  
**Course:** Neural Networks - Computer Engineering  
**Institution:** [Your University Name]
