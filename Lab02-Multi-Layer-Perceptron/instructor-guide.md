# Lab 02: Multi-Layer Perceptron - Instructor Guide

## Neural Networks Course - Computer Engineering

---

## 📋 Lab Overview

**Duration:** 4 hours  
**Difficulty:** Intermediate  
**Prerequisites:** Lab 01 - Single Neuron  
**Learning Objectives:**

- Understand limitations of single neurons
- Learn multi-layer neural network architecture
- Master Object-Oriented Programming (OOP) in Python
- Implement a complete Multi-Layer Perceptron (MLP)
- Apply MLP to real AI problems

---

## 🎯 Teaching Strategy

This lab uses a **building-up approach**:

1. **Review & Limitations** - Why single neurons aren't enough
2. **OOP Introduction** - Using PUBG Mobile game analogy
3. **Architecture** - Understanding layers and connections
4. **Mathematics** - Forward propagation through layers
5. **Implementation** - Build MLP with and without OOP
6. **Application** - Solve real AI problem
7. **Task** - Practical assignment

---

## 🔄 Part 1: Review and Limitations (20 minutes)

### Teaching Points

#### 1.1 Single Neuron Recap

Quick review of Lab 01:

- Single neuron: weighted sum + activation
- Can classify simple patterns
- Works well for linearly separable problems

**[NEED IMAGE: Linear decision boundary showing two classes separated by a line]**

#### 1.2 The XOR Problem

**Draw on board:**

```
XOR Truth Table:
A | B | Output
0 | 0 |   0
0 | 1 |   1
1 | 0 |   1
1 | 1 |   0
```

**Plot on coordinate system:**

- Point (0,0) → class 0
- Point (0,1) → class 1
- Point (1,0) → class 1
- Point (1,1) → class 0

**Challenge students:** "Can you draw a single straight line that separates class 0 from class 1?"

**Answer:** No! This is NOT linearly separable.

**[NEED IMAGE: XOR problem visualization showing 4 points that cannot be separated by a single line]**

#### 1.3 Why We Need Multiple Layers

**Key Insight:**

- Single neuron = single decision boundary (line/plane)
- Multiple neurons = multiple decision boundaries
- Multiple layers = complex, curved decision boundaries

**Real-world examples of non-linear problems:**

1. Face recognition (complex patterns)
2. Speech recognition (temporal patterns)
3. Game playing (strategic decisions)
4. Medical diagnosis (multiple interacting factors)

**[NEED IMAGE: Progression showing how multiple neurons can create curved decision boundaries]**

---

## 🎮 Part 2: Object-Oriented Programming (60 minutes)

### Teaching Points

**Note to Instructor:** This is crucial! Many students struggle with OOP. Use the PUBG Mobile analogy extensively as students are familiar with the game.

#### 2.1 The Problem Without OOP

**Scenario:** Managing game characters in PUBG Mobile

**Ask students:** "How would you store information about different players in PUBG?"

**Without OOP approach (messy):**

```python
# Player 1
player1_name = "ProGamer"
player1_health = 100
player1_armor = 75
player1_position_x = 150
player1_position_y = 200
player1_weapon = "M416"

# Player 2
player2_name = "SnipeMaster"
player2_health = 80
player2_armor = 50
player2_position_x = 300
player2_position_y = 450
player2_weapon = "AWM"

# And 98 more players...
```

**Problems:**

- Repetitive code
- Hard to manage 100 players
- Difficult to add new features
- No organization

**[NEED IMAGE: Cluttered code screenshot showing many variables]**

#### 2.2 Introducing Classes and Objects

**Class = Blueprint/Template**
Think of it like a character creation screen in PUBG:

- Defines what properties a player has (health, armor, position)
- Defines what actions a player can do (shoot, move, heal)

**Object = Specific Instance**
Each player in the game is an object created from the Player class.

**Real-world analogy:**

- **Class = Car blueprint** (design specifications)
- **Object = Your actual car** (specific instance with its own color, mileage, etc.)

#### 2.3 PUBG Mobile Example

**Demonstrate:**
Refer students to `oop-tutorial.py` which shows:

**Before OOP:**

- Managing 5 players with 30+ variables
- Functions with many parameters
- Difficult to track what belongs to whom

**After OOP:**

- Clean Player class
- Each player is one object
- Easy to manage 100+ players
- Simple to add features

**Key OOP Concepts to Cover:**

1. **Class Definition**

   ```python
   class Player:
       # Blueprint for all players
   ```

2. **Constructor (`__init__`)**

   - Initializes new player
   - Sets starting values
   - Like "Create Character" button

3. **Attributes (Properties)**

   - `self.health`
   - `self.armor`
   - `self.position`
   - Like character stats

4. **Methods (Actions)**

   - `self.move()`
   - `self.shoot()`
   - `self.heal()`
   - Like character abilities

5. **Self Parameter**
   - Refers to "this specific player"
   - How each player tracks their own data

**Teaching Tips:**

- Live code the PUBG example
- Create 3-4 player objects in front of students
- Call methods on different players
- Show how each maintains its own state

**[NEED IMAGE: PUBG Mobile screenshot with character stats highlighted]**

#### 2.4 Why OOP for Neural Networks?

**Without OOP:**

```python
# Managing multiple layers manually
layer1_weights = [...]
layer1_biases = [...]
layer2_weights = [...]
layer2_biases = [...]
# Calculate layer 1
# Calculate layer 2
# Very messy!
```

**With OOP:**

```python
class NeuralNetwork:
    def __init__(self):
        self.layers = [...]

    def predict(self, inputs):
        # Clean, organized code
```

**Benefits for Neural Networks:**

- Each neuron is an object
- Each layer is an object
- Network is an object containing layers
- Easy to add/remove layers
- Reusable code

---

## 🏗️ Part 3: Multi-Layer Perceptron Architecture (45 minutes)

### Teaching Points

#### 3.1 MLP Structure

**Three Types of Layers:**

1. **Input Layer**
   - Not really a "layer" of neurons
   - Just holds the input data
   - Number of nodes = number of features
2. **Hidden Layer(s)**
   - Where the "magic" happens
   - Extract features and patterns
   - Can have multiple hidden layers (deep learning!)
   - Each neuron connected to all previous layer neurons
3. **Output Layer**
   - Produces final prediction
   - Number of nodes = number of classes/outputs
   - Binary classification: 1 or 2 nodes
   - Multi-class: multiple nodes

**[NEED IMAGE: MLP architecture diagram showing input (3 nodes), hidden (4 nodes), output (2 nodes) with all connections]**

#### 3.2 Network Notation

**Common notation:**

- Architecture: [3, 4, 2]
  - 3 input features
  - 4 hidden neurons
  - 2 output neurons

**Example: Email Spam Classifier**

- **Input layer:** 10 features (word frequencies)
- **Hidden layer 1:** 8 neurons
- **Hidden layer 2:** 4 neurons
- **Output layer:** 1 neuron (spam/not spam)
- **Architecture:** [10, 8, 4, 1]

#### 3.3 Connections and Weights

**Key Points:**

- **Fully Connected:** Each neuron connects to all neurons in next layer
- **Weight Matrix:** All connections have weights
- For layer with m inputs and n neurons: m × n weights
- **Number of parameters:** Can get very large!

**Example calculation:**

```
Architecture: [3, 4, 2]

Layer 1 (Input → Hidden):
- Inputs: 3
- Neurons: 4
- Weights: 3 × 4 = 12
- Biases: 4
- Total: 16 parameters

Layer 2 (Hidden → Output):
- Inputs: 4
- Neurons: 2
- Weights: 4 × 2 = 8
- Biases: 2
- Total: 10 parameters

Total network: 16 + 10 = 26 parameters
```

**[NEED IMAGE: Weight matrix visualization showing connections as a matrix]**

---

## 📐 Part 4: Mathematics of Forward Propagation (45 minutes)

### Teaching Points

#### 4.1 Layer-by-Layer Computation

**General formula for one layer:**

For layer ℓ:
$$\mathbf{z}^{(\ell)} = \mathbf{W}^{(\ell)} \mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)}$$
$$\mathbf{a}^{(\ell)} = f(\mathbf{z}^{(\ell)})$$

**Symbol explanation:**

- **ℓ:** Layer number (superscript in parentheses)
- **z^(ℓ):** Net input vector for layer ℓ
- **W^(ℓ):** Weight matrix for layer ℓ
- **a^(ℓ-1):** Activation output from previous layer (inputs to this layer)
- **b^(ℓ):** Bias vector for layer ℓ
- **f:** Activation function
- **a^(ℓ):** Activation output of layer ℓ
- **Bold letters:** Vectors/matrices (multiple values)

**Note:** a^(0) = input features

#### 4.2 Concrete Example: XOR Solution

**Network:** [2, 2, 1] - solves XOR!

**Step-by-step calculation:**

Given input: [1, 0]

**Hidden Layer Calculation:**

Weights W^(1):

```
W^(1) = [[1.0, 1.0],
         [1.0, 1.0]]
```

Biases: b^(1) = [-0.5, -1.5]

Neuron 1:

```
z₁ = (1.0 × 1) + (1.0 × 0) + (-0.5) = 0.5
a₁ = sigmoid(0.5) ≈ 0.62
```

Neuron 2:

```
z₂ = (1.0 × 1) + (1.0 × 0) + (-1.5) = -0.5
a₂ = sigmoid(-0.5) ≈ 0.38
```

Hidden layer output: [0.62, 0.38]

**Output Layer Calculation:**

Weights W^(2): [1.0, -2.0]
Bias: b^(2) = -0.5

```
z = (1.0 × 0.62) + (-2.0 × 0.38) + (-0.5) = -0.64
output = sigmoid(-0.64) ≈ 0.35 ≈ 0 (XOR output for [1,0])
```

**Work through all 4 XOR cases on board!**

**[NEED IMAGE: Step-by-step forward propagation diagram with values flowing through network]**

#### 4.3 Matrix Formulation

**Why matrices?**

- Efficient computation
- Clean code
- Leverage optimized libraries (NumPy)

**Example:**

```
Input: [1, 0]
W^(1) = [[1.0, 1.0],
         [1.0, 1.0]]
b^(1) = [-0.5, -1.5]

Matrix multiplication:
z^(1) = W^(1) × input + b^(1)
     = [[1.0, 1.0],  × [1]  + [-0.5]
        [1.0, 1.0]]    [0]    [-1.5]
     = [0.5, -0.5]
```

**Teach matrix dimensions:**

- (m × n) matrix × (n × 1) vector = (m × 1) vector
- Must match inner dimensions!

---

## 💻 Part 5: Implementation (60 minutes)

### Teaching Points

**Note to Instructor:** Show both versions side-by-side to emphasize OOP benefits.

#### 5.1 Without OOP (Procedural)

Refer to `mlp-implementation.py` - Section 1

**Show students:**

```python
# Many global variables
weights_layer1 = [...]
biases_layer1 = [...]
weights_layer2 = [...]
biases_layer2 = [...]

# Functions with many parameters
def forward_layer1(inputs, weights, biases):
    # ...

def forward_layer2(inputs, weights, biases):
    # ...

# Main prediction function ties it all together
def predict(inputs):
    hidden = forward_layer1(inputs, weights_layer1, biases_layer1)
    output = forward_layer2(hidden, weights_layer2, biases_layer2)
    return output
```

**Problems:**

- Hard to scale to more layers
- Lots of parameter passing
- Difficult to maintain
- Error-prone

#### 5.2 With OOP (Clean)

Refer to `mlp-implementation.py` - Section 2

**Show the clean version:**

```python
class MLP:
    def __init__(self, architecture):
        self.layers = []
        # Build layers automatically

    def forward(self, inputs):
        # Clean propagation through layers

    def predict(self, inputs):
        # Simple interface
```

**Benefits:**

- Easy to add layers
- Encapsulated logic
- Reusable
- Professional code structure

**Live Coding:**

- Build the MLP class step by step
- Test with XOR problem
- Show how easy it is to change architecture

#### 5.3 Testing and Validation

**Demonstrate:**

1. Create simple test cases
2. Verify XOR solution works
3. Show how to debug layer by layer
4. Print intermediate values

---

## 🎯 Part 6: Practical Application (45 minutes)

### Teaching Points

#### 6.1 Real Problem: Iris Flower Classification

**Dataset:** Iris flowers - classic ML dataset

- **3 species:** Setosa, Versicolor, Virginica
- **4 features:** Sepal length, sepal width, petal length, petal width
- **150 samples:** 50 of each species

**[NEED IMAGE: Iris flower types showing the three different species]**

**Why this problem?**

- Multi-class classification (3 classes)
- Real botanical data
- Manageable size
- Visual patterns

**[NEED IMAGE: Scatter plot of iris data showing clusters of three species]**

#### 6.2 Network Design

**Architecture choice:**

- **Input:** 4 features → 4 nodes
- **Hidden:** 8 neurons (good starting point)
- **Output:** 3 neurons (one per class)
- **Architecture:** [4, 8, 3]

**Activation functions:**

- Hidden layer: ReLU (fast, effective)
- Output layer: Softmax (for multi-class probability)

**Softmax function:**
$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

Converts outputs to probabilities that sum to 1.

#### 6.3 Implementation Walkthrough

Refer to `mlp-implementation.py` - Section 3

**Cover:**

1. Data loading and preprocessing
2. Network initialization
3. Making predictions
4. Interpreting results

**Important:** This lab focuses on understanding MLP structure, NOT training.
Training (backpropagation) comes in later labs.

---

## 📝 Part 7: Student Task (30 minutes)

**Note to Instructor:**
Assign the tasks from `student-task.py`.

### Task Overview:

1. **Task 1:** Implement digit recognition network (0-9)

   - Design appropriate architecture
   - Understand why certain architectures work better

2. **Task 2:** Experiment with different architectures

   - Compare shallow vs deep networks
   - Understand tradeoffs

3. **Task 3:** Build a game move predictor
   - Practical application
   - Multiple inputs and outputs

**Grading Rubric:**

- Correct implementation: 50%
- Architecture justification: 20%
- Testing and analysis: 15%
- Code quality and comments: 15%

---

## 🎓 Assessment Checklist

By the end of this lab, students should be able to:

- [ ] Explain why single neurons have limitations
- [ ] Describe MLP architecture components
- [ ] Understand forward propagation mathematics
- [ ] Explain OOP concepts (class, object, method)
- [ ] Implement a multi-layer perceptron in Python
- [ ] Apply MLP to classification problems
- [ ] Design appropriate network architectures

---

## 🔍 Common Student Difficulties

### Issue 1: OOP Confusion

**Problem:** Students don't understand `self`  
**Solution:** Use PUBG analogy - "self is like 'this player' knowing their own stats"

### Issue 2: Matrix Dimensions

**Problem:** Dimension mismatch errors  
**Solution:** Draw matrices on board, show dimension compatibility rules

### Issue 3: Forward Propagation Flow

**Problem:** Lost in the layer-to-layer calculations  
**Solution:** Step through one example with actual numbers on board

### Issue 4: Class vs Object

**Problem:** "When do I use the class vs the object?"  
**Solution:** "Class is the blueprint, object is the actual thing. You build objects from classes."

### Issue 5: Weight Initialization

**Problem:** "What values should weights start with?"  
**Solution:** Random small values (explain why in future training lab)

---

## 📚 Additional Resources

### For Students:

1. 3Blue1Brown - "Neural Networks" series (YouTube)
2. Python OOP Tutorial (Real Python)
3. Interactive MLP visualization tools

### For Instructor:

1. Nielsen - "Neural Networks and Deep Learning" (Chapters 1-2)
2. Goodfellow - "Deep Learning" (Chapter 6)
3. CS231n Stanford course notes

---

## 🕐 Time Management

| Activity              | Time        | Notes                            |
| --------------------- | ----------- | -------------------------------- |
| Review & Limitations  | 20 min      | Interactive discussion           |
| OOP Tutorial          | 60 min      | Live coding with PUBG example    |
| MLP Architecture      | 45 min      | Draw diagrams, explain structure |
| Mathematics           | 45 min      | Work through XOR on board        |
| Implementation        | 60 min      | Code along, show both versions   |
| Practical Application | 45 min      | Iris dataset example             |
| Student Task          | 30 min      | Independent work with support    |
| **Total**             | **4 hours** |                                  |

---

## 🎯 Next Lab Preview

In **Lab 03 - Training Neural Networks**, students will learn:

- Backpropagation algorithm
- Gradient descent optimization
- Loss functions
- How to train networks from scratch

**Preparation:** Students must understand forward propagation thoroughly. Lab 02 is the foundation!

---

**Version:** 1.0  
**Last Updated:** October 2025  
**Course:** Neural Networks - Computer Engineering  
**Institution:** [Your University Name]
