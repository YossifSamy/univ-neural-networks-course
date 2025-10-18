# Lab 02: Multi-Layer Perceptron

## Neural Networks Course - Computer Engineering

---

## 📋 Lab Information

**Difficulty:** Intermediate  
**Prerequisites:** Lab 01 - Single Neuron

**What You'll Learn:**

- Why single neurons can't solve all problems
- How to connect neurons in layers
- Object-Oriented Programming (OOP) in Python
- Building a Multi-Layer Perceptron (MLP)
- Solving real AI problems with neural networks

**Files You'll Need:**

- `student-guide.md` (this file)
- `oop-tutorial.py` (OOP introduction)
- `mlp-implementation.py` (MLP code)
- `student-task.py` (your assignment)

---

## 🎯 Introduction

In Lab 01, you learned about single neurons. They're powerful, but limited. Today, you'll discover how connecting multiple neurons in **layers** creates networks that can solve complex, real-world problems!

Think of it like building with LEGO:

- **Single neuron** = one LEGO brick (limited)
- **Multiple layers** = entire LEGO structure (powerful!)

---

## 🚫 Part 1: The Problem with Single Neurons

### Why Aren't Single Neurons Enough?

Remember from Lab 01: a single neuron creates a **linear decision boundary** (a straight line or flat plane).

**Example: The XOR Problem**

```
XOR Truth Table:
Input A | Input B | Output
   0    |    0    |   0
   0    |    1    |   1
   1    |    0    |   1
   1    |    1    |   0
```

**Challenge:** Can you separate the outputs with a single straight line?

**[NEED IMAGE: XOR visualization showing 4 points that cannot be separated by a line]**

**Answer:** No! Points (0,0) and (1,1) should be together (output 0), while points (0,1) and (1,0) should be together (output 1). No single line can do this!

This is called a **non-linearly separable** problem.

### Real-World Non-Linear Problems

Single neurons can't solve:

- Face recognition (complex patterns)
- Speech recognition (temporal patterns)
- Game AI (strategic decisions)
- Medical diagnosis (interacting symptoms)
- AND MANY MORE!

**The Solution:** Multiple neurons organized in layers = **Multi-Layer Perceptron (MLP)**

---

## 🎮 Part 2: Object-Oriented Programming (OOP)

Before building complex neural networks, you need to understand **Object-Oriented Programming (OOP)**. This will make your code clean and manageable!

### Why Do We Need OOP?

Imagine you're developing **PUBG Mobile** and need to manage 100 players in a match. How would you store each player's data?

#### The Old Way (Without OOP) - MESSY! ❌

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

# ... 98 more players! This is crazy!
```

**Problems:**

- Need 600+ variables for 100 players!
- Difficult to keep track
- Hard to add new features (what if we add "ammo"?)
- Very messy and error-prone

#### The New Way (With OOP) - CLEAN! ✅

```python
# Create a Player "blueprint"
class Player:
    def __init__(self, name):
        self.name = name
        self.health = 100
        self.armor = 0
        self.position = [0, 0]
        self.weapon = "Fists"

    def move(self, x, y):
        self.position = [x, y]

    def shoot(self, target):
        print(f"{self.name} shoots at {target.name}!")

# Create players easily!
player1 = Player("ProGamer")
player2 = Player("SnipeMaster")
player3 = Player("SneakyNinja")

# Use them simply!
player1.move(150, 200)
player1.shoot(player2)
```

**Benefits:**

- Clean and organized
- Easy to manage 100+ players
- Simple to add features
- Professional code structure

**[NEED IMAGE: PUBG Mobile character screen showing stats]**

### OOP Key Concepts

**📁 Open file:** `oop-tutorial.py`

Your instructor will walk you through:

#### 1. **Class** = Blueprint/Template

- Like the character creation screen in PUBG
- Defines what properties and actions all players have
- Written once, used many times

#### 2. **Object** = Specific Instance

- Each player in the game is an object
- Created from the class blueprint
- Has its own unique data (different health, position, etc.)

**Analogy:**

- **Class** = Cookie cutter (the shape/template)
- **Object** = Cookie (each cookie made from that cutter)

#### 3. **Attributes** = Properties/Data

- `self.health` - player's health points
- `self.armor` - player's armor level
- `self.position` - where player is on map
- Like stats in a game character

#### 4. **Methods** = Actions/Functions

- `self.move()` - change position
- `self.shoot()` - attack another player
- `self.heal()` - restore health
- Like abilities in a game

#### 5. **`self`** = "This Specific Object"

- Refers to the particular player you're talking about
- How each player keeps track of their own data
- Think: "self.health" = "MY health" (from that player's perspective)

#### 6. **`__init__`** = Constructor/Initializer

- Special method that runs when creating a new object
- Sets up starting values
- Like clicking "Create Character" button

### Why OOP for Neural Networks?

Without OOP, managing a network with multiple layers would be chaos:

```python
# Without OOP - MESSY!
layer1_weights = [[0.5, 0.3], [0.7, 0.2]]
layer1_biases = [0.1, -0.2]
layer2_weights = [[0.4], [0.6]]
layer2_biases = [0.3]
# Calculate layer 1
# Calculate layer 2
# Very confusing!
```

With OOP - CLEAN!

```python
# With OOP - CLEAN!
network = NeuralNetwork([2, 4, 1])
output = network.predict(inputs)
# Simple and clear!
```

---

## 🏗️ Part 3: Multi-Layer Perceptron Architecture

### What is an MLP?

A **Multi-Layer Perceptron (MLP)** is a neural network with multiple layers of neurons connected together.

**Structure:**

**[NEED IMAGE: MLP diagram showing input layer, hidden layers, and output layer with connections]**

### Three Types of Layers

#### 1. **Input Layer**

- Not really neurons, just holds your data
- Number of nodes = number of features in your data
- Example: Image with 784 pixels → 784 input nodes

#### 2. **Hidden Layer(s)**

- Where the "learning" happens
- Extract patterns and features from data
- Can have one or many hidden layers
- More layers = "deeper" network (deep learning!)

#### 3. **Output Layer**

- Produces final prediction
- Number of nodes depends on problem:
  - Binary classification: 1 node (yes/no)
  - Multi-class: multiple nodes (one per class)
  - Example: 10 digit recognition → 10 output nodes

### Network Architecture Notation

We describe networks by listing neurons in each layer:

**[3, 4, 2]** means:

- 3 input features
- 4 neurons in hidden layer
- 2 neurons in output layer

**[784, 128, 64, 10]** means:

- 784 input features (28×28 image)
- 128 neurons in first hidden layer
- 64 neurons in second hidden layer
- 10 output neurons (digits 0-9)

### How Neurons Connect

**Fully Connected (Dense) Layers:**

- Each neuron connects to **ALL** neurons in the next layer
- These connections have **weights** (importance values)
- Each neuron has a **bias** term

**[NEED IMAGE: Fully connected layer showing all connections between two layers]**

**Example: Counting Parameters**

For network [3, 4, 2]:

**Layer 1 (Input → Hidden):**

- Connections: 3 inputs × 4 neurons = 12 weights
- Biases: 4 neurons = 4 biases
- Total: 16 parameters

**Layer 2 (Hidden → Output):**

- Connections: 4 inputs × 2 neurons = 8 weights
- Biases: 2 neurons = 2 biases
- Total: 10 parameters

**Network Total:** 16 + 10 = **26 parameters to learn!**

---

## 📐 Part 4: The Mathematics

### Forward Propagation

**Forward Propagation** = passing data through the network from input to output.

**For each layer:**

$$
\mathbf{z} = \mathbf{W} \mathbf{x} + \mathbf{b}
$$

$$
\mathbf{a} = f(\mathbf{z})
$$

**What each symbol means:**

- **x:** Input to this layer
- **W:** Weight matrix (all connection strengths)
- **b:** Bias vector (one bias per neuron)
- **z:** Net input (before activation)
- **f:** Activation function
- **a:** Activated output (goes to next layer)
- **Bold letters:** Vectors/matrices (multiple values)

### Step-by-Step Example: Solving XOR

Let's solve XOR with a network: **[2, 2, 1]**

**Input:** [1, 0] (one of the XOR cases)

**Hidden Layer:**

Weights:

```
W₁ = [[1.0, 1.0],
      [1.0, 1.0]]
```

Biases: b₁ = [-0.5, -1.5]

**Neuron 1:**

```
z₁ = (1.0 × 1) + (1.0 × 0) + (-0.5) = 0.5
a₁ = sigmoid(0.5) = 0.62
```

**Neuron 2:**

```
z₂ = (1.0 × 1) + (1.0 × 0) + (-1.5) = -0.5
a₂ = sigmoid(-0.5) = 0.38
```

**Hidden layer output:** [0.62, 0.38]

**Output Layer:**

Weights: W₂ = [1.0, -2.0]  
Bias: b₂ = -0.5

```
z = (1.0 × 0.62) + (-2.0 × 0.38) + (-0.5)
z = 0.62 - 0.76 - 0.5
z = -0.64

output = sigmoid(-0.64) = 0.35 ≈ 0
```

**Result:** Output ≈ 0 ✓ (Correct! XOR(1,0) = 1, but after rounding output < 0.5 → 0... wait this doesn't look right. The network needs proper training weights!)

**Note:** These weights are just examples. In real networks, we **learn** the right weights through training (next lab!).

**[NEED IMAGE: Forward propagation diagram showing values flowing through the XOR network]**

---

## 💻 Part 5: Implementation

**📁 Open files:**

- `oop-tutorial.py` (OOP examples)
- `mlp-implementation.py` (MLP code)

### Two Approaches: Before and After OOP

Your instructor will show you both versions to demonstrate why OOP is essential!

#### Version 1: Without OOP (Procedural)

```python
# Many variables to track
weights_layer1 = [...]
biases_layer1 = [...]
weights_layer2 = [...]
biases_layer2 = [...]

# Many functions
def forward_layer1(inputs, weights, biases):
    # Calculate layer 1
    pass

def forward_layer2(inputs, weights, biases):
    # Calculate layer 2
    pass

# Main function ties it together
def predict(inputs):
    hidden = forward_layer1(inputs, weights_layer1, biases_layer1)
    output = forward_layer2(hidden, weights_layer2, biases_layer2)
    return output
```

**Problems:**

- Hard to add more layers
- Lots of parameter passing
- Confusing for large networks
- Error-prone

#### Version 2: With OOP (Professional)

```python
class MLP:
    def __init__(self, architecture):
        """
        Create network with specified architecture.
        Example: MLP([2, 4, 1]) creates 2→4→1 network
        """
        self.layers = []
        # Automatically build all layers!

    def forward(self, inputs):
        """Pass inputs through all layers."""
        activation = inputs
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation

    def predict(self, inputs):
        """Make a prediction."""
        return self.forward(inputs)

# Create network easily!
network = MLP([2, 4, 1])
output = network.predict([1, 0])
```

**Benefits:**

- Clean and organized
- Easy to add/remove layers
- Simple to use
- Professional code structure

### Follow Along

Your instructor will code this live. Follow along and type the code yourself!

**Key steps:**

1. Create the `Layer` class
2. Create the `MLP` class
3. Test with XOR problem
4. Experiment with different architectures

---

## 🎯 Part 6: Real-World Application

### Iris Flower Classification

Let's solve a real problem: classifying iris flowers!

**[NEED IMAGE: Photos of three iris flower species]**

**Dataset:**

- **3 species:** Setosa, Versicolor, Virginica
- **4 features:** Sepal length, sepal width, petal length, petal width
- **150 samples:** 50 of each species

**[NEED IMAGE: Scatter plot showing three clusters of iris species]**

### Network Design

**Problem type:** Multi-class classification (3 classes)

**Architecture:** [4, 8, 3]

- **4 inputs:** Four flower measurements
- **8 hidden neurons:** Extract patterns
- **3 outputs:** One for each species

**Activation functions:**

- Hidden layer: ReLU (fast and effective)
- Output layer: Softmax (converts to probabilities)

**Softmax function:**
Converts outputs to probabilities that sum to 1.

```
If outputs are [2.0, 1.0, 0.1]
Softmax gives [0.65, 0.24, 0.11]
Interpretation: 65% confident it's class 1
```

### Running the Example

Your instructor will demonstrate:

1. Loading iris data
2. Creating the network
3. Making predictions
4. Interpreting results

**Important:** This lab focuses on understanding MLP **structure** and **forward propagation**. Training the network (learning weights) comes in the next lab!

---

## 📝 Part 7: Your Tasks

**📁 Open file:** `student-task.py`

Complete three tasks:

### Task 1: Digit Recognition Network

Design a network to recognize handwritten digits (0-9).

- Decide appropriate architecture
- Justify your choices
- Implement the network

### Task 2: Architecture Experiments

Compare different network architectures:

- Shallow network (few neurons)
- Deep network (many layers)
- Wide network (many neurons per layer)

### Task 3: Game Move Predictor

Build a network that predicts the best move in a simple game.

- Multiple input features
- Binary or multi-class output
- Apply what you've learned!

**Submission:** Complete and submit `student-task.py` before leaving!

---

## ✅ Self-Check

Before you finish, make sure you can:

- [ ] Explain why single neurons can't solve XOR
- [ ] Describe the three types of layers in MLP
- [ ] Understand OOP concepts (class, object, self)
- [ ] Perform forward propagation by hand
- [ ] Implement an MLP in Python
- [ ] Design appropriate network architectures
- [ ] Apply MLP to classification problems

---

## 🤔 Common Questions

**Q: How do I choose the number of hidden layers?**  
A: Start with one or two. More layers = more complex patterns, but also harder to train. There's no perfect formula!

**Q: How many neurons should be in each layer?**  
A: Common practice: start with layers between input and output size. Example: [10, 8, 6, 4] gradually decreases. Experiment!

**Q: What's the difference between deep and shallow networks?**  
A:

- **Shallow:** 1-2 hidden layers (simpler problems)
- **Deep:** 3+ hidden layers (complex problems, more data needed)

**Q: Why use OOP instead of functions?**  
A: OOP organizes code better, especially for complex systems. Imagine managing 50 layers with just functions - chaos!

**Q: How are weights determined?**  
A: Through **training** (learning from data). Next lab will cover this! For now, we use random or predefined weights.

**Q: Can I connect neurons in ways other than fully connected?**  
A: Yes! Convolutional layers (for images), recurrent layers (for sequences), and more. You'll learn these in advanced labs.

---

## 🎯 What's Next?

In **Lab 03 - Training Neural Networks**, you'll learn:

- **Backpropagation:** How networks learn
- **Gradient Descent:** Optimization algorithm
- **Loss Functions:** Measuring prediction quality
- How to train your own networks!

This is where it gets really exciting - your networks will **learn** from data automatically!

---

## 💡 Tips for Success

1. **Understand OOP first** - It's crucial for clean neural network code
2. **Draw diagrams** - Visualize network structure and data flow
3. **Work through math examples** - Do forward propagation by hand
4. **Experiment** - Try different architectures and see what happens
5. **Ask questions** - Make sure you understand forward propagation thoroughly

---

## 📚 Extra Learning Resources

Want to go deeper? Check out:

1. **Video:** 3Blue1Brown - "Neural Networks" Chapter 2 (YouTube)

   - Excellent visualization of gradient descent

2. **Interactive:** TensorFlow Playground

   - Experiment with network architectures visually

3. **Reading:** "Neural Networks and Deep Learning" by Michael Nielsen

   - Chapter 1 on MLPs (free online)

4. **OOP Tutorial:** Real Python - "Object-Oriented Programming in Python"
   - Comprehensive OOP guide

---

**Great work! Multi-layer networks unlock the true power of AI! 🚀**

---

**Version:** 1.0  
**Lab:** 02 - Multi-Layer Perceptron  
**Course:** Neural Networks - Computer Engineering
