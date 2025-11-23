# Comprehensive Review Session - Interactive Activity Sheet

**Neural Networks Course - Computer Engineering**  
**Reviewing:** Labs 01, 02, and 03  
**Duration:** 90 minutes  
**Format:** Interactive group work + class discussions

---

## 🎯 Session Objectives

By the end of this session, you should be able to:

- Trace the complete pipeline: single neuron → MLP → training
- Perform forward propagation calculations
- Understand backpropagation conceptually
- Design and justify network architectures
- Apply all concepts to real problems

---

## 📚 PART 1: The Big Picture (10 minutes)

### Individual Concept Map (5 minutes)

**Your Task:** Connect the concepts from all three labs.

Fill in the blanks to complete the learning journey:

```
Lab 01: Single Neuron
├─ Inputs: x₁, x₂, ..., xₙ
├─ Calculation: z = _____________ (formula?)
├─ Activation: output = _____________ (what function?)
└─ Limitation: Can only solve _____________ problems

Lab 02: Multi-Layer Perceptron
├─ Why needed? Single neurons can't solve _____________
├─ Architecture: [input, _______, output]
├─ Forward propagation: Pass data through _____________
└─ OOP benefit: _____________

Lab 03: Training
├─ Problem: Finding optimal _____________
├─ Loss function: Measures _____________
├─ Gradient descent: Go _____________ (which direction?)
└─ Backpropagation: Calculate _____________ for all layers
```

### Pair Discussion (5 minutes)

**Partner with someone nearby.**

**Discuss:** "If someone asked you to explain neural networks in 2 minutes, what would you say? Practice on your partner!"

```
My 2-minute explanation:
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________
```

---

## 🧠 PART 2: Lab 01 Review - Single Neuron (20 minutes)

### Group Challenge: Water Bottle 2.0 (15 minutes)

**Form groups of 3-4 students.**

**New Scenario:** You're designing a neuron to classify if a **student is studying effectively** or **wasting time**.

**Available Inputs (0-10 scale):**

1. Phone screen time in last hour (0=none, 10=constant)
2. Number of pages read (0=none, 10=many)
3. Social media notifications (0=none, 10=many)
4. Focus level reported (0=distracted, 10=focused)

**Your Tasks:**

**Task 1: Design Your Neuron (7 min)**

Choose weights for each input:

```
w₁ (phone time) = _____ (positive or negative? Why?)
w₂ (pages read) = _____ (positive or negative? Why?)
w₃ (notifications) = _____ (positive or negative? Why?)
w₄ (focus level) = _____ (positive or negative? Why?)
b (bias) = _____
```

**Justification:**

```
Why we chose these weights:
_______________________________________________________________
_______________________________________________________________
```

**Task 2: Test Your Neuron (5 min)**

Test on these three students:

**Student A (The Focused One):**

- Inputs: [1, 8, 0, 9] → Phone time=1, Pages=8, Notifications=0, Focus=9

```
z = (w₁ × 1) + (w₂ × 8) + (w₃ × 0) + (w₄ × 9) + b
  = _______
output = step(z) = _______
```

**Student B (The Distracted One):**

- Inputs: [9, 1, 10, 2] → Phone time=9, Pages=1, Notifications=10, Focus=2

```
z = _______
output = _______
```

**Student C (The Mixed Case):**

- Inputs: [4, 5, 5, 6]

```
z = _______
output = _______
```

**Task 3: Activation Function Choice (3 min)**

Which activation function would work best for this problem?

- [ ] Step function (binary: studying or not)
- [ ] Sigmoid (probability: 0-100% effective)
- [ ] ReLU (intensity of studying)

**Our choice:** ******\_****** because **********\_\_\_**********

### Class Discussion (5 minutes)

**Each group shares:** Your weight choices and why.

---

## 🏗️ PART 3: Lab 02 Review - MLP Architecture (25 minutes)

### Individual Quick Quiz (5 minutes)

**True or False:**

1. A single neuron can solve the XOR problem. **T / F**
2. In OOP, `self` refers to the specific object instance. **T / F**
3. A network [4, 8, 3] has 8 input features. **T / F**
4. Forward propagation goes from output to input. **T / F**
5. More hidden layers always means better performance. **T / F**

**Short Answer:**

6. What does `[2, 4, 3, 1]` mean in network architecture notation?

```
_______________________________________________________________
```

7. Why can't a single neuron solve XOR?

```
_______________________________________________________________
```

### Group Design Challenge (15 minutes)

**Stay in your groups.**

**Scenario:** You're building an AI for **Egyptian license plate recognition**.

**Input:** Image of license plate (let's simplify: 20 features extracted)
**Output:** 3 Arabic letters + 4 digits

**Example:** أ ه م 1234 (Alif, Ha, Meem, 1234)

**Your Tasks:**

**Task 1: Analyze the Problem (5 min)**

```
How many input features? _____
How many possible Arabic letters? _____ (there are 28!)
How many possible digits per position? _____ (0-9)
How many total outputs needed? _____
```

**Hint:** For letters, you might use one-hot encoding (28 outputs for first letter, 28 for second, etc.)

**Task 2: Design Architecture (7 min)**

Design TWO different architectures:

**Conservative Design:**

```
Architecture: [_____, _____, _____]
Justification: ________________________________________
_______________________________________________________________
```

**Ambitious Design:**

```
Architecture: [_____, _____, _____, _____]
Justification: ________________________________________
_______________________________________________________________
```

**Task 3: Calculate Parameters (3 min)**

For your conservative design:

```
Layer 1 weights: _____ × _____ = _____
Layer 1 biases: _____
Layer 2 weights: _____ × _____ = _____
Layer 2 biases: _____
Total parameters: _____
```

### OOP Quick Code Review (5 minutes)

**Find and fix the errors in this code:**

```python
class NeuralLayer:
    def init(num_neurons):
        weights = [[0.5] * num_neurons]
        biases = [0.0] * num_neurons

    def forward(inputs):
        outputs = []
        for i in range(len(weights)):
            z = biases[i]
            for j in range(len(inputs)):
                z += weights[i][j] * inputs[j]
            outputs.append(sigmoid(z))
        return outputs
```

**Errors found:**

1. ***
2. ***
3. ***

---

## 🎓 PART 4: Lab 03 Review - Training Concepts (25 minutes)

### Mountain Climbing Analogy Check (5 minutes)

**Complete the analogy table:**

| Mountain Hiking | Neural Network Training | Your Explanation   |
| --------------- | ----------------------- | ------------------ |
| Your position   | ******\_******          | ********\_******** |
| Altitude        | ******\_******          | ********\_******** |
| ******\_******  | Optimal weights         | ********\_******** |
| Feeling slope   | ******\_******          | ********\_******** |
| ******\_******  | Updating weights        | ********\_******** |
| Step size       | ******\_******          | ********\_******** |

### Gradient Descent Simulation (10 minutes)

**Pair activity - you'll simulate gradient descent on paper!**

**The Problem:** Find minimum of function: $f(w) = (w - 3)^2$

The minimum is at w=3 (where f(w)=0), but you don't know that yet!

**Starting point:** w = 0  
**Learning rate:** 0.5  
**Gradient formula:** $\frac{df}{dw} = 2(w - 3)$

**Your job:** Update w step by step until you reach the minimum.

**Iteration 1:**

```
Current w: 0
Gradient: 2(0 - 3) = _____
Update: w_new = w_old - (learning_rate × gradient)
      = 0 - (0.5 × _____) = _____
New w: _____
```

**Iteration 2:**

```
Current w: _____
Gradient: 2(_____ - 3) = _____
Update: w_new = _____ - (0.5 × _____) = _____
New w: _____
```

**Iteration 3:**

```
Current w: _____
Gradient: _____
New w: _____
```

**Iteration 4:**

```
Current w: _____
Gradient: _____
New w: _____
```

**Question:** Are you getting closer to w=3? Yes / No

**What would happen if learning rate was:**

- Too small (0.01)? ******************\_\_\_\_******************
- Too large (5.0)? ********************\_********************

### Loss Function Understanding (5 minutes)

**Given these predictions and actual values:**

| Example | Actual (y) | Predicted (ŷ) | Error  |
| ------- | ---------- | ------------- | ------ |
| 1       | 1          | 0.9           | **\_** |
| 2       | 0          | 0.2           | **\_** |
| 3       | 1          | 0.6           | **\_** |
| 4       | 0          | 0.1           | **\_** |

**Calculate Mean Squared Error (MSE):**

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

```
MSE = (1/4) × [(_____)² + (_____)² + (_____)² + (_____)²]
    = (1/4) × [_____ + _____ + _____ + _____]
    = _____
```

**Lower MSE is better or worse?** ******\_******

### Backpropagation Concept Check (5 minutes)

**No calculations - just understanding!**

Answer these conceptual questions:

**1. What does backpropagation do?**

```
_______________________________________________________________
```

**2. Why is it called "back" propagation?**

```
_______________________________________________________________
```

**3. Forward propagation gives us predictions. Backpropagation gives us:**

- [ ] Better predictions
- [ ] Gradients to update weights
- [ ] The loss value
- [ ] New training data

**4. In a 3-layer network [4, 8, 2], which layer's gradients are calculated first?**

- [ ] Input layer
- [ ] Hidden layer
- [ ] Output layer

**5. What happens if we don't use backpropagation?**

```
_______________________________________________________________
```

---

## 🔬 PART 5: Putting It All Together (10 minutes)

### Complete Pipeline Exercise

**Form groups of 3-4.**

**Scenario:** Training a network to recognize handwritten Arabic digits (٠-٩).

**Complete this pipeline:**

**Step 1: Architecture Design**

```
Input size: _____ (how many pixels? assume 28×28 image)
Hidden layers: [_____, _____] (your choice)
Output size: _____ (how many digits 0-9?)
Complete architecture: [_____, _____, _____, _____]
```

**Step 2: Forward Propagation (conceptual)**

```
1. Take image pixels as input → feed to layer 1
2. Layer 1 calculates: z = _____________ then a = _____________
3. Pass a to layer 2, which calculates: _____________
4. Final layer outputs: _____________ (what does this represent?)
```

**Step 3: Loss Calculation**

```
If actual digit is ٥ (5) but network predicted ٣ (3):
- Loss is _____ (high/low)
- We need to _____ (increase/decrease) loss
```

**Step 4: Backpropagation (conceptual)**

```
1. Calculate gradient at output layer
2. Propagate gradients _____________ (which direction?)
3. Calculate gradients for all _____________
```

**Step 5: Weight Update**

```
For each weight: w_new = w_old - _____________
Direction to move: _____________ (uphill/downhill)
```

**Step 6: Repeat**

```
Keep iterating until: _____________
```

---

## 🧩 BONUS CHALLENGE: Debug the Training Loop (if time permits)

**This training loop has 5 bugs. Find and fix them!**

```python
class SimpleNetwork:
    def __init__(self):
        self.weights = [0.5, 0.3]
        self.bias = 0.1

    def forward(inputs):
        z = bias
        for i in range(len(inputs)):
            z += weights[i] * inputs[i]
        return sigmoid(z)

    def train(self, X_train, y_train, epochs=100, learning_rate=0.1):
        for epoch in range(epochs):
            total_loss = 0

            for i in range(len(X_train)):
                # Forward pass
                prediction = forward(X_train[i])

                # Calculate loss
                loss = (y_train[i] - prediction) ** 2
                total_loss -= loss

                # Calculate gradient (simplified)
                gradient = -2 * (y_train[i] - prediction)

                # Update weights (wrong direction!)
                for j in range(len(self.weights)):
                    self.weights[j] += learning_rate * gradient * X_train[i][j]

                self.bias += learning_rate * gradient

            if epoch % 10:
                print(f"Epoch {epoch}: Loss = {total_loss}")
```

**Bugs Found:**

1. Line **\_: **********************\_\_\_************************
2. Line **\_: **********************\_\_\_************************
3. Line **\_: **********************\_\_\_************************
4. Line **\_: **********************\_\_\_************************
5. Line **\_: **********************\_\_\_************************

---

## ✅ Self-Assessment Checklist

**Check what you can confidently do:**

### Lab 01 - Single Neuron

- [ ] I can calculate weighted sum and apply activation functions
- [ ] I understand what different activation functions do
- [ ] I can implement a neuron in Python
- [ ] I know the limitations of single neurons

### Lab 02 - Multi-Layer Perceptron

- [ ] I can explain why we need multiple layers
- [ ] I understand OOP concepts (class, object, self)
- [ ] I can design network architectures for problems
- [ ] I can trace forward propagation through a network
- [ ] I can calculate the number of parameters

### Lab 03 - Training

- [ ] I understand the mountain climbing analogy
- [ ] I can explain what loss functions measure
- [ ] I understand how gradient descent works
- [ ] I know what backpropagation does (conceptually)
- [ ] I can implement a basic training loop

### Integration

- [ ] I can trace the complete pipeline: design → forward → loss → backward → update
- [ ] I can apply neural networks to real-world problems
- [ ] I understand when to use which activation function
- [ ] I can debug simple neural network code

---

## 🤔 Reflection Questions

**1. What was the most challenging concept across all three labs?**

```
_______________________________________________________________
_______________________________________________________________
```

**2. Which lab made the most sense to you? Why?**

```
_______________________________________________________________
```

**3. How do the three labs connect? Draw a simple flowchart:**

```




```

**4. What's one "aha!" moment you had during this review?**

```
_______________________________________________________________
```

**5. What concept do you still need more practice with?**

```
_______________________________________________________________
```

**6. How would you explain neural networks to your family/friends?**

```
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________
```

---

## 🎯 Key Formulas Reference

**Single Neuron:**

$$
z = \sum_{i=1}^{n} w_i x_i + b
$$

$$
\text{output} = f(z)
$$

**Activation Functions:**

- Step: $f(z) = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{otherwise} \end{cases}$
- Sigmoid: $f(z) = \frac{1}{1 + e^{-z}}$
- ReLU: $f(z) = \max(0, z)$

**Loss (MSE):**

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

**Gradient Descent:**

$$
w_{\text{new}} = w_{\text{old}} - \alpha \frac{\partial L}{\partial w}
$$

Where α = learning rate, L = loss

---

## 📝 Action Items After This Session

**Before next class:**

- [ ] Review any concepts you marked as needing practice
- [ ] Redo one calculation from each lab by hand
- [ ] Read your code from previous labs - can you understand it?
- [ ] Prepare questions for instructor

**To deepen understanding:**

- [ ] Watch 3Blue1Brown Neural Networks series
- [ ] Try implementing XOR from scratch without looking at notes
- [ ] Experiment with different learning rates in your code
- [ ] Discuss with classmates what you found challenging

---

**Remember:** Neural networks are complex! It's okay to not understand everything perfectly. The key is to keep practicing and asking questions.

**You've learned A LOT in three labs. Be proud of your progress! 🚀**

---

**Version:** 1.0  
**Course:** Neural Networks - Computer Engineering  
**Duration:** 90 minutes  
**Covers:** Labs 01, 02, 03
