# Lab 02 Review Session - Interactive Activity Sheet

**Neural Networks Course - Computer Engineering**  
**Duration:** 90 minutes  
**Format:** Interactive group work + class discussions

---

## 🎯 Session Objectives

By the end of this session, you should be able to:

- Explain why single neurons have limitations
- Understand multi-layer perceptron architecture
- Master OOP concepts in Python
- Trace forward propagation through networks
- Apply MLPs to real problems

---

## 📚 PART 1: Single Neuron Limitations (15 minutes)

### Individual Reflection (3 minutes)

**Question 1:** In your own words, what does a single neuron do?

```
Your answer:
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________
```

**Question 2:** What is the mathematical formula for a neuron's output?

```
Your answer:
_______________________________________________________________
```

### Group Discussion Activity (12 minutes)

**Form groups of 3-4 students.**

**Activity: The XOR Challenge**

Your group has been given this problem:

| Input A | Input B | Desired Output |
| ------- | ------- | -------------- |
| 0       | 0       | 0              |
| 0       | 1       | 1              |
| 1       | 0       | 1              |
| 1       | 1       | 0              |

**Task 1:** Plot these 4 points on a graph (5 min)

- Use (Input A, Input B) as coordinates
- Mark points with Output=0 as ✓
- Mark points with Output=1 as O

**Task 2:** Try to draw ONE straight line that separates ✓ from O (3 min)

**Task 3:** Discuss with your group (4 min):

1. Were you successful? Why or why not?
2. What does this tell you about single neurons?
3. Can you think of a real-world problem that has this same issue?

**Group Reporter:** Be ready to share your findings!

```
Our group's conclusion:
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________
```

---

## 🎮 PART 2: Object-Oriented Programming (25 minutes)

### Individual Quick Check (5 minutes)

**Match the OOP concept with its PUBG Mobile analogy:**

| Concept   | Letter | PUBG Mobile Analogy                                  |
| --------- | ------ | ---------------------------------------------------- |
| Class     | \_\_\_ | A. Your actual character "ProGamer" with 87 health   |
| Object    | \_\_\_ | B. The action of moving from one position to another |
| Attribute | \_\_\_ | C. The character creation screen/blueprint           |
| Method    | \_\_\_ | D. Your character's health value (100, 87, 45, etc.) |
| `self`    | \_\_\_ | E. "This specific player" - knowing YOUR stats       |

**Fill in the blanks:**

```python
class Player:
    def __init__(______, name):
        ______.name = name
        ______.health = 100

    def shoot(______, target):
        ______.ammo -= 1
        target.health -= 20
```

Words to use: `self` (use it 5 times!)

### Pair Programming Activity (20 minutes)

**Pair up with a classmate.**

**Scenario:** You're building a simple café management system.

**Task 1: Design the Class (10 min)**

Complete this class design:

```python
class CoffeeDrink:
    """Blueprint for coffee drinks at our café"""

    def __init__(self, name, size, price):
        # TODO: Initialize attributes
        # Attributes needed: name, size, price, temperature (default "hot")
        pass

    def add_ice(self):
        # TODO: Change temperature to "iced"
        # Print: "[drink name] is now iced!"
        pass

    def get_info(self):
        # TODO: Print drink details
        # Format: "Medium Latte - 25 LE (hot)"
        pass
```

**Task 2: Test Your Class (5 min)**

Create these drinks and test:

```python
# Create a medium latte
drink1 = CoffeeDrink("Latte", "Medium", 25)

# Create a large cappuccino
drink2 = CoffeeDrink("Cappuccino", "Large", 30)

# Make the latte iced
# Show both drinks' info
```

**Task 3: Reflection (5 min)**

Discuss with your partner:

1. How is `self` used in the class?
2. What's the difference between `CoffeeDrink` and `drink1`?
3. How would you add a new method `add_sugar()`?

```
Key insight we learned:
_______________________________________________________________
_______________________________________________________________
```

---

## 🧠 PART 3: MLP Architecture Understanding (20 minutes)

### Group Whiteboard Activity (15 minutes)

**Form groups of 3-4. Each group gets whiteboard space.**

**Challenge:** Design an MLP for the following problem:

**Problem:** Predict whether a student will pass the Neural Networks course

**Available Data (Inputs):**

1. Attendance percentage (0-100)
2. Lab assignment scores average (0-100)
3. Midterm exam score (0-100)
4. Hours studied per week (0-40)

**Output:**

- Pass (1) or Fail (0)

**Your Tasks:**

**Step 1:** Design the architecture (7 min)

- How many input nodes? **\_**
- How many hidden layers? **\_**
- How many neurons per hidden layer? **\_**
- How many output nodes? **\_**
- Write your architecture as: [___, ___, ___]

**Step 2:** Draw your network (5 min)

- Draw all layers as circles
- Show connections between layers
- Label each layer

**Step 3:** Calculate parameters (3 min)

- How many weights in layer 1? **\_**
- How many biases in layer 1? **\_**
- How many weights in layer 2? **\_**
- How many biases in layer 2? **\_**
- Total parameters? **\_**

**Formula reminder:**

```
Weights between layers = (neurons in previous) × (neurons in current)
Biases = number of neurons in current layer
```

### Class Discussion (5 minutes)

**Each group shares:**

1. Your architecture choice
2. Why you chose that number of hidden neurons
3. One thing you found challenging

---

## 🔢 PART 4: Forward Propagation Trace (20 minutes)

### Individual Challenge (10 minutes)

**Given this tiny network:** [2, 2, 1]

**Layer 1 (Input → Hidden):**

```
Weights:  [[0.5, 0.3],
           [0.2, 0.4]]
Biases:   [-0.1, -0.2]
Activation: Step function (output 1 if z ≥ 0, else 0)
```

**Layer 2 (Hidden → Output):**

```
Weights:  [[0.6],
           [0.8]]
Bias:     [-0.3]
Activation: Step function
```

**Input:** [1.0, 0.5]

**Your Task: Trace the forward propagation!**

**Hidden Layer - Neuron 1:**

```
z₁ = (0.5 × 1.0) + (0.3 × 0.5) + (-0.1) = _______
a₁ = step(z₁) = _______
```

**Hidden Layer - Neuron 2:**

```
z₂ = (0.2 × 1.0) + (0.4 × 0.5) + (-0.2) = _______
a₂ = step(z₂) = _______
```

**Hidden layer output:** [_____, _____]

**Output Layer:**

```
z_out = (0.6 × a₁) + (0.8 × a₂) + (-0.3) = _______
output = step(z_out) = _______
```

**Final prediction:** **\_\_\_**

### Pair Verification (10 minutes)

**Partner up and:**

1. Compare your calculations (5 min)
2. If different, find where you disagree (3 min)
3. Together, verify the correct answer (2 min)

**Reflection question:** What would happen if we changed the input to [0, 1]? Try it!

---

## 🎯 PART 5: Real-World Application (10 minutes)

### Group Brainstorming Activity

**Stay in your groups.**

**Scenario:** You work at a tech startup in Cairo. Your boss asks you to build an AI solution.

**Choose ONE problem:**

**Option A: Taxi Fare Predictor**

- Inputs: Distance (km), time of day (0-23), traffic level (1-5), day of week
- Output: Estimated fare (LE)

**Option B: Food Delivery Time Estimator**

- Inputs: Distance, number of items, restaurant prep time, traffic, weather
- Output: Delivery time (minutes)

**Option C: Student Grade Predictor**

- Inputs: Study hours, previous quiz scores, attendance, assignment completion
- Output: Expected final grade (A, B, C, D, F)

**Your Group's Tasks (7 min):**

1. **Choose** one problem
2. **Design** MLP architecture [input, hidden, output]
3. **Justify** your design:
   - Why that many inputs?
   - Why that hidden layer size?
   - Why that output format?
4. **Identify** one challenge in solving this problem

```
Our chosen problem: _______________________________________

Our architecture: [_____, _____, _____]

Our justification:
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________

One challenge:
_______________________________________________________________
```

### Quick Presentations (3 min)

2-3 groups share their designs (30 seconds each)

---

## 🧩 FINAL CHALLENGE: Code Debugging (Bonus if time permits)

### Individual/Pair Work

**This code has bugs! Find and fix them:**

```python
class SimpleNeuron:
    def init(self, num_inputs):
        weights = [0.5] * num_inputs
        bias = 0.0

    def predict(inputs):
        z = bias
        for i in range(len(inputs)):
            z += weights[i] * inputs[i]

        if z >= 0:
            return 1
        else:
            return 0

# Test
neuron = SimpleNeuron(3)
result = neuron.predict([1, 0, 1])
print(result)
```

**Bugs to find:**

1. Bug in `__init__` method: ****************\_****************
2. Bug with `self`: ********************\_********************
3. Bug with attributes: ******************\_******************

**Fixed code:**

```python
# Write your corrected version here




```

---

## ✅ Self-Assessment Checklist

Check what you can confidently do:

- [ ] I can explain why single neurons can't solve XOR
- [ ] I understand the difference between class and object
- [ ] I can use `self` correctly in Python classes
- [ ] I can design an MLP architecture for a given problem
- [ ] I can trace forward propagation through a network
- [ ] I can calculate the number of parameters in a network
- [ ] I can apply MLP concepts to real-world problems

**Areas I need more practice:**

```
_______________________________________________________________
_______________________________________________________________
```

---

## 🤔 Reflection Questions (End of Session)

**1. What was the most challenging concept today?**

```
_______________________________________________________________
```

**2. What's one "aha!" moment you had?**

```
_______________________________________________________________
```

**3. How would you explain OOP to a friend who's learning programming?**

```
_______________________________________________________________
_______________________________________________________________
```

**4. What's one question you still have?**

```
_______________________________________________________________
```

---

## 📝 Take-Home Challenge (Optional)

Implement a simple MLP class that can:

1. Store weights and biases for 2 layers
2. Perform forward propagation
3. Make predictions on XOR problem

Bring your code to the next session for peer review!

---

**Remember:** The best way to learn is by doing and discussing with peers. Don't hesitate to ask questions!

**Version:** 1.0  
**Course:** Neural Networks - Computer Engineering  
**Duration:** 90 minutes
