# Neural Networks Teaching Labs - AI Agent Instructions

## Project Overview

This is a **teaching materials repository** for a university Neural Networks course with **two progressive labs**. Each lab contains instructor guides, student guides, implementation examples, and student assignments. The codebase is designed for **education**, not production—code is intentionally verbose with extensive comments to support learning.

## Architecture & Structure

### Two-Lab Progression

1. **Lab 01 (3 hours)**: Single neurons using **procedural functions** - focuses on mathematical foundations
2. **Lab 02 (4 hours)**: Multi-layer perceptrons using **OOP classes** - introduces architecture design

**Critical**: Lab 01 deliberately uses functions (not classes) to keep focus on neuron mathematics. Lab 02 then introduces OOP to show how it simplifies complex networks. This is a pedagogical decision, not technical debt.

### File Organization Pattern

Each lab follows this structure:

```
LabXX-Topic/
├── instructor-guide.md       # Teaching strategies, timing, common issues
├── student-guide.md          # Learning material with analogies & examples
├── [tutorial].py            # Educational code (python-basics.py, oop-tutorial.py)
├── [topic]-implementation.py # Complete working examples
└── student-task.py          # Assignment template with TODOs
```

## Code Conventions

### Educational Code Style

- **Extreme verbosity**: Every function/method has detailed docstrings explaining purpose, parameters, and educational context
- **Progressive complexity**: Code builds from simple to complex within each file
- **Print statements everywhere**: Output explains each step for learning purposes
- **Manual implementations**: No NumPy/TensorFlow—students learn by implementing math directly
- **Predictable patterns**: Functions/methods follow consistent naming (e.g., `calculate_weighted_sum`, `apply_activation`, `forward`)

### Naming Patterns

- Functions: `verb_noun` pattern (`calculate_weighted_sum`, `neuron_predict`)
- Classes: PascalCase (`Player`, `Layer`, `MLP`)
- Variables: descriptive_names over brevity (`fruit_weights` not `fw`, `activation` not `act`)
- Test data: `[topic]_inputs`, `[topic]_weights`, `[topic]_bias` (e.g., `water_weights`, `xor_inputs`)

### Activation Functions

Standard implementations across both labs:

- `step_function(z)` - Binary decisions (Lab 01 focus)
- `sigmoid_function(z)` - Probability outputs
- `relu_function(z)` - Deep networks (Lab 02)
- `tanh_function(z)` - Zero-centered
- `leaky_relu_function(z)` - Alternative ReLU

### Network Architecture Notation

`[input_size, hidden1_size, hidden2_size, ..., output_size]`

- Example: `[2, 2, 1]` = 2 inputs, 1 hidden layer (2 neurons), 1 output (solves XOR)
- Example: `[4, 8, 3]` = 4 inputs, 1 hidden (8 neurons), 3 outputs (Iris classification)

## Key Pedagogical Analogies

### Lab 01: Water Bottle Temperature Classifier

Real-world example threading through the entire lab:

- Inputs: touch sensation, visual cues (steam/condensation), context
- Weights: `[0.7, 0.2, 0.1]` representing importance (touch matters most)
- Bias: `-2.0` (conservative threshold for "HOT")
- **Maintain this example** when modifying Lab 01 content

### Lab 02: PUBG Mobile for OOP

Game mechanics teach OOP concepts:

- Class = character creation screen (blueprint)
- Object = your actual character with specific stats
- Methods = actions (shoot, move, heal)
- Attributes = properties (health, position, weapon)
- `self` = "this specific player"
- **This analogy is highly effective**—students love gaming examples

### Lab 02: XOR as Motivation

XOR demonstrates single neuron limitations:

- Cannot be solved by a single neuron (not linearly separable)
- Requires hidden layer to create complex decision boundary
- Standard test case: `[[0,0], [0,1], [1,0], [1,1]]` → `[0, 1, 1, 0]`

## Development Workflows

### Creating New Examples

1. Start with a relatable real-world scenario (games, daily life, not abstract math)
2. Define inputs/outputs with clear scales and meanings
3. Choose logical weights reflecting real importance
4. Test with obvious cases (edge cases, typical cases, boundary cases)
5. Add detailed print statements showing calculations

### Modifying Student Tasks

Student tasks have this pattern:

```python
# TODO: [Clear instruction]
def function_name(params):
    """Docstring explaining what student should implement"""
    pass  # Student fills this

# Test code in comments - student uncomments after implementation
"""
print("Testing...")
result = function_name(...)
"""
```

### Testing Standards

- No formal test framework—educational code uses inline test cases
- Test cases progress: obvious → typical → edge cases
- Always show expected vs actual output with ✓/✗ symbols
- Format: `print(f"Input: {input} → Output: {output} → Expected: {expected} {status}")`

## Common Pitfalls to Avoid

1. **Don't add NumPy/TensorFlow**: Students must implement math manually for understanding
2. **Don't refactor Lab 01 to use OOP**: The progression from functions→OOP is intentional
3. **Don't remove "verbose" comments**: What seems obvious to AI agents isn't obvious to beginners
4. **Don't break the water bottle analogy**: It's threaded through Lab 01's entire narrative
5. **Don't skip activation function comparisons**: Students need to see different behaviors side-by-side

## Math Formatting Rules

**Critical**: All math blocks must use **multi-line format** for proper rendering:

❌ **Wrong** (inline block):

```markdown
$$z = \sum_{i=1}^{n} w_i \cdot x_i + b$$
```

✅ **Correct** (multi-line block):

```markdown
$$
z = \sum_{i=1}^{n} w_i \cdot x_i + b
$$
```

This ensures proper rendering in markdown viewers and maintains consistency across all documentation.

## Image Placeholders

Throughout guides, you'll see:

```markdown
**[NEED IMAGE: Description of needed image]**
```

These mark where visual aids enhance learning. When adding content:

- **Preserve these markers** unless you're actually adding images
- Common needs: neuron diagrams, activation function graphs, architecture visualizations, XOR plots

## Assessment Approach

Student tasks use point-based rubrics:

- Task implementation (largest portion)
- Thoughtful answers to reflection questions
- Working test cases
- Code comments and clarity

**Grading philosophy**: Understanding > speed > perfection. Instructors value explanations and experimentation over perfect accuracy.

## Extension Guidelines

When adding new labs or examples:

1. Follow the 5-part structure: Analogy → Math → Programming → Implementation → Tasks
2. Use gaming/daily life analogies over abstract concepts
3. Include detailed comparison sections (with OOP vs without, different activations, etc.)
4. Add reflection questions encouraging metacognition
5. Maintain the progression: simple → complex, concrete → abstract

## Key Files to Reference

- `README.md`: Complete course structure and teaching philosophy
- `QUICK-START-GUIDE.md`: Instructor reference with timing and tips
- `Lab01-Single-Neuron/neuron-implementation.py`: Canonical function-based implementation
- `Lab02-Multi-Layer-Perceptron/oop-tutorial.py`: OOP teaching reference (PUBG example)
- `Lab02-Multi-Layer-Perceptron/mlp-implementation.py`: Shows procedural→OOP transition

---

**Teaching Philosophy**: Progressive teaching through relatable analogies → mathematical foundation → practical implementation → hands-on tasks. Code is intentionally educational, not production-optimized.
