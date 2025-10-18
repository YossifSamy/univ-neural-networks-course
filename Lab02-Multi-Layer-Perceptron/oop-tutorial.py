"""
Lab 02: Object-Oriented Programming Tutorial
=============================================

This file teaches Object-Oriented Programming (OOP) using PUBG Mobile as an example.
Students are familiar with PUBG, making OOP concepts easier to understand!

Author: Neural Networks Course
Lab: 02 - Multi-Layer Perceptron
"""

print("=" * 70)
print(" " * 15 + "OBJECT-ORIENTED PROGRAMMING TUTORIAL")
print(" " * 20 + "Using PUBG Mobile as Example")
print("=" * 70)

# =============================================================================
# PART 1: THE PROBLEM - MANAGING GAME CHARACTERS WITHOUT OOP
# =============================================================================

print("\n" + "=" * 70)
print("PART 1: THE PROBLEM - WITHOUT OOP (Don't do this!)")
print("=" * 70)

print("""
Imagine you're developing PUBG Mobile and need to track 100 players in a match.
Without OOP, you'd need to do this:
""")

# Player 1 data
player1_name = "ProGamer"
player1_health = 100
player1_armor = 75
player1_position_x = 150.5
player1_position_y = 200.3
player1_weapon = "M416"
player1_ammo = 40
player1_kills = 3
player1_is_alive = True

# Player 2 data
player2_name = "SnipeMaster"
player2_health = 80
player2_armor = 50
player2_position_x = 300.7
player2_position_y = 450.2
player2_weapon = "AWM"
player2_ammo = 15
player2_kills = 5
player2_is_alive = True

# Player 3 data
player3_name = "SneakyNinja"
player3_health = 60
player3_armor = 0
player3_position_x = 500.1
player3_position_y = 100.8
player3_weapon = "UMP45"
player3_ammo = 25
player3_kills = 1
player3_is_alive = True

# ... and 97 more players! That's 900+ variables! 😱

print(f"\nPlayer 1: {player1_name}")
print(f"  Health: {player1_health}, Armor: {player1_armor}")
print(f"  Position: ({player1_position_x}, {player1_position_y})")
print(f"  Weapon: {player1_weapon}, Ammo: {player1_ammo}")
print(f"  Kills: {player1_kills}, Alive: {player1_is_alive}")

print(f"\nPlayer 2: {player2_name}")
print(f"  Health: {player2_health}, Armor: {player2_armor}")
print(f"  Position: ({player2_position_x}, {player2_position_y})")
print(f"  Weapon: {player2_weapon}, Ammo: {player2_ammo}")
print(f"  Kills: {player2_kills}, Alive: {player2_is_alive}")

print("\n❌ PROBLEMS WITH THIS APPROACH:")
print("  1. Need 900+ variables for 100 players (9 variables each)")
print("  2. Extremely hard to manage and track")
print("  3. Functions need many parameters")
print("  4. Very error-prone (easy to mix up variables)")
print("  5. Difficult to add new features")
print("  6. Code duplication everywhere")

# Example of a function without OOP
def player_move_old(name, pos_x, pos_y, new_x, new_y):
    """Move a player - messy way"""
    print(f"{name} moves from ({pos_x}, {pos_y}) to ({new_x}, {new_y})")
    return new_x, new_y

def player_shoot_old(shooter_name, shooter_ammo, target_name, target_health, target_armor, damage=20):
    """Player shoots another player - very messy!"""
    if shooter_ammo <= 0:
        print(f"{shooter_name} has no ammo!")
        return shooter_ammo, target_health, target_armor
    
    # Calculate damage
    absorbed = min(damage, target_armor)
    health_damage = damage - absorbed
    
    target_armor -= absorbed
    target_health -= health_damage
    shooter_ammo -= 1
    
    print(f"{shooter_name} shoots {target_name}!")
    print(f"  Damage: {damage} (Armor absorbed: {absorbed}, Health: {health_damage})")
    print(f"  {target_name}'s remaining - Health: {target_health}, Armor: {target_armor}")
    
    return shooter_ammo, target_health, target_armor

# Using the messy functions
print("\n--- Testing Messy Approach ---")
player1_position_x, player1_position_y = player_move_old(
    player1_name, player1_position_x, player1_position_y, 200, 250
)

player1_ammo, player2_health, player2_armor = player_shoot_old(
    player1_name, player1_ammo, player2_name, player2_health, player2_armor, 25
)

print(f"\nAfter actions:")
print(f"{player1_name}: Ammo={player1_ammo}")
print(f"{player2_name}: Health={player2_health}, Armor={player2_armor}")

print("\n💡 There MUST be a better way... Enter OOP!")


# =============================================================================
# PART 2: INTRODUCING CLASSES AND OBJECTS
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 2: THE SOLUTION - WITH OOP (The right way!)")
print("=" * 70)

print("""
OOP lets us create a "blueprint" (Class) for players, then create
individual players (Objects) from that blueprint.

Think of it like:
  CLASS = Cookie cutter (the shape/template)
  OBJECT = Cookie (individual cookie made from that cutter)

Or in PUBG terms:
  CLASS = Character creation screen (the template)
  OBJECT = Your actual character in the game (with YOUR specific stats)
""")

# Define the Player class (blueprint)
class Player:
    """
    Blueprint for creating PUBG Mobile players.
    
    This defines what every player should have (attributes)
    and what every player can do (methods).
    """
    
    def __init__(self, name, starting_position=(0, 0)):
        """
        Constructor: Initialize a new player when created.
        This runs automatically when you create a new Player object.
        
        Think of this as the "Create Character" button in PUBG.
        
        Parameters:
            name (str): Player's name
            starting_position (tuple): Starting (x, y) coordinates
        """
        # Attributes (properties) - each player has these
        self.name = name
        self.health = 100  # Everyone starts with 100 health
        self.armor = 0     # Start with no armor
        self.position = list(starting_position)
        self.weapon = "Fists"  # Start with no weapon
        self.ammo = 0
        self.kills = 0
        self.is_alive = True
        
        print(f"✓ Player '{self.name}' created at position {self.position}")
    
    def move(self, new_x, new_y):
        """
        Move the player to a new position.
        
        This is a METHOD - an action the player can perform.
        """
        old_pos = self.position.copy()
        self.position = [new_x, new_y]
        print(f"{self.name} moved from {old_pos} to {self.position}")
    
    def pickup_weapon(self, weapon_name, ammo_count):
        """Pick up a weapon and ammo."""
        self.weapon = weapon_name
        self.ammo = ammo_count
        print(f"{self.name} picked up {weapon_name} with {ammo_count} ammo!")
    
    def pickup_armor(self, armor_value):
        """Pick up armor."""
        self.armor = armor_value
        print(f"{self.name} picked up armor (Level {armor_value})!")
    
    def shoot(self, target, damage=20):
        """
        Shoot another player.
        
        Parameters:
            target (Player): The player being shot (another Player object!)
            damage (int): Amount of damage to deal
        """
        # Check if shooter has ammo
        if self.ammo <= 0:
            print(f"❌ {self.name} has no ammo!")
            return False
        
        # Check if target is alive
        if not target.is_alive:
            print(f"❌ {target.name} is already eliminated!")
            return False
        
        # Use one ammo
        self.ammo -= 1
        
        # Calculate damage (armor absorbs some)
        absorbed = min(damage, target.armor)
        health_damage = damage - absorbed
        
        # Apply damage
        target.armor -= absorbed
        target.health -= health_damage
        
        print(f"💥 {self.name} shoots {target.name}!")
        print(f"   Damage: {damage} (Armor absorbed: {absorbed}, Health damage: {health_damage})")
        print(f"   {target.name}: Health={target.health}, Armor={target.armor}")
        
        # Check if target is eliminated
        if target.health <= 0:
            target.is_alive = False
            self.kills += 1
            print(f"   💀 {target.name} has been eliminated!")
            print(f"   🏆 {self.name} now has {self.kills} kills!")
            return True
        
        return False
    
    def heal(self, amount=50):
        """Use a medkit to heal."""
        if not self.is_alive:
            print(f"❌ {self.name} is eliminated and cannot heal!")
            return
        
        old_health = self.health
        self.health = min(100, self.health + amount)  # Cap at 100
        healed = self.health - old_health
        print(f"💊 {self.name} used medkit: +{healed} health (Now: {self.health})")
    
    def get_status(self):
        """Display player's current status."""
        status = "ALIVE ✓" if self.is_alive else "ELIMINATED ✗"
        print(f"\n📊 {self.name}'s Status [{status}]")
        print(f"   Health: {self.health}/100")
        print(f"   Armor: {self.armor}")
        print(f"   Position: {self.position}")
        print(f"   Weapon: {self.weapon}")
        print(f"   Ammo: {self.ammo}")
        print(f"   Kills: {self.kills}")


# =============================================================================
# PART 3: CREATING AND USING OBJECTS
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: CREATING PLAYERS (OBJECTS) FROM THE CLASS")
print("=" * 70)

print("\nCreating three players...")

# Create player objects (instances of the Player class)
player1 = Player("ProGamer", (150, 200))
player2 = Player("SnipeMaster", (300, 450))
player3 = Player("SneakyNinja", (500, 100))

print("\n✨ That's it! Three players created with clean code!")
print("   Compare this to the 27 variables we needed before!")

# Display initial status
player1.get_status()
player2.get_status()


# =============================================================================
# PART 4: USING METHODS (ACTIONS)
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 4: PLAYERS IN ACTION")
print("=" * 70)

print("\n--- Game Begins! ---")

# Players loot weapons and armor
print("\n🔍 Looting phase...")
player1.pickup_weapon("M416", 40)
player1.pickup_armor(75)

player2.pickup_weapon("AWM", 15)
player2.pickup_armor(50)

player3.pickup_weapon("UMP45", 25)
# Player 3 finds no armor!

# Players move around
print("\n🏃 Movement phase...")
player1.move(200, 250)
player2.move(320, 470)
player3.move(480, 120)

# Combat begins!
print("\n⚔️ COMBAT PHASE! ⚔️")

# ProGamer shoots SnipeMaster
player1.shoot(player2, damage=25)

# SnipeMaster shoots back
player2.shoot(player1, damage=35)

# SneakyNinja joins the fight
player3.shoot(player2, damage=20)

# SnipeMaster heals
player2.heal(50)

# More combat
print("\n⚔️ Intense battle continues...")
player1.shoot(player2, damage=30)
player1.shoot(player2, damage=30)

# Check final status
print("\n" + "=" * 70)
print("FINAL STATUS")
print("=" * 70)

player1.get_status()
player2.get_status()
player3.get_status()


# =============================================================================
# PART 5: KEY OOP CONCEPTS EXPLAINED
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 5: KEY OOP CONCEPTS")
print("=" * 70)

print("""
1. CLASS (Blueprint):
   - Defines the structure and behavior
   - Written once, used many times
   - Like a template or recipe
   
   Example: class Player

2. OBJECT (Instance):
   - Specific example created from the class
   - Has its own unique data
   - Multiple objects can exist from one class
   
   Example: player1, player2, player3 are all different objects

3. ATTRIBUTES (Properties):
   - Data stored in each object
   - Accessed with: self.attribute_name
   - Each object has its own copy
   
   Example: self.health, self.weapon, self.position

4. METHODS (Actions):
   - Functions that belong to the class
   - Can access and modify object's attributes
   - Called on specific objects
   
   Example: player1.shoot(), player2.move()

5. SELF:
   - Refers to "this specific object"
   - First parameter in all methods
   - How each object keeps track of its own data
   
   Think: self.health = "MY health" (from the player's perspective)

6. __init__ (Constructor):
   - Special method that runs when creating new object
   - Sets up initial values
   - Called automatically
   
   Example: player1 = Player("ProGamer") → runs __init__

7. ENCAPSULATION:
   - Bundling data and methods together
   - Objects manage their own data
   - Clean interface for interaction
   
   Example: All player data is in the Player object
""")


# =============================================================================
# PART 6: WHY OOP IS BETTER
# =============================================================================

print("\n" + "=" * 70)
print("PART 6: COMPARISON - WITH vs WITHOUT OOP")
print("=" * 70)

print("""
WITHOUT OOP (Old way):
❌ Need 900+ variables for 100 players
❌ Functions with many parameters
❌ Hard to track which data belongs to which player
❌ Code duplication everywhere
❌ Difficult to add features
❌ Error-prone and messy

WITH OOP (Modern way):
✓ One Player class, create 100 objects
✓ Clean methods with few parameters
✓ Each object knows its own data
✓ No duplication - write once, use everywhere
✓ Easy to extend (just add methods/attributes)
✓ Professional, maintainable code

CODE COMPARISON:

Without OOP:
    player1_ammo, player2_health, player2_armor = player_shoot_old(
        player1_name, player1_ammo, player2_name, 
        player2_health, player2_armor, 25
    )
    # 😵 7 parameters! Hard to read!

With OOP:
    player1.shoot(player2, 25)
    # 😎 Clean and clear!
""")


# =============================================================================
# PART 7: OOP FOR NEURAL NETWORKS
# =============================================================================

print("\n" + "=" * 70)
print("PART 7: WHY OOP FOR NEURAL NETWORKS")
print("=" * 70)

print("""
Neural networks are complex systems with:
- Many layers
- Many neurons per layer
- Many connections (weights)
- Multiple activation functions

WITHOUT OOP:
❌ Tracking weights for each layer manually
❌ Functions with dozens of parameters
❌ Difficult to add/remove layers
❌ Messy code for complex architectures

WITH OOP:
✓ Each neuron can be an object
✓ Each layer can be an object
✓ Network is an object containing layers
✓ Easy to build complex architectures
✓ Clean, professional code

Example structure:

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def activate(self, inputs):
        # Calculate output
        pass

class Layer:
    def __init__(self, num_neurons):
        self.neurons = [Neuron(...) for _ in range(num_neurons)]
    
    def forward(self, inputs):
        # Process inputs through all neurons
        pass

class NeuralNetwork:
    def __init__(self, architecture):
        self.layers = [Layer(n) for n in architecture]
    
    def predict(self, inputs):
        # Pass through all layers
        pass

# Create a network easily!
network = NeuralNetwork([3, 4, 2])
output = network.predict([1, 0, 1])

Beautiful! 🎨
""")


# =============================================================================
# PART 8: PRACTICE EXERCISES
# =============================================================================

print("\n" + "=" * 70)
print("PART 8: PRACTICE EXERCISES")
print("=" * 70)

print("""
Try these exercises to solidify your OOP understanding:

1. Add a new method to the Player class:
   - drop_weapon(): Player drops their weapon
   - Test it with player1.drop_weapon()

2. Add a "squad" attribute:
   - Each player belongs to a squad (team name)
   - Modify __init__ to accept squad parameter
   - Add get_squad() method

3. Create a Vehicle class:
   - Attributes: type (car/bike), fuel, speed, driver
   - Methods: drive(), refuel(), pickup_player()
   - Create objects: car1, bike1

4. Create a Weapon class:
   - Attributes: name, damage, ammo, fire_rate
   - Methods: fire(), reload()
   - Modify Player to use Weapon objects instead of strings

5. Add a GameMatch class:
   - Attributes: players (list), circle_position, time_remaining
   - Methods: start_match(), end_match(), get_leaderboard()
   - Create a full match with multiple players

These exercises will prepare you for implementing neural networks with OOP!
""")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
🎮 OOP MAKES COMPLEX SYSTEMS MANAGEABLE!

Key Takeaways:
1. Classes are blueprints, objects are instances
2. Attributes store data, methods perform actions
3. 'self' refers to the specific object
4. '__init__' initializes new objects
5. OOP provides clean, organized, maintainable code

Without OOP: Managing 100 PUBG players = 900+ variables 😱
With OOP: Managing 100 PUBG players = 1 class, 100 objects 😎

The same principle applies to neural networks:
- Without OOP: Chaotic mess of variables and functions
- With OOP: Clean, professional, scalable code

You're now ready to build Multi-Layer Perceptrons with OOP!
Proceed to: mlp-implementation.py
""")

print("\n" + "=" * 70)
print("END OF OOP TUTORIAL")
print("=" * 70)
