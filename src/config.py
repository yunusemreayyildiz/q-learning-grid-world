
# ALGORITHM PARAMETERS for Multi-Agent Reinforcement Learning (MARL) 
LEARNING_RATE = 0.1          # alpha (a)
DISCOUNT_FACTOR = 0.95       # gamma (y)
EPISODES = 300               # total epoach count

# Epsilon-Greedy exploration values
EPSILON_START = 1.0          
EPSILON_DECAY = 0.995        
EPSILON_MIN = 0.05

# ENVIRONMENT PARAMETERS
GRID_SIZE = 100              # 100x100 Grid size 
NUM_AGENTS = 50              # number of agents 
NUM_OBSTACLES = 50      # number of obstacles 

# REWARD STRUCTURE
REWARD_CONFIG = {
    'goal_reach': 300,      # reaching to the goal point (100)
    'collision': -10,       # Collision with an obstacle or another agent: (10)(given in assigment)
    'step_cost': -0.15         # Normal movement cost (-1)
}

# VISUALIZATION/LOGGING
SCREEN_WIDTH = 600           # Pygame screen width
SCREEN_HEIGHT = 600          # Pygame screen height
TILE_SIZE = SCREEN_WIDTH // GRID_SIZE  # 600 / 100 = 6 to make each tile 6x6 pixels 
FPS = 40          # velocity of the game loop

LOG_FILE_MULTI = 'logs/training_log_multi.txt' # Log file name [cite: 42, 55]