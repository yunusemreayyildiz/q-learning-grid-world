import pygame
import matplotlib.pyplot as plt
import pandas as pd
from config import TILE_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT, LOG_FILE_MULTI
import numpy as np

# Color Definitions for Pygame Visualization
BG = (0, 0, 0)
Grey = (70, 70, 70)
Red = (255, 0, 0)    # Representing Obstacles 
Green = (0, 255, 0)  # Representing the Goal Point 
Blue = (0, 0, 255)   # Representing the Autonomous Agents

# --- 1. Pygame Visualization Module ---

def initialize_pygame(caption='Multi-Agent Reinforcement Learning'):
    """Initializes the Pygame window and sets the title."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(caption)
    return screen

def draw_environment(screen, env):
    """Renders the grid, goal, obstacles, and all agents on the screen."""
    
    # Fill background and draw grid lines
    screen.fill(BG)
    for x in range(0, SCREEN_WIDTH, TILE_SIZE):
        pygame.draw.line(screen, Grey, (x, 0), (x, SCREEN_HEIGHT))
    for y in range(0, SCREEN_HEIGHT, TILE_SIZE):
        pygame.draw.line(screen, Grey, (0, y), (SCREEN_WIDTH, y))

    # Draw the target destination (Green Square)
    gx, gy = env.goal_pos
    pygame.draw.rect(screen, Green, (gx * TILE_SIZE, gy * TILE_SIZE, TILE_SIZE, TILE_SIZE))
    
    # Draw static obstacles (Red Squares)
    for ox, oy in env.obstacles:
        pygame.draw.rect(screen, Red, (ox * TILE_SIZE, oy * TILE_SIZE, TILE_SIZE, TILE_SIZE))

    # Draw active agents (Blue Circles)
    for pos in env.agent_positions:
        px, py = pos
        pygame.draw.circle(screen, Blue, 
                           (px * TILE_SIZE + TILE_SIZE // 2, py * TILE_SIZE + TILE_SIZE // 2), 
                           TILE_SIZE // 2)

    pygame.display.update()

# --- 2. Analysis and Plotting Functions ---

def plot_rewards_from_log(log_file=LOG_FILE_MULTI):
    """Generates the cumulative reward plot from the training log file."""
    print(f"Generating reward plots from log file: {log_file}")
    
    try:
        # Load training data into a DataFrame
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"Error: {log_file} not found. Please complete the training.")
        return

    # Aggregate total rewards to find the average per episode across all agents
    avg_reward_per_episode = df.groupby('Episode')['TotalReward'].mean()
    
    # Apply a rolling window to smooth out volatility in the learning curve
    window_size = 10
    smoothed_avg_reward = avg_reward_per_episode.rolling(window=window_size, min_periods=1).mean()

    # Visualization of the reward curve
    plt.figure(figsize=(12, 6))
    plt.plot(avg_reward_per_episode.index, avg_reward_per_episode.values, color='skyblue', alpha=0.4, label='Episode Average')
    plt.plot(smoothed_avg_reward.index, smoothed_avg_reward.values, color='blue', linewidth=2, label=f'{window_size} Episode Rolling Average')

    plt.title('Mean Cumulative Reward Per Episode (Multi-Agent Q-Learning)') # Fixed "Peward" typo
    plt.xlabel('Episode')
    plt.ylabel('Mean Total Reward')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

def plot_epsilon_decay(log_file=LOG_FILE_MULTI):
    """Visualizes how the exploration rate (epsilon) decreases over time."""
    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"Error: {log_file} not found.")
        return
        
    # Extract unique epsilon values used in each episode
    epsilon_decay = df.drop_duplicates(subset=['Episode']).set_index('Episode')['Epsilon']
    
    plt.figure(figsize=(12, 4))
    plt.plot(epsilon_decay.index, epsilon_decay.values, color='orange')
    plt.title('Epsilon (Exploration Rate) Decay Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon Value')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_reached_goal_rate(log_file=LOG_FILE_MULTI):
    """Calculates and plots the percentage of agents that successfully reached the goal."""
    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"Error: {log_file} not found.")
        return

    # Count successful goal reaches per episode
    reached_counts = (
        df[df['Status'] == 'ReachedGoal']
        .groupby('Episode')
        .size()
    )

    # Count total agents recorded per episode
    total_agents = (
        df.groupby('Episode')
        .size()
    )

    # Calculate success rate (0.0 to 1.0)
    reached_goal_rate = reached_counts / total_agents
    reached_goal_rate = reached_goal_rate.fillna(0) # Handle episodes with 0% success

    # Calculate moving average for success rate trend
    window = 10
    smoothed_rate = reached_goal_rate.rolling(window, min_periods=1).mean()

    # Plot the success rate evolution
    plt.figure(figsize=(12, 5))
    plt.plot(reached_goal_rate.index, reached_goal_rate.values,
             alpha=0.3, label='Raw Episode Success Rate')
    plt.plot(smoothed_rate.index, smoothed_rate.values,
             linewidth=2, label=f'{window}-Episode Rolling Average')

    plt.title('Agent Success Rate (ReachedGoal % Based on Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (0 to 1)')
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()