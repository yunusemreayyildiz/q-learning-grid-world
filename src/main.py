# main.py
import pygame
import numpy as np
import sys
from config import (EPISODES, NUM_AGENTS, NUM_OBSTACLES, EPSILON_START, 
                    EPSILON_DECAY, EPSILON_MIN, LOG_FILE_MULTI, FPS)
from enviroment import Environment
from agent import Agent
from utils import plot_rewards_from_log, plot_epsilon_decay, plot_reached_goal_rate

# Function to record training metrics for later analysis and plotting 
def log_training_data(episode, agent_id, total_reward, status, epsilon, log_file):
    with open(log_file, 'a') as f:
        f.write(f"{episode},{agent_id},{total_reward:.2f},{status},{epsilon:.4f}\n")

# Main training loop for the Multi-Agent Reinforcement Learning system 
def train_multi_agent():
    # Initialize the grid environment with a set number of obstacles 
    env = Environment(NUM_OBSTACLES)
    # Create multiple independent agents with 4 possible actions (Up, Down, Left, Right)
    agents = [Agent(i, num_actions=4) for i in range(NUM_AGENTS)]
    
    # Initialize log file with headers 
    with open(LOG_FILE_MULTI, 'w') as f:
        f.write("Episode,Agent,TotalReward,Status,Epsilon\n")

    current_epsilon = EPSILON_START
    
    # Core training loop iterating through each episode 
    for episode in range(1, EPISODES + 1):
        env.reset_layout() # Randomize target and agent starting positions 
        env.reset() 
        # Update epsilon for the exploration-exploitation balance
        current_epsilon = max(EPSILON_MIN, current_epsilon * EPSILON_DECAY) 
        
        episode_rewards = [0] * NUM_AGENTS
        agents_done = [False] * NUM_AGENTS # Tracks completion status for each agent

        # Step-limited simulation loop to prevent infinite loops in sparse reward settings 
        for step in range(2 * env.grid_size * 2): 
            
            # Handle Pygame window events for termination
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit() 
            
            # Update each agent sequentially within the environment 
            for agent_id, agent in enumerate(agents):
                if agents_done[agent_id]:
                    continue # Skip agents that already reached the goal or collided
                
                # Fetch local observation (state) for the specific agent 
                state = env.get_state(agent_id)
                # Choose action using the e-greedy policy [cite: 6]
                action = agent.choose_action(state, current_epsilon)
                
                # Execute action and receive feedback (Next State, Reward, Status)
                next_state, reward, done, status = env.step(agent_id, action)
                # Update the agent's Q-table using the Bellman equation
                agent.learn(state, action, reward, next_state)
                episode_rewards[agent_id] += reward
                
                if done:
                    agents_done[agent_id] = True
            
            # Render the environment and maintain simulation speed
            #env.draw()
            #pygame.time.Clock().tick(FPS)
            
            # Exit step loop early if all agents have completed their task
            if all(agents_done):
                break

        # Log episode results for all 50 agents at the end of simulation
        for agent_id, agent in enumerate(agents):
            status = 'ReachedGoal' if agents_done[agent_id] else 'TimedOut'
            log_training_data(episode, agent_id, episode_rewards[agent_id], status, current_epsilon, LOG_FILE_MULTI)
        
        # Print progress every 10 episodes for monitoring training 
        if episode % 10 == 0:
            avg_reward = sum(episode_rewards) / NUM_AGENTS
            print(f"Episode: {episode}/{EPISODES} | Epsilon: {current_epsilon:.4f} | Avg. Reward: {avg_reward:.2f}")

    pygame.quit()
    print("\nLearning completed. Results preparing...")
    
    # Generate visualization plots as described in the project report 
    plot_rewards_from_log()
    plot_epsilon_decay()
    plot_reached_goal_rate()
    return agents, env


if __name__ == '__main__':
    agents, env = train_multi_agent()