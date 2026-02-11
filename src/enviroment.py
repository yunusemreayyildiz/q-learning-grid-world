
import random
import pygame
import numpy as np
from config import GRID_SIZE, NUM_AGENTS, REWARD_CONFIG
from utils import initialize_pygame, draw_environment

class Environment:
    def __init__(self, num_obstacles):
        self.grid_size = GRID_SIZE
        self.num_agents = NUM_AGENTS
        self.num_obstacles = num_obstacles
        self.goal_pos = None
        self.obstacles = set()
        self.agent_positions = []
        # basic movements of the agents 
        self.action_map = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)} 
        self.reset_layout() 
        self.reset() 

        # start the pygame 
        self.screen = initialize_pygame() 

    def _place_random_entity(self, exclude_list=[]):
        while True:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            pos = (x, y)
            if pos not in exclude_list:
                return pos
    #function that place the obstacles on the enviroment 
    def _place_obstacles(self, count):
        obstacles = set()
        for _ in range(count):
            obs_pos = self._place_random_entity(exclude_list=list(obstacles) + [self.goal_pos])
            self.obstacles.add(obs_pos)
        return obstacles
    
    def is_blocked(self, agent_id, action):
        x, y = self.agent_positions[agent_id]
        dx, dy = self.action_map[action]
        nx, ny = x + dx, y + dy

        if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
            return 1

        if (nx, ny) in self.obstacles:
            return 1

        for i, pos in enumerate(self.agent_positions):
            if i != agent_id and pos == (nx, ny):
                return 1

        return 0
    
    def reset_layout(self):
        self.obstacles = set()
        self.goal_pos = self._place_random_entity()
        current_exclude = {self.goal_pos}
        for _ in range(self.num_obstacles):
            obs_pos = self._place_random_entity(exclude_list=list(current_exclude))
            self.obstacles.add(obs_pos)
            current_exclude.add(obs_pos)

    def reset(self):
        exclude_list = list(self.obstacles) + [self.goal_pos]
        self.agent_positions = [self._place_random_entity(exclude_list=exclude_list) 
                                for _ in range(self.num_agents)]
        return self.agent_positions 

    def get_state(self, agent_id):
        ax, ay = self.agent_positions[agent_id]
        gx, gy = self.goal_pos

        dx = ax - gx
        dy = ay - gy

        return (dx, dy)
    def step(self, agent_id, action):
        current_x, current_y = self.agent_positions[agent_id]
        dx, dy = self.action_map.get(action, (0, 0))
        next_x, next_y = current_x + dx, current_y + dy
        
        reward = REWARD_CONFIG['step_cost'] 
        done = False
        status = 'Moving'

        # 
        if not (0 <= next_x < self.grid_size and 0 <= next_y < self.grid_size):
            reward = REWARD_CONFIG['collision'] 
            return self.get_state(agent_id), reward, False, 'WallCollision'

        next_pos = (next_x, next_y)

        # Obstacles collision reward mechanism
        if next_pos in self.obstacles:
            reward = REWARD_CONFIG['collision'] 
            return self.get_state(agent_id), reward, False, 'ObstacleCollision'

        # Collison with other agents 
        for i, pos in enumerate(self.agent_positions):
            if i != agent_id and pos == next_pos and next_pos != self.goal_pos:
                reward = REWARD_CONFIG['collision']
                return self.get_state(agent_id), reward, False, 'AgentCollision'

        # DISTANCE-BASED SHAPING
        # I implement this feature to make more understandable for the agents if agent move to correct position give this agent little bit a reward
        gx, gy = self.goal_pos
        old_dist = abs(current_x - gx) + abs(current_y - gy)
        new_dist = abs(next_x - gx) + abs(next_y - gy)
        reward += 0.05 * (old_dist - new_dist)

        # 3. reaching to the goal point 
        if next_pos == self.goal_pos:
            reward = REWARD_CONFIG['goal_reach'] 
            done = True
            status = 'ReachedGoal'
        self.agent_positions[agent_id] = next_pos
            
        return self.get_state(agent_id), reward, done, status

    def draw(self):
        draw_environment(self.screen, self)