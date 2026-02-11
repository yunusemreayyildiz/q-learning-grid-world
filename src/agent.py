import random
import numpy as np
from config import LEARNING_RATE, DISCOUNT_FACTOR

class Agent:
    def __init__(self, agent_id, num_actions):
        self.id = agent_id
        self.num_actions = num_actions # (Up, Down, Left, Right)
        # (Q-table)
        self.q_table = {} 

    def get_q_value(self, state, action):# this function using for a get q value for all agents
        # Store the state as a tuple 
        state_key = state
        return self.q_table.get((state_key, action), 0.0) # if cannot find return 0
    def get_state(self, agent_pos, goal_pos):# I used a get state funtion based o the distance to the goal point 
        ax, ay = agent_pos
        gx, gy = goal_pos
        return (ax - gx, ay - gy)#returns that the distance between agent and the goal point 

    def choose_action(self, state, epsilon):
        """Epsilon-greedy"""
        if random.random() < epsilon:
            return random.randrange(self.num_actions)
        else:
            q_values = [self.get_q_value(state, a) for a in range(self.num_actions)]#select the best action for this agent 
            
            # return the max Q value if some of them Q value are same make random decision between them 
            max_q = max(q_values)
            best_actions = [a for a, q in enumerate(q_values) if q == max_q]
            return random.choice(best_actions)
        
    def learn(self, state, action, reward, next_state):
        current_q = self.get_q_value(state, action)
        # find the optimum solution for current states like a state by state detect the optimum solution based on Q values of agents
        q_values_next = [self.get_q_value(next_state, a) for a in range(self.num_actions)]
        max_future_q = max(q_values_next)
        #Equation of the Q learning 
        # Q(s, a) = Q(s, a) + a [r + y * max Q(s', a') - Q(s, a)]
        # decribition of this formula is for new q value use the previous q value as base + Learning rate times 
        new_q = current_q + LEARNING_RATE * (
            reward + DISCOUNT_FACTOR * max_future_q - current_q
        )
        
        #change the Q table for the new state 
        self.q_table[(state, action)] = new_q
        return new_q
        #selector funton for the q table's optimum solution 
    def get_max_q(self, state):
        q_values = [self.get_q_value(state, a) for a in range(self.num_actions)]
        return max(q_values)