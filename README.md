# RL-Drive-MultiAgent: Navigation with Multiple Q-Learning Agents

This project implements a **Multi-Agent Reinforcement Learning (MARL)** system where several autonomous agents learn to navigate a $100 \times 100$ grid environment. The primary goal is to reach a randomly placed target while successfully avoiding obstacles and inter-agent collisions.

##  Project Overview
In this simulation, each of the 50 agents operates as an independent autonomous vehicle.Using the **Q-Learning algorithm**, agents learn optimal navigation strategies through continuous exploration and exploitation of their shared environment.

##  Learning Mechanisms & Objectives
* **Q-Learning Implementation:** Agents maintain independent Q-tables to learn the best actions (Up, Down, Left, Right) for any given state.
* **Reward Structure:** * **Goal Reach:** $+100$ points.
    * **Collision:** $-10$ penalty (obstacle or other agent).
    * **Step Cost:** $-1$ per regular movement to encourage efficiency.
* **Exploration Strategy:** An Epsilon-Greedy approach is used, starting at $\epsilon=1.0$ and decaying by $0.995$ per episode down to a minimum of $0.05$.



## Technical Stack
* **Language:** Python.
* **Simulation Engine:** Pygame (for real-time grid visualization).
* **Data Analysis:** Matplotlib (for training reward plots).
* **Parameters:** $\alpha=0.1$ (Learning Rate), $\gamma=0.95$ (Discount Factor), over 300 training episodes.

## Repository Structure
* `src/`: Contains all source files including `config.py` for parameters and `utils.py` for plotting.
* `logs/`: Contains `training_log_multi.txt` which tracks Episode, Agent, Reward, and Status.
* `Report.pdf`: A comprehensive technical report describing observed behaviors and parameter tuning.

## Simulation Preview
![Agent Navigation](./assets/simulation.gif)
*
*Note: Agents start with random movements and gradually optimize their paths as epsilon decreases and cumulative rewards increase.*

## Performance Analysis
The system evaluates agents based on:
1.  **Correctness:** Proper Q-learning update rule implementation.
2.  **Environment Design:** Effective reward/penalty logic.
3.  **Visualization:** Real-time feedback of the $100 \times 100$ grid environment.
