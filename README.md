# RL-Drive-MultiAgent: Navigation with Multiple Q-Learning Agents

This project implements a **Multi-Agent Reinforcement Learning (MARL)** system where several autonomous agents learn to navigate a $100 \times 100$ grid environment. [cite_start]The primary goal is to reach a randomly placed target while successfully avoiding obstacles and inter-agent collisions[cite: 7, 18, 20].

##  Project Overview
[cite_start]In this simulation, each of the 50 agents operates as an independent autonomous vehicle[cite: 7, 19]. [cite_start]Using the **Q-Learning algorithm**, agents learn optimal navigation strategies through continuous exploration and exploitation of their shared environment[cite: 8, 9].

##  Learning Mechanisms & Objectives
* [cite_start]**Q-Learning Implementation:** Agents maintain independent Q-tables to learn the best actions (Up, Down, Left, Right) for any given state[cite: 9, 28, 33].
* [cite_start]**Reward Structure:** * **Goal Reach:** $+100$ points[cite: 24].
    * [cite_start]**Collision:** $-10$ penalty (obstacle or other agent)[cite: 25].
    * [cite_start]**Step Cost:** $-1$ per regular movement to encourage efficiency[cite: 26].
* [cite_start]**Exploration Strategy:** An Epsilon-Greedy approach is used, starting at $\epsilon=1.0$ and decaying by $0.995$ per episode down to a minimum of $0.05$[cite: 39].



## Technical Stack
* [cite_start]**Language:** Python[cite: 54].
* [cite_start]**Simulation Engine:** Pygame (for real-time grid visualization)[cite: 15, 46].
* [cite_start]**Data Analysis:** Matplotlib (for training reward plots)[cite: 45].
* [cite_start]**Parameters:** $\alpha=0.1$ (Learning Rate), $\gamma=0.95$ (Discount Factor), over 300 training episodes[cite: 37, 38, 40].

## Repository Structure
* [cite_start]`src/`: Contains all source files including `config.py` for parameters and `utils.py` for plotting[cite: 36, 45, 54].
* [cite_start]`logs/`: Contains `training_log_multi.txt` which tracks Episode, Agent, Reward, and Status[cite: 42, 44].
* [cite_start]`Report.pdf`: A comprehensive technical report describing observed behaviors and parameter tuning[cite: 56, 57, 58].

## Simulation Preview
![Agent Navigation](./assets/simulation.gif)
[cite_start]*Note: Agents start with random movements and gradually optimize their paths as epsilon decreases and cumulative rewards increase[cite: 61, 62, 63].*

## Performance Analysis
The system evaluates agents based on:
1.  [cite_start]**Correctness:** Proper Q-learning update rule implementation[cite: 49].
2.  [cite_start]**Environment Design:** Effective reward/penalty logic[cite: 49].
3.  [cite_start]**Visualization:** Real-time feedback of the $100 \times 100$ grid environment[cite: 18, 49].