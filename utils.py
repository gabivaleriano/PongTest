#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import random
import os

import numpy as np

def set_seed(seed):
    """Set random seed for reproducibility"""
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    print(f"Random seed set to {seed}")

def compute_directional_pellet_salience(pacman_x, pacman_y, traversable_positions, eaten_pellet_positions, n_steps=10, n_actions=5):
    """
    For each action direction, estimate how many uneaten pellets
    are in the next n_steps positions.
    
    eaten_pellet_positions: set of (x,y) where pellets were eaten
    Returns C of shape [n_actions]
    """
    # [noop, up, right, left, down]
    directions = [(0, 0), (0, -1), (1, 0), (-1, 0), (0, 1)]
    
    # Ms. Pac-Man maze approximate bounds
    X_MIN, X_MAX = 13, 170
    Y_MIN, Y_MAX = 2, 158
    STEP_H = 2.5   # horizontal: alternates 2 and 3
    STEP_V = 3.5   # vertical: alternates 3 and 4
    
    C = np.zeros(n_actions)
    
    for i, (dx, dy) in enumerate(directions):
        if dx == 0 and dy == 0:  # noop — always 0
            C[i] = 0.0
            continue

        step_size = STEP_H if dx != 0 else STEP_V
        pellet_score = 0.0

        for step in range(1, n_steps + 1):
            next_x = (pacman_x + dx * step_size * step - X_MIN) % (X_MAX - X_MIN) + X_MIN
            next_y = np.clip(pacman_y + dy * step_size * step, Y_MIN, Y_MAX)
            pos = (int(next_x), int(next_y))
        
            if pos not in traversable_positions:
                break  # wall hit, stop projecting in this direction
        
            discount = 1.0 / step
            
            if pos not in eaten_pellet_positions:
                pellet_score += discount
        
        C[i] = pellet_score

    # Normalize AFTER all actions are computed
    total = np.sum(C)
    if total > 0:
        C = C / total
    else:
        # Fallback: uniform over movement actions only, noop stays 0
        C[1:] = 1.0 / (n_actions - 1)

    return C


# In[ ]:




