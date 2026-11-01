#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import torch
import torch.nn as nn
#import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()        
        self.network = nn.Sequential( # Standard DQN convolutional network (Mnih et al. 2015)
            
            # Input: 4 channels (4 stacked grayscale frames), output: 32 feature maps
            # 8x8 kernel, stride 4 -> aggressively reduces spatial resolution right away
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),

            # Second conv layer: 32 -> 64 channels, 4x4 kernel, stride 2
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),

            # Third conv layer: 64 -> 64 channels, 3x3 kernel, stride 1 (keeps spatial resolution)
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),

            # Flattens the 3D tensor (channels x height x width) into a 1D vector
            nn.Flatten(),

            # Fully connected layer: 3136 -> 512 (3136 comes from the flattened size after convolutions)
            nn.Linear(3136, 512), nn.ReLU(),

            # Output layer: 512 -> n_actions (one estimated Q-value per possible action)
            nn.Linear(512, n_actions),
        )
        
    def forward(self, x):
        # Normalize pixel values from [0,255] to [0,1] before feeding into the network
        return self.network(x / 255.)


# In[ ]:




