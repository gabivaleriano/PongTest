#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import os
import pickle

from tqdm import tqdm
from dqn import DQN
from replay_buffer import ReplayBuffer
from utils import set_seed

import ale_py
import gymnasium as gym
from gymnasium.wrappers import TransformReward


# In[2]:


def train_with_seed(seed=42, 
                    steps=1_000_000):
      
    set_seed(seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = gym.make("ALE/Pong-v5",
               obs_type="rgb",
                  frameskip=4)
    
    env = gym.wrappers.ResizeObservation(env, (84, 84)) # Resize each frame to 84x84 pixels 
    env = gym.wrappers.GrayscaleObservation(env) # Convert frames from RGB to grayscale
    env = gym.wrappers.FrameStackObservation(env, 4) 
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    # Create networ
    net = DQN(env.action_space.n).to(device)
    target = DQN(env.action_space.n).to(device)
    target.load_state_dict(net.state_dict())
    
    # Optimizer and buffer
    opt = optim.RMSprop(net.parameters(), lr=0.00025, alpha=0.95, eps=0.01, momentum=0.0)
    buf = ReplayBuffer()
    eps = lambda t: max(0.1, 1.0 - 0.9 * (t / 1_000_000))
    
    state,_ = env.reset(seed=seed)
    state = np.asarray(state)
    info = {}
    
    episode_count = 0
    
    bar = tqdm(total=steps, desc=f"Seed {seed}")

    all_metrics = []
    
    for t in range(1, steps + 1):
        # Epsilon-greedy action selection
        if random.random() < eps(t):
            a = env.action_space.sample()

        else:
            with torch.no_grad():
                q = net(torch.tensor(state, device=device).unsqueeze(0))
                q_values = q.squeeze(0).cpu().numpy()    
            
        # Environment step
        ns, r, term, trunc, info = env.step(a)
        ram = env.unwrapped.ale.getRAM()
        ns = np.asarray(ns)
        done = term or trunc
        
        # Store experience
        buf.push(state, a, r, ns, done)
        state = ns
        bar.update(1)
        
        # Episode ended
        if done:
            metrics = {}
            episode_count += 1
            state, _ = env.reset(seed=seed)
            state = np.asarray(state)

            #if 'metrics' in info:
                #metrics = info['metrics']
            metrics['episode'] = episode_count
            metrics['total reward'] = info["episode"].get("r", 0)

            all_metrics.append(metrics)
                
        # Training step
        if len(buf) >= 10000 and t % 4 == 0:
            s, a_batch, r_batch, ns, d = buf.sample(32)
            s = torch.tensor(s, device=device)
            ns = torch.tensor(ns, device=device)
            a_batch = torch.tensor(a_batch, device=device).unsqueeze(1)
            r_batch = torch.tensor(r_batch, device=device).unsqueeze(1)
            d = torch.tensor(d, device=device).unsqueeze(1)
            
            # Compute Q-values
            q = net(s).gather(1, a_batch)
            
            # Compute target
            with torch.no_grad():
                nq = target(ns).max(1)[0].unsqueeze(1)
                tgt = r_batch + 0.99 * nq * (1 - d)
            
            # Update network
            #loss = F.mse_loss(q, tgt)
            loss = F.smooth_l1_loss(q, tgt)  # Huber loss, beta=1.0 by default -> clips TD error to [-1,1]
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        # Update target network
        if t % 10000 == 0:
            target.load_state_dict(net.state_dict())
        
        # Update progress bar
        if t % 50000 == 0:
            bar.set_postfix({"eps": f"{eps(t):.2f}"})
    
    bar.close()    
    
    return net, all_metrics


# In[3]:


def complete_training(num_seeds=5, 
                   steps=1_000_000,
                   save_dir='results',
                   eval_episodes = 100):

    os.makedirs(save_dir, exist_ok=True)
    
    all_results = {}
        
    agent_results = {
        'training': [],
        'evaluation': []
    }
    
    # Train with multiple seeds
    seeds = [1, 42, 123, 456, 789][:num_seeds]

    for seed in seeds: 
        net, metrics = train_with_seed(
            seed=seed,
            steps=steps)
             
        agent_results['training'].append({
        'seed': seed,
        'metrics': metrics})

        # Evaluate
        eval_metrics = evaluate_agent(
            net=net,
            num_episodes=eval_episodes,
            base_seed=seed * 1000)
        
        agent_results['evaluation'].append({
            'train_seed': seed,
            'eval_metrics': eval_metrics
        })
                
    all_results = agent_results

    final_path = os.path.join(save_dir, 'results.pkl')
    with open(final_path, 'wb') as f:
        pickle.dump(all_results, f)


# In[4]:


def evaluate_agent(net, 
                   num_episodes=100, 
                   base_seed=42, 
                   deterministic=True):

        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\n{'='*60}")
    print(f"Evaluating agent for {num_episodes} episodes")
    print(f"{'='*60}\n")
    
        # Create environment
    env = gym.make("ALE/Pong-v5",
               obs_type="rgb",
                  frameskip=4)
    
    env = gym.wrappers.ResizeObservation(env, (84, 84)) # Resize each frame to 84x84 pixels 
    env = gym.wrappers.GrayscaleObservation(env) # Convert frames from RGB to grayscale
    env = gym.wrappers.FrameStackObservation(env, 4) 
    env = gym.wrappers.RecordEpisodeStatistics(env)
    net.eval()
    
    eval_metrics = []
    
    for episode in tqdm(range(num_episodes), desc="Evaluation"):
        # Use different seed for each episode
        episode_seed = base_seed + episode
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        random.seed(episode_seed)
        
        state, _ = env.reset(seed=episode_seed)
        info = {}
        done = False
        
        while not done:
            with torch.no_grad():
                q = net(torch.tensor(state.__array__(), device=device).unsqueeze(0))
                q_values = q.squeeze(0).cpu().numpy()

                if deterministic:
                    a = int(np.argmax(q_values))
                else:
                    # Small epsilon for variation
                    if random.random() < 0.05:
                        a = env.action_space.sample()
                    else:
                        a = int(np.argmax(q_values))
            
            state, r, term, trunc, info = env.step(a)
            done = term or trunc
            
            if done and 'metrics' in info:
                metrics = info['metrics']
                metrics['total reward'] = info["episode"].get("r", 0)
                metrics['eval_episode'] = episode
                metrics['eval_seed'] = episode_seed
                    
                eval_metrics.append(metrics)
    
    net.train()
    return eval_metrics

