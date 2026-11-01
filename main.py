#!/usr/bin/env python
# coding: utf-8

# In[1]:


from training import complete_training
from utils import set_seed

import argparse

def main():
    """Main function to run experiments"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train DQN on Pong')
    
    # Experiment settings
    parser.add_argument('--num_seeds', type=int, default=5,
                       help='Number of seeds for training')
    parser.add_argument('--steps', type=int, default=1_000_000,
                       help='Total training steps per seed')
    parser.add_argument('--eval_episodes', type=int, default=100,
                       help='Number episodes for evaluation')

    # Output settings
    parser.add_argument('--save-dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()

    # Print configuration
    print("="*60)
    print("PONG DQN EXPERIMENT")
    print("="*60)
    print(f"Number of seeds: {args.num_seeds}")
    print(f"Training steps: {args.steps:,}")
    print(f"Save directory: {args.save_dir}")

    print("="*60)
    print()

    complete_training(
    num_seeds=args.num_seeds,
    steps=args.steps,
    eval_episodes=args.eval_episodes)

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print(f"Results saved to: {args.save_dir}")
    print("="*60)

if __name__ == "__main__":
    main()


# In[ ]:




