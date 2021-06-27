#!/usr/bin/env python3
"""Q-Learning"""
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """loads the pre-made FrozenLakeEnv evnironment from OpenAI’s gym"""
    if desc is None and map_name is None:
        env = gym.make('FrozenLake8x8-v0')
    else:
        env = gym.make("FrozenLake8x8-v0",
                       is_slippery=is_slippery,
                       map_name=map_name,
                       desc=desc)
    return env
