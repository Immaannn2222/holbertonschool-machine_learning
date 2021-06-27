#!/usr/bin/env python3
"""Q-Learning"""
import gym
import numpy as np


def play(env, Q, max_steps=100):
    """lets's play"""
    env.reset()
    state = 0
    done = False
    for _ in range(max_steps):
        action = np.argmax(Q[state, :])
        new_state, reward, done, info = env.step(action)
        env.render()
        if done:
            env.render()
            break
        state = new_state
    env.close()
    return reward
