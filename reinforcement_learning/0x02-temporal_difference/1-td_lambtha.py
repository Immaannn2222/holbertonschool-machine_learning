#!/usr/bin/env python3
"""0x02-temporal_difference"""
import numpy as np


def td_lambtha(
    env, V, policy, lambtha, episodes=5000, max_steps=100,
    alpha=0.1, gamma=0.99
):
    """Performs TD-lambtha backward prediction"""
    episode = [[], []]
    x = [0 for i in range(env.observation_space.n)]
    for i in range(episodes):
        state = env.reset()
        for j in range(max_steps):
            x = list(np.array(x) * lambtha * gamma)
            x[state] += 1
            action = policy(state)
            new_state, reward, done, info = env.step(action)
            delta_t = reward + gamma * V[new_state] - V[state]
            V[state] = V[state] + alpha * delta_t * x[state]
            if done:
                break
            state = new_state
    y = np.array(V)
    return y
