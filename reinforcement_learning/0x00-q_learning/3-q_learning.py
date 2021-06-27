#!/usr/bin/env python3
"""Q-Learning"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """function that performs Q-learning"""
    rewards = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total = 0
        step = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, info = env.step(action)
            if done is True and reward == 0:
                reward = -1
            Q[state, action] += alpha * \
                (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
            total += reward
            state = new_state

            if done:
                break
            step += 1

        epsilon = min_epsilon + (epsilon - min_epsilon) * \
            np.exp(-epsilon_decay * ep)
        rewards.append(total)
    return Q, rewards
