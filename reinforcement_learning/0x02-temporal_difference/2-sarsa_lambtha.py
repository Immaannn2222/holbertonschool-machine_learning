#!/usr/bin/env python3
"""0x02-temporal_difference"""
import numpy as np


import numpy as np


def make_greedy_policy(Q, epsilon):
    """Epsilon-greedy algorithm"""
    tactic = np.zeros(shape=Q.shape) + epsilon
    optimum = np.argmax(Q, axis=-1)
    tactic[range(len(tactic)), optimum] = 1 - epsilon
    return tactic


def play_episode(env, tactic, max_steps):
    """Plays a single episode"""
    state = env.reset()
    action = int(np.random.choice(tactic[state]))
    state_action_reward = [(state, action, None)]
    for _ in range(max_steps):
        state, reward, done, _ = env.step(action)
        action = int(np.random.choice(tactic[state]))
        state_action_reward.append((state, action, reward))
        if done:
            break
    return state_action_reward


def sarsa_lambtha(
    env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
    epsilon=1, min_epsilon=0.1, epsilon_decay=0.05
):
    """Performs Backward view SARSA"""
    tactic = make_greedy_policy(Q, epsilon)
    for episode in range(episodes):
        ET = 0
        state_action_reward = play_episode(env, tactic, max_steps)
        state_length = len(state_action_reward) - 1
        for i in range(state_length):
            state, action, _ = state_action_reward[i]
            s, a, r = state_action_reward[i + 1]
            ET *= lambtha * gamma
            ET += 1
            delta = r + gamma * Q[s, a] - \
                Q[state, action]
            Q += alpha * delta * ET
        epsilon = min_epsilon + \
            (epsilon - min_epsilon) * np.exp(-epsilon_decay * episode)
        tactic = make_greedy_policy(Q, epsilon)
    return Q
