#!/usr/bin/env python3
"""0x02-temporal_difference"""
import numpy as np
import gym


def play_episode(env, policy, max_steps):
    """A single play repisode"""
    state = env.reset()
    action = policy(state)
    state_action_reward = [(state, action, None)]
    for _ in range(max_steps):
        state, reward, done, _ = env.step(action)
        action = policy(state)
        state_action_reward.append((state, action, reward))
    return state_action_reward


def monte_carlo(
     env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """performs the Monte Carlo algorithm"""
    seen_state_action = set()
    for _ in range(episodes):
        state_action_reward = play_episode(env, policy, max_steps)
        T = len(state_action_reward) - 1
        G = 0
        for t in range(T - 1, -1, -1):
            state, action, _ = state_action_reward[t]
            _, _, reward_t_1 = state_action_reward[t + 1]
            G = gamma * G + reward_t_1
            if not (state, action) in seen_state_action:
                V[state] = V[state] + alpha * (G - V[state])
            seen_state_action.add((state, action))
    return V
