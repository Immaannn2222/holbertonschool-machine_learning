#!/usr/bin/env python3
"""Policy Gradient module"""
import numpy as np
from policy_gradient import policy
from policy_gradient import policy_gradient


def play_episode(env, weight, i, show_result):
    """Plays a single i"""
    state = env.reset()[None, :]
    state_action_reward_grad = []
    while True:
        if show_result and (i % 1000 == 0):
            env.render()
        action, grad = policy_gradient(state, weight)
        state, reward, done, _ = env.step(action)
        state = state[None, :]
        state_action_reward_grad.append((state, action, reward, grad))
        if done:
            break
    env.close()
    return state_action_reward_grad


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """Trains a Policy Gradient REINFORCE model"""
    weight = np.random.rand(4, 2)
    epis = []
    for i in range(nb_episodes):
        sarg = play_episode(env, weight, i, show_result)
        x = len(sarg) - 1
        score = 0
        for t in range(0, x):
            _, _, reward, grad = sarg[t]
            score += reward
            Y = np.sum([
                gamma**sarg[k][2] *
                sarg[k][2] for k in range(t + 1, x + 1)])
            weight += alpha * Y * grad
        epis.append(score)
        print("{}: {}".format(i, score), end="\r", flush=False)
    return epis
