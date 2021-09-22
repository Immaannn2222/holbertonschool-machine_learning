#!/usr/bin/env python3
""" Play the Atari game"""
from rl.agents import DQNAgent
from tensorflow.keras.optimizers import Adam
import numpy as np
import gym


construct_model = __import__('train').construct_model
build_agent = __import__('train').build_agent
env = gym.make('Breakout-v0')
height, width, channels = env.observation_space.shape
actions = env.action_space.n
model = construct_model(height, width, channels, actions)
ag = build_agent(model, actions)
ag.compile(Adam(lr=1e-4))
ag.load_weights('policy.h5')
scr = ag.test(env, nb_episodes=10, visualize=True)
