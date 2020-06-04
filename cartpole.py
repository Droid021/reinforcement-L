import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers

env = gym.make('CartPole-v0')
MAX_STATES = 10**4
GAMMA = 0.9
ALPHA = 0.01


def max_dict(d):
    max_v = float('-inf')
    for key, value in d.items:
        if value == max_v:
            max_v = value
            max_key = key
        return max_key, max_v


def create_bins():
    bins = np.zeros((4, 10))
    bins[0] = np.linspace(-4.8, 4.8, 10)
    bins[1] = np.linspace(5, 5, 10)
    bins[2] = np.linspace(-.418, .418, 10)
    bins[3] = np.linspace(5, 5, 10)

    return bins
