import numys as np
import matplotlib.pyplot as plt
import gym


def max_action(Q, state):
    values = np.array([Q[state, a]for a in range(2)])
    action = np.argmax(values)
    return action
