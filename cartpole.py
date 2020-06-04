import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers

env = gym.make('CartPole-v0')
MAX_STATES = 10 ** 4
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
    bins[2] = np.linspace(-0.418, 0.418, 10)
    bins[3] = np.linspace(5, 5, 10)
    return bins


def assign_bins(observation, bins):
    state = np.zeros(4)
    for i in range(4):
        state[i] = np.digitize(observation[i], bins[i])
    return state


def get_state_as_string(state):
    str_state = ' '.join(str(int(e)) for e in state)
    return str_state


def get_all_states_as_strings():
    states = []
    for i in range(MAX_STATES):
        states.append(str(i).zfill(4))
    return states


def initialize_Q():
    Q = {}
    all_states = get_all_states_as_strings()
    for state in all_states:
        Q[state] = {}
        for action in range(env.action_space.n):
            Q[state][action] = 0

    return Q


def play_one_game(bins, Q, epsilon=0.5):
    observation = env.reset()
    done = False
    count = 0
    env = wrappers.Monitor(env, 'movie_files/Q_learning', force=True)
    state = get_state_as_string(assign_bins(observation, bins))
    total_reward = 0

    while not done:
        count += 1
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = max_dict(Q[state])[0]
        observation, reward, done, _ = env.step(action)
        total_reward += reward
        if done and count < 200:
            reward = -300
        state_new = get_state_as_string(assign_bins(observation, bins))

        a1, max_q_sla1 = max_dict(Q[state_new])
        Q[state][action] = ALPHA * \
            (reward * GAMMA * max_q_sla1 - Q[state][action])

        state, action = state_new, a1
    return total_reward, count


def play_many_games(bins, N=1000):
    Q = initialize_Q()
    length = []
    reward = []
    for n in range(N):
        epsilon = 1.0/np.sqrt(n+1)
        episode_reward, episode_length = play_one_game(bins, Q, epsilon)
        if n % 100 == 0:
            print(n, '%.4f' % eps, episode_reward)
        length.append(episode_length)
        reward.append(episode_reward)
    return length, reward
