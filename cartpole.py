import gym
import numpy as np
from gym import wrappers

env = gym.make('CartPole-v1')
best_len = 0
episode_lens = []
best_weights = np.zeros(4)

for i in range(100):
    new_weights = np.random.uniform(-1.0, 1.0, 4)
    length = []

    for j in range(100):
        observation = env.reset()
        done = False
        count = 0

        while not done:
            count += 1
            action = 1 if np.dot(observation, new_weights) > 0 else 0
            observation, reward, done, _ = env.step(action)

            if done:
                break
        length.append(count)
    average_lenth = float(sum(length) / len(length))

    if average_lenth > best_len:
        best_len = average_lenth
        best_weights = new_weights
    episode_lens.append(average_lenth)
    if i % 10 == 0:
        print('best length is ', best_len)

done = False
count = 0
env = wrappers.Monitor(env, 'movie_files', force=True)
observation = env.reset()
while not done:
    count += 1
    action = 1 if np.dot(observation, best_weights) > 0 else 0
    observation, reward, done, _ = env.step(action)

    if done:
        break
print('with best weights game lasted', count, 'moves')
