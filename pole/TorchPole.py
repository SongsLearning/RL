from collections import deque
import random
import torch
import numpy as np
import gym
import matplotlib.pyplot as plt

dtype = torch.float
device = torch.device("cuda:0")

env = gym.make('CartPole-v1')
env._max_episode_steps = 10001

max_ep = 3000


def makemodel():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 30),
        torch.nn.ReLU(),
        torch.nn.Linear(30, 2),
    )
    return model


mainDQN = makemodel()
targetDQN = makemodel()
targetDQN.load_state_dict(mainDQN.state_dict())


loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-5
optimizer = torch.optim.Adam(mainDQN.parameters(), lr=learning_rate)

dis = 0.9
REPLAY_MEMORY = 50000
rewards = []
replay_buffer = deque()


def simple_replay_train(train_batch):
    x_stack = torch.empty(0).reshape(0, 4)
    y_stack = torch.empty(0).reshape(0, 2)

    for state, a, r, next_state, d in train_batch:
        x_ = torch.tensor(state)
        y_ = mainDQN(x_.float()).float()
        x_ = x_.float()

        next_state = torch.tensor(next_state)
        if d:
            y[a] = r
        else:
            y[a] = r + dis * torch.max(targetDQN(next_state.float()).float())

        y_ = y_.unsqueeze(0)
        x_ = x_.unsqueeze(0)

        y_stack = torch.cat([y_stack, y_])
        x_stack = torch.cat([x_stack, x_])

    y_pred = mainDQN(x_stack)
    loss = loss_fn(y_stack, y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


for i in range(3000):
    s = env.reset()
    running_reward = 0
    e = 1. / ((i / 50) + 1)
    done = False
    while not done:

        x = torch.tensor(s)
        y = mainDQN(x.float()).float()

        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = int(torch.argmax(y))

        s1, reward, done, _ = env.step(action)

        if done:
            rewards.append(running_reward)
            reward = -100

        replay_buffer.append((s, action, reward, s1, done))

        if len(replay_buffer) > REPLAY_MEMORY:
            replay_buffer.popleft()

        running_reward += reward

        s = s1

    if i % 10 == 1 and i > 100:
        for _ in range(3):
            minibatch = random.sample(replay_buffer, 10)
            simple_replay_train(minibatch)
        targetDQN.load_state_dict(mainDQN.state_dict())


plt.bar(range(len(rewards)), rewards, color="blue")
plt.show()

while True:
    s = env.reset()
    for i in range(max_ep):
        env.render()
        x = torch.tensor(s)
        y = mainDQN(x.float()).float()
        action = int(torch.argmax(y))
        s1, reward, done, _ = env.step(action)

        if done:
            break
