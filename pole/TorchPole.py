from collections import deque
import random
import torch
import numpy as np
import gym
import matplotlib.pyplot as plt

dtype = torch.float
device = torch.device("cuda:0")

env = gym.make('CartPole-v0')

max_ep = 200

model = torch.nn.Sequential(
    torch.nn.Linear(4, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 30),
    torch.nn.ReLU(),
    torch.nn.Linear(30, 2),
)
loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

dis = 0.9
REPLAY_MEMORY = 500
rewards = []
replay_buffer = deque()


def simple_replay_train(train_batch):
    x_stack = torch.empty(0).reshape(0, 4)
    y_stack = torch.empty(0).reshape(0, 2)

    for state, action, reward, next_state, done in train_batch:
        x = torch.tensor(state)
        y = model(x.float()).float()
        x = x.float()
        if done:
            y[action] = -100
        else:
            x1 = torch.tensor(s1)
            y1 = model(x1.float()).float()
            y[action] = reward + dis * int(torch.argmax(y1))

        y = y.unsqueeze(0)
        x = x.unsqueeze(0)

        y_stack = torch.cat([y_stack, y])
        x_stack = torch.cat([x_stack, x])

    y_pred = model(x_stack)
    loss = loss_fn(y_pred, y_stack)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


for i in range(1000):
    s = env.reset()
    running_reward = 0
    e = 1. / ((i / 100) + 1)

    for j in range(max_ep):

        x = torch.tensor(s)
        y = model(x.float()).float()

        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = int(torch.argmax(y))

        s1, reward, done, _ = env.step(action)  # Get our reward for taking an action given a bandit.
        replay_buffer.append((s, action, reward, s1, done))

        if len(replay_buffer) > REPLAY_MEMORY:
            replay_buffer.popleft()

        running_reward += reward

        if done:
            rewards.append(running_reward)
            reward = -100
            break
        s = s1

    if i % 10 == 1:
        for _ in range(50):
            minibatch = random.sample(replay_buffer, 10)
            simple_replay_train(minibatch)

plt.bar(range(len(rewards)), rewards, color="blue")
plt.show()

while True:
    s = env.reset()
    for i in range(max_ep):
        env.render()
        x = torch.tensor(s)
        y = model(x.float()).float()
        action = int(torch.argmax(y))
        s1, reward, done, _ = env.step(action)

        if done:
            break
