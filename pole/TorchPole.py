import torch
import numpy as np
import gym

dtype = torch.float
device = torch.device("cpu")

env = gym.make('CartPole-v0')

max_ep = 200

model = torch.nn.Sequential(
    torch.nn.Linear(4, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 2),
)
loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

dis = 0.9

for i in range(100):
    s = env.reset()
    running_reward = 0
    ep_history = []
    e = 1. / ((i / 100) + 1)

    for j in range(max_ep):

        x = torch.tensor(s)
        y = model(x.float()).float()

        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = int(torch.argmax(y))

        s1, reward, done, _ = env.step(action)  # Get our reward for taking an action given a bandit.

        if done:
            y[action] = -100
        else:
            x1 = torch.tensor(s1)
            y1 = model(x1.float()).float()
            y[action] = reward + dis * int(torch.argmax(y1))

        loss = loss_fn(model(x.float()).float(), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if done:
            break
        s = s1


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