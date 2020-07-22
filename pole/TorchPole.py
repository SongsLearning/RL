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


for i in range(1000):
    s = env.reset()
    running_reward = 0
    ep_history = []
    for j in range(max_ep):
        x = torch.tensor(s)
        action = model(x.float()).float()
        s1, r, d, _ = env.step(action)  # Get our reward for taking an action given a bandit.


        if d == True:
            break

        loss = loss_fn(s, )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        s = s1


