import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from gym.envs.registration import register

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': False}
)

env = gym.make('FrozenLake-v3')


def makemodel():
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 4),
    )
    return model


DQNmodel = makemodel()

loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 0.0005
optimizer = torch.optim.Adam(DQNmodel.parameters(), lr=learning_rate)

jList = []
rList = []
eList = []
dis = .95
e = 0.1
num_episodes = 5000

wincnt = 0


for i in range(num_episodes):
    s = env.reset()
    rAll = 0
    done = False
    j = 0
    eList.append(e)
    loss = None
    while not done:
        j += 1
        s_hot = torch.from_numpy(np.identity(16)[s:s + 1][0]).float()
        allQ = DQNmodel(s_hot)
        a = int(torch.argmax(allQ))
        if np.random.rand(1) < e:
            a = env.action_space.sample()

        s1, r, done, _ = env.step(a)

        s1_hot = torch.from_numpy(np.identity(16)[s1:s1 + 1][0]).float()
        Q1all = DQNmodel(s1_hot)

        if r == 0.0:
            r = -0.01
        if done and r < 0.0:
            r = -1
        if done and r > 0.0:
            r = 1
            wincnt += 1


        with torch.no_grad():
            maxQ1 = float(torch.max(Q1all))
            targetQ = allQ.clone()
            targetQ[a] = r + dis * maxQ1

        loss = loss_fn(allQ, targetQ)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rAll += r
        s = s1



        if done:
            e = 1. / ((i / 50) + 10)
            break

    if i % 100 == 99:
        print(i, loss.item())

    jList.append(j)
    rList.append(rAll)

print("이동 횟수 평균 : " + str(sum(jList) / num_episodes) + "%")

plt.plot(rList)
plt.show()
plt.plot(jList)
plt.show()

print(wincnt)