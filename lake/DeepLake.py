import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from gym.envs.registration import register
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """transition 저장"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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
    model = model.cuda()
    return model


memory = ReplayMemory(10000)

DQNmodel = makemodel()

loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 0.0005
optimizer = torch.optim.Adam(DQNmodel.parameters(), lr=learning_rate)

jList = []
rList = []
eList = []
dis = .95
e = 0.1
num_episodes = 10000

wincnt = 0

BATCH_SIZE = 100


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    next_state_batch = torch.cat(batch.next_state)

    action_batch = list(batch.action)
    reward_batch = list(batch.reward)
    reward_batch = torch.from_numpy(np.array(reward_batch)).float().cuda()

    state_batch = state_batch.view(BATCH_SIZE, 16)
    next_state_batch = next_state_batch.view(BATCH_SIZE, 16)
    Q = DQNmodel(state_batch.cuda())
    Q1all = DQNmodel(next_state_batch.cuda())

    with torch.no_grad():
        maxQ1 = float(torch.max(Q1all))
        targetQ = Q.clone()

        targetQ = targetQ.transpose(0, 1)
        targetQ[action_batch] = reward_batch + dis * maxQ1
        targetQ = targetQ.transpose(0, 1)

    loss = loss_fn(Q, targetQ)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(i, loss.item())


for i in range(num_episodes):
    s = env.reset()
    rAll = 0
    done = False
    j = 0
    eList.append(e)
    while not done:
        j += 1
        s_hot = torch.from_numpy(np.identity(16)[s:s + 1][0]).float()
        allQ = DQNmodel(s_hot.cuda())
        a = int(torch.argmax(allQ))
        if np.random.rand(1) < e:
            a = env.action_space.sample()

        s1, r, done, _ = env.step(a)

        s1_hot = torch.from_numpy(np.identity(16)[s1:s1 + 1][0]).float()

        if r == 0.0:
            r = -0.01
        if done and r < 0.0:
            r = -1
        if done and r > 0.0:
            r = 5
            wincnt += 1

        memory.push(s_hot, a, s1_hot, r)

        rAll += r
        s = s1

        if done:
            e = 1. / ((i / 50) + 10)
            break

        if j > 100:
            break
    if i % 50 == 0:
        optimize_model()

    jList.append(j)
    rList.append(rAll)

print("이동 횟수 평균 : " + str(sum(jList) / num_episodes) + "%")

plt.plot(rList, 'r')
plt.show()
plt.plot(jList)
plt.show()

print(wincnt)
