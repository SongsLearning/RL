import torch
import numpy as np

dtype = torch.float
device = torch.device("cpu")

bandit_arms = [0.2, 0, -0.2, -2]
num_arms = len(bandit_arms)


def pullBandit(bandit):
    result = np.random.randn(1)
    if result > bandit:
        return 1
    else:
        return -1


for i in range(1000):
    pass