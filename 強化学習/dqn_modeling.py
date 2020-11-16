import copy
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu as relu
import torch.optim as optim
from torch.autograd import Variable
import gym
from gym import wrappers
import cloudpickle

class Q(nn.Module):
  def __init__(self):
    super(Q, self).__init__()
    self.fc1 = nn.Linear(num_obs, hidden)
    self.fc2 = nn.Linear(hidden, hidden)
    self.fc3 = nn.Linear(hidden, hidden)
    self.fc4 = nn.Linear(hidden, num_acts)
    
  def forward(self, x):
    x = relu(self.fc1(x))
    x = relu(self.fc2(x))
    x = relu(self.fc3(x))
    x = relu(self.fc4(x))
    return x

if __name__ == '__main__':

    env = gym.make("CartPole-v0")

    num_epoch = 3000
    max_step = 200
    memory_size = 200
    batch_size = 50
    epsilon = 1.0
    epsilon_decrease = 0.001
    epsilon_min = 0.1
    epsilon_reduce = 200
    train_frequency = 10
    update_frequency = 20
    gamma = 0.99
    log_frequency = 1000

    total_step = 0
    memory = []
    total_rewards = []
    losses = []

    num_obs = env.observation_space.shape[0]
    num_acts = env.action_space.n
    hidden = 100


    model = Q()
    Q_ast = copy.deepcopy(model)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for epoch in range(num_epoch):
        next_obs = env.reset()
        step = 0
        done = False
        total_reward = 0

        while not done and step < max_step:
            next_act = env.action_space.sample()

            # ε-greedy
            if np.random.rand() > epsilon:
                next_obs_ = np.array(next_obs, dtype="float32").reshape((1, num_obs))
                next_obs_ = torch.from_numpy(next_obs_)
                next_act = model(next_obs_)
                _, datas = torch.max(next_act.data, 1)
                next_act = datas.numpy()[0]

            obs, reward, done, _ = env.step(next_act)

            if done:
                reward = -1

            memory.append((next_obs, next_act, reward, obs, done))
            if len(memory) > memory_size:
                memory.pop(0)

            if len(memory) == memory_size:
                if total_step % train_frequency == 0:
                    memory_ = np.random.permutation(memory)
                    memory_range = range(len(memory_))
                    for i in memory_range[::batch_size]:
                        batch = np.array(memory_[i:i+batch_size])
                        next_obss = np.array(batch[:,0].tolist(), dtype="float32").reshape((batch_size, num_obs))
                        next_acts = np.array(batch[:, 1].tolist(), dtype="int32")
                        rewards = np.array(batch[:, 2].tolist(), dtype="int32")
                        obss = np.array(batch[:,3].tolist(), dtype="float32").reshape((batch_size, num_obs))
                        dones = np.array(batch[:,4].tolist(), dtype="bool")

                        next_obss_ = torch.from_numpy(next_obss)
                        
                        model.eval()
                        q = model(next_obss_)

                        obss_ = torch.from_numpy(obss)
                        maxs, _ = torch.max(Q_ast(obss_).data, 1)
                        maxq = maxs.numpy()

                        target = copy.deepcopy(q.data.numpy())
                        for j in range(batch_size):
                            target[j, next_acts[j]] = rewards[j]+gamma*maxq[j]*(not dones[j])


                        model.train()
                        loss = nn.MSELoss()(q, torch.from_numpy(target))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    if total_step % update_frequency == 0:
                        Q_ast = copy.deepcopy(model)

                if epsilon > epsilon_min and total_step > epsilon_reduce:
                    epsilon -= epsilon_decrease

                total_reward += reward
                step += 1
                total_step += 1
                next_obs = obs

            total_rewards.append(total_reward)

        if (epoch+1) % log_frequency == 0:
            print(epoch+1, total_reward)
    
    with open('model2.pkl', 'wb') as f:
        cloudpickle.dump(model, f)
    print("finish!")