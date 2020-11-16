import cloudpickle
import torch
import numpy as np
import gym
from gym import wrappers

if __name__ == '__main__':
    with open('model2.pkl', 'rb') as f:
        model = cloudpickle.load(f)
    

    
    env = gym.make("CartPole-v0")
    next_obs = env.reset()
    step = 0
    done = False
    total_reward = 0
    total_rewards = []
    max_step = 400
    num_obs = env.observation_space.shape[0]
    num_acts = env.action_space.n
    total_step = 0
    
    
    while not done and step < max_step:
        env.render()
        next_act = env.action_space.sample()

        # greedy
        next_obs_ = np.array(next_obs, dtype="float32").reshape((1, num_obs))
        next_obs_ = torch.from_numpy(next_obs_)
        next_act = model(next_obs_)
        _, datas = torch.max(next_act.data, 1)
        next_act = datas.numpy()[0]

        obs, reward, done, _ = env.step(next_act)
        if done:
            reward = -1

        total_reward += reward
        step += 1
        total_step += 1
        next_obs = obs
    
        total_rewards.append(total_reward)
