import gym
from gym import error, spaces, utils
from gym.utils import seeding
import scipy.io as sio

class RtdEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    ...
  def step(self, action):
    ...
  def reset(self):
    ...
  def render(self, mode='human'):
    ...
  def close(self):
    ...
  def render(self):
    ...

if __name__ == '__main__':
  data = sio.loadmat('/home/wang/Desktop/RL_HarshTerrain_planning/RTD_cost/aaa.mat')
  print(data['aaa'][2])
  pass
