import math

import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
# import matlab
# import matlab.engine
import scipy.io as scio
from spinup.my_env.frs_rl_env_segway import segway
import sys
#id = "FRSEnv-v0"

class FRSEnv(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self):
        self.max_speed = 1.0
        self.max_pos = 5.0
        self.min_act = -1.0
        self.max_act = 1.0

        self.low = np.array([-self.max_pos, -self.max_pos, -self.max_speed, -self.max_speed, -np.pi, -self.max_pos, -self.max_pos, -self.max_pos, -self.max_pos])
        self.high = np.array([self.max_pos, self.max_pos, self.max_speed, self.max_speed, np.pi, self.max_pos, self.max_pos, self.max_pos, self.max_pos])

        self.observation_space = spaces.Box(low=self.low,high=self.high,dtype=np.float32)
        self.action_space = spaces.Box(low=self.min_act,high=self.max_act,shape=(3,),dtype=np.float32)

        self.sim_start = False
        # self.eng = matlab.engine.connect_matlab()
        self.it = 0

    def step(self, action):
        action = np.clip(action,self.action_space.low,self.action_space.high)

        a1 = action[0,0]
        a2 = action[0,1]
        a3 = action[0,2]
        a1 = float(a1)
        a2 = float(a2)
        a3 = float(a3)

        a1 = (a1+1)*0.005 #cur
        a2 = (a2+1)*0.005 #xaa
        a3 = (a3+1)*0.01 #yaa
        # print('action:', a1,a2,a3)

        next_o,done,success=self.segway.one_step(cur_factor=a1,xaa_factor=a2,yaa_factor=a3)

        reward = 0
        reward -= 0.1
        if success ==1:
            reward = 6
            self.it = self.it + 1
        if success == 0:
            reward = -9
        #reward -= math.pow(next_o[0],2)+math.pow(next_o[1],2)
        return next_o,reward,done,{}

    def render(self, mode='human'):
        #no need
        pass

    def reset(self):

        self.segway=segway()
        next_o,_,_=self.segway.one_step(cur_factor=0.0005,xaa_factor=0.0005,yaa_factor=0.0)
        return next_o

    def close(self):
        print('env close')
        sys.exit()


    def getresult(self):
        print(self.it)

if __name__ == '__main__':

    env = gym.make("FRSEnv-v0")
    acc= [-1,-1,-1]
    env.reset()
    obs, reward, done, info = env.step(acc)
    it = 0
    while done == 0:
        obs, reward, done, info = env.step(acc)
        print(obs, reward, done, info)
        it = it + 1
        print(it)
    env.close()

