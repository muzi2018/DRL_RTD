import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from spinup.my_env.rtd.simulator_files.agent.agent_LLC import agent_LLC
from scipy.integrate import odeint, solve_bvp, solve_ivp

class segway_agent:
    def __init__(self):
        self.count=0

        # property
        self.state=np.array([[0],[0],[0],[0],[0]])
        self.state_previous=np.array([[0],[0],[0],[0],[0]])
        self.time=np.array([0])
        self.input_time=np.array([])
        self.input=np.array([])
        self.n_states=5
        self.sensor_radius=4

        self.max_speed = 1.5  # m/s
        self.max_accel = 3.75 # m / s ^ 2
        self.max_yaw_rate = 1.0 # rad / s
        self.max_yaw_accel = 5.9 # rad / s ^ 2;

        # state indices python index starting form 0
        # self.position_indices = [0,1]
        self.position_indices = np.array([[0],[1]])
        self.heading_index = 2
        self.yaw_rate_index = 3
        self.speed_index = 4

        # gains from sys id
        self.accel_motor_gain = 3.0
        self.yaw_accel_motor_gain = 2.95
        self.LLC=agent_LLC()
        self.footprint=0.38
        self.footprint_vertices=np.array([])

    def make_footprint_plot_data(self):
        N=100
        a_vec = np.linspace(0, 2 * np.pi, N)
        C=np.concatenate([self.footprint*np.cos(a_vec).reshape(1, -1),
                          self.footprint*np.sin(a_vec).reshape(1, -1)])
        return C

    def move(self,t_move,T_ref,U_ref,Z_ref):

        #   去掉大于move时间的T

        T_used, U_used, Z_used = self.move_setup(t_move, T_ref, U_ref, Z_ref)

        zcur = self.state[:, -1]
        t_sample = 0.01
        T = np.arange(0, t_move, t_sample)
        T = np.append(T, t_move)

        Z = odeint(self.dynamics, zcur, T, args=(T_ref, U_ref, Z_ref))
        Z=Z.T
        # commit data
        self.state_previous=self.state
        self.state=np.concatenate((self.state,Z[:,1:]),axis=1)
        self.time=np.concatenate((self.time,self.time[-1]+T[1:]),axis=0)

        # self.input_time = [self.input_time, self.input_time[-1] + T_used[2:-1]]
        # self.input = [self.input, U_used[:, 1: -2]]


    def move_setup(self,t_move,T_ref,U_ref,Z_ref):
        flag=0
        # 行向量 T_ref
        # T_ref = T_ref.reshape((1, -1))
        T_ref=np.array(T_ref).reshape(1, -1)
        U_ref=np.array(U_ref)
        T = T_ref[T_ref<=t_move]
        T = np.append(T,t_move)

        f = interpolate.interp1d(T_ref.squeeze(), U_ref, kind='linear', fill_value="extrapolate")

        U=f(T)
        f = interpolate.interp1d(T_ref.squeeze(), Z_ref, kind='linear', fill_value="extrapolate")
        Z=f(T)
        return T,U,Z

    def dynamics(self,z,t,T,U,Z):
        x,y,h,w,v=z

        u = self.LLC.get_control_inputs(self,t,z,T,U,Z)

        # if self.count==2014:
        #     print(u)

        w_des = u[0]
        # if self.count==2015:
        #     print(w_des)
        v_des = u[1]
        # if self.count==2015:
        #     print(v_des)

        k_g = self.yaw_accel_motor_gain
        k_a = self.accel_motor_gain
        g = k_g * (float(w_des) - float(w))
        # self.count = self.count+1
        # print(self.count)

        a = k_a * (float(v_des) - float(v))

        # g = bound_values(g, A.max_yaw_accel);
        hi=abs(self.max_yaw_accel)
        lo=-abs(self.max_yaw_accel)

        if g>hi:
            g=hi
        elif g<lo:
            g=lo

        # a = bound_values(a, A.max_accel);
        hi=abs(self.max_accel)
        lo=-abs(self.max_accel)
        if a>hi:
            a=hi
        elif a<lo:
            a=lo

        # output the dynamics
        xd = v * math.cos(h)
        yd = v * math.sin(h)
        hd = w
        wd = g
        vd = a

        zd = [xd,yd,hd,wd,vd]
        return zd



    def reset(self,state):
        self.state=np.zeros((self.n_states,1))
        self.state[0:3,:]=state