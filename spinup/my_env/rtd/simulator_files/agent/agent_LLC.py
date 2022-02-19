import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

import math

class agent_LLC:
    def __init__(self):
        self.count=0
        # control gains
        self.x_position_gain = 20
        self.y_position_gain = 20
        self.heading_gain = 10
        self.yaw_rate_gain = 20
        self.speed_gain = 20

        # timing
        self.lookahead_time = 0.02   # s (0.1 works very well)

    def get_control_inputs(self,agent,t,z,T,U,Z):
        t_fdbk = min(t + self.lookahead_time, T[:,-1])

        f = interpolate.interp1d(T.squeeze(), Z, kind='linear', fill_value="extrapolate")
        z_des = f(float(t_fdbk))

        f = interpolate.interp1d(T.squeeze(), U, kind='linear', fill_value="extrapolate")
        u_ff = f(float(t_fdbk))
        u_ff = u_ff.tolist()

        h = z[2]
        w = z[3]
        v = z[4]

        h_des = z_des[2]
        w_des = z_des[3]
        v_des = z_des[4]

        #rotate current and desired position into zero-heading
        R_h = np.array([[math.cos(h) ,-math.sin(h)],
               [math.sin(h),math.cos(h)]])

        p_err = R_h.dot((z_des[0:2] - z[0:2]))

        # get control gains
        k_x = self.x_position_gain 
        k_y = self.y_position_gain 
        k_h = self.heading_gain 
        k_v = self.speed_gain 
        k_w = self.yaw_rate_gain 
        
        u = [k_h * (h_des - h) + k_w * (w_des - w) + k_y * p_err[1],
            k_v * (v_des - v) + k_x * p_err[0]]

        u = u + u_ff

        return u
