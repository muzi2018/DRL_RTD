import numpy as np
from scipy import interpolate

def segway_trajectory_producing_model(z,t,T_in,U_in):

    x,y,h=z
    f = interpolate.interp1d(T_in.squeeze(), U_in, kind='linear', fill_value="extrapolate")
    u = f(t)

    w_des = u[0]
    v_des = u[1]

    zd=[v_des*np.cos(h),v_des*np.sin(h),w_des]
    return zd
