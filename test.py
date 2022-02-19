# import gym
# import matlab
# import matlab.engine
# import time
# eng.+所有matlab中的操作

# eng = matlab.engine.start_matlab()
# eng.cd('~/Documents/someDirWithMatlabFunctions/')

# eng = matlab.engine.connect_matlab()
# time_start = time.time()
# ret = eng.run_segway_RTD_simulation(nargout=0)
# time_end = time.time()
# print('time cost', time_end - time_start, 's')


# x = 4.0 
# eng.workspace['y'] = x 
# print(eng.workspace['j'])

# engine = matlab.engine.start_matlab() # Start MATLAB process
# print(engine.sqrt(matlab.double([1.,2.,3.,4.,5.])))
# engine = matlab.engine.start_matlab("-desktop") # Start MATLAB process with graphic UI

# env = gym.make('Rtd-v1')
# env.reset()
# while 1:
#     env.render()
# env.close()



# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from matplotlib.patches import Circle
# import numpy as np
#
# circle = Circle((0, 0), radius = 0.75, fc = 'y')
# plt.gca().add_patch(circle)
#
# verts = circle.get_path().vertices
# trans = circle.get_patch_transform()
# points = trans.transform(verts)
# print(points)
#
# plt.plot(points[:,0],points[:,1])
# plt.axis('scaled')
# plt.show()

# from spinup.my_env.rtd.simulator_files.agent.agent import segway_agent
from scipy.integrate import odeint, solve_bvp, solve_ivp
from scipy import interpolate
import numpy as np
import math
import numpy as np

def make_segway_braking_trajectory(t_plan,t_stop,w_des,v_des):
    # set up timing
    t_sample = 0.01
    t_total = t_plan+t_stop
    T=np.arange(0, t_total, t_sample)
    T=np.append(T,t_total)

    t_log = T>t_plan
    braking_scale_power=4
    scale=np.ones(T.shape)
    scale[t_log]=((t_total-T[t_log])/t_stop)**braking_scale_power

    w_traj = w_des*scale
    v_traj = v_des*scale
    U_in = np.vstack((w_traj, v_traj))

    # compute desired trajectory
    z0 = [0,0,0]

    def segway_trajectory_producing_model(z, t, T_in, U_in):
        x,y,theta=z

        f = interpolate.interp1d(T_in, U_in, kind='linear',fill_value="extrapolate")
        u = f(t)
        w_des = u[0]
        v_des = u[1]

        # dynamics
        dzdt = [v_des * np.cos(theta), v_des * np.sin(theta), w_des]
        return dzdt

    Z=odeint(segway_trajectory_producing_model,z0,T,args=(T,U_in))
    Z=np.array(Z)
    Z=np.vstack([Z.T,w_traj,v_traj])

    U = np.zeros((2,len(T)))
    T=T.reshape((1,-1))
    return T,U,Z

# make desired trajectory
def make_segway_desired_trajectory(t_f,w_des,v_des):
    # set up timing
    t_sample = 0.01
    T=np.arange(0, t_f, t_sample)
    T=np.append(T,t_f)
    N_t=len(T)

    # get inputs for desired trajectories
    w_traj = w_des*np.ones([1,N_t],float)
    v_traj = v_des*np.ones([1,N_t],float)
    U_in = np.vstack((w_traj, v_traj))

    # compute desired trajectory
    z0 = [0,0,0]

    def segway_trajectory_producing_model(z, t, T_in, U_in):
        x,y,theta=z

        f = interpolate.interp1d(T_in, U_in, kind='linear',fill_value="extrapolate")
        u = f(t)
        w_des = u[0]
        v_des = u[1]

        # dynamics
        dzdt = [v_des * np.cos(theta), v_des * np.sin(theta), w_des]
        return dzdt

    Z=odeint(segway_trajectory_producing_model,z0,T,args=(T,U_in))
    Z=np.array(Z)
    Z=np.vstack([Z.T,w_traj,v_traj])

    # computer inputs for robot
    a_traj = np.diff(v_traj) / t_sample
    a_traj = np.append(a_traj,0)
    a_traj = a_traj.reshape((1,-1))
    U = np.vstack([w_traj,a_traj])
    T=T.reshape((1,-1))
    return T,U,Z

##########################################running############################################
if __name__ == '__main__':
    import pickle
    f = open('/home/wang/Desktop/data.txt','w')
    list=[1.1,1.2,1.3]
    for i in list:
        str1=f'-------------{i}th data-------------'
        f.write(str1)
        f.write('\n')
        str1=f'data{i} '
        f.write(str1)
        f.write(str(i))
        f.write('\t\n')
        f.write('\n')

    f.close()


    # f = open('/home/wang/Desktop/data.pkl', 'wb')
    # for i in range(3,10):
    #
    #     pickle.dump(str(i),f)
    # f.close()
    #
    # f = open('/home/wang/Desktop/data.pkl','rb')
    # var_pickle1=[]
    # var_pickle1 = pickle.load(f)
    # # var_pickle2 = pickle.load(f)
    #
    # print(var_pickle1)
    # f.close()



    # import cyipopt
    # x0 = [1.0, 5.0, 5.0, 1.0]
    #
    # lb = [1.0, 1.0, 1.0, 1.0]
    # ub = [5.0, 5.0, 5.0, 5.0]
    #
    # cl = [25.0, 40.0]
    # cu = [2.0e19, 40.0]
    #
    #
    # class HS071():
    #
    #     def objective(self, x):
    #         """Returns the scalar value of the objective given x."""
    #         return x[0] * x[3] * np.sum(x[0:3]) + x[2]
    #
    #     def gradient(self, x):
    #         """Returns the gradient of the objective with respect to x."""
    #         return np.array([
    #             x[0] * x[3] + x[3] * np.sum(x[0:3]),
    #             x[0] * x[3],
    #             x[0] * x[3] + 1.0,
    #             x[0] * np.sum(x[0:3])
    #         ])
    #
    #     def constraints(self, x):
    #         """Returns the constraints."""
    #         return np.array((np.prod(x), np.dot(x, x)))
    #
    #     def jacobian(self, x):
    #         """Returns the Jacobian of the constraints with respect to x."""
    #         return np.concatenate((np.prod(x) / x, 2 * x))
    #
    #     def hessianstructure(self):
    #         """Returns the row and column indices for non-zero vales of the
    #         Hessian."""
    #
    #         # NOTE: The default hessian structure is of a lower triangular matrix,
    #         # therefore this function is redundant. It is included as an example
    #         # for structure callback.
    #
    #         return np.nonzero(np.tril(np.ones((4, 4))))
    #
    #     def hessian(self, x, lagrange, obj_factor):
    #         """Returns the non-zero values of the Hessian."""
    #
    #         H = obj_factor * np.array((
    #             (2 * x[3], 0, 0, 0),
    #             (x[3], 0, 0, 0),
    #             (x[3], 0, 0, 0),
    #             (2 * x[0] + x[1] + x[2], x[0], x[0], 0)))
    #
    #         H += lagrange[0] * np.array((
    #             (0, 0, 0, 0),
    #             (x[2] * x[3], 0, 0, 0),
    #             (x[1] * x[3], x[0] * x[3], 0, 0),
    #             (x[1] * x[2], x[0] * x[2], x[0] * x[1], 0)))
    #
    #         H += lagrange[1] * 2 * np.eye(4)
    #
    #         row, col = self.hessianstructure()
    #
    #         return H[row, col]
    #
    #     def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
    #                      d_norm, regularization_size, alpha_du, alpha_pr,
    #                      ls_trials):
    #         """Prints information at every Ipopt iteration."""
    #
    #         msg = "Objective value at iteration #{:d} is - {:g}"
    #
    #         print(msg.format(iter_count, obj_value))
    #
    # nlp = cyipopt.Problem(
    #    n=len(x0),
    #    m=len(cl),
    #    problem_obj=HS071(),
    #    lb=lb,
    #    ub=ub,
    #    cl=cl,
    #    cu=cu,
    # )
    #
    # nlp.add_option('mu_strategy', 'adaptive')
    # nlp.add_option('tol', 1e-7)
    #
    # x, info = nlp.solve(x0)


######################################## mosek test #####################################################
    # from mosek.fusion import *
    # with Model('cqo1') as M:
#############################################################################################################

    # w_0 = 0.0   # rad/s
    # v_0 = 0.75   # m/s
    #
    # # trajectory parameters
    # w_des = 1.0   # rad/s
    # v_des = 1.25   # m/s
    # t_plan = 0.5   # s
    # t_f = 1   # for non-braking trajectory
    #
    # A_go = segway_agent()
    # # set the agents to the initial condition
    # z_0 = np.array([0 ,0 ,0 ,w_0 ,v_0]).reshape([5,1])
    # A_go.reset(z_0)
    # t_stop = A_go.max_speed / A_go.max_accel
    # [T_go,U_go,Z_go]=make_segway_braking_trajectory(t_plan,t_stop,w_des,v_des)
    #
    # A_go.move(t_plan+t_stop,T_go,U_go,Z_go)
    # print('po')

# A_go.move(t_f,T_go,U_go,Z_go)
