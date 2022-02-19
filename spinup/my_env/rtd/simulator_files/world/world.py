import math
import time

import numpy as np
import matplotlib.pyplot as plt
from spinup.my_env.rtd.simulator_files.agent.agent import segway_agent
from spinup.my_env.rtd.utils.geometry.geometry import *
import copy
import scipy.spatial as spt
from shapely.geometry import Polygon
from scipy import interpolate
from spinup.my_env.rtd.simulator_files.agent.agent_LLC import agent_LLC
from scipy.integrate import odeint, solve_bvp, solve_ivp
class world():
    def __init__(self):
        self.start = np.array([])
        self.goal = np.array([])
        self.goal_radius = 0.5
        self.bounds = np.array([-4,5,-2.5,2.5])
        self.bounds_as_check_crash=np.array([])
        self.bounds_as_obstacle = np.array([])
        self.current_time=0

        self.N_obstacles=2
        self.obstacles = np.array([])
        self.obstacles_unseen = np.array([])
        self.obstacles_seen = np.array([])
        self.obstacles_struct={}

        self.buffer = 0.38
        self.obstacle_size_bounds = np.array([0.3,0.3])
        self.obstacle_rotation_bounds = np.array([-math.pi,math.pi])
        self.obstacles_center=np.full([2,1],np.nan)
        self.index_seen=np.full([1,1],np.nan)

        self.world_info={'obstacles':np.array([]),'bounds':np.array([]),'start':np.array([]),'goal':np.array([]),'dimension':2,
                         'obstacles_struct':{},'obs_seen_center':np.full([2,1],np.nan),'dis_obs_seen_agent':np.array([])}

        self.dimension=2

    def setup_rand(self):
        # get room bounds
        B=self.bounds
        xlo = B[0]
        xhi = B[1]
        ylo = B[2]
        yhi = B[3]

        # get obstacle info
        obs_size = self.obstacle_size_bounds
        obs_rotation_bounds = self.obstacle_rotation_bounds

        # generate start position on left side of room with initial
        # heading of 0, and make sure it's not too close to the walls
        b=self.buffer

        xlo = xlo + 2 * b
        xhi = xhi - 2 * b
        ylo = ylo + 2 * b
        yhi = yhi - 2 * b

        self.start = np.array([[xlo], [(yhi - ylo) * np.random.random_sample() + ylo], [0]])
        # self.start = np.array([[-2], [1], [0]])
        self.goal=np.array([[xhi],[(yhi - ylo) * np.random.random_sample() + ylo]])

        # generate obstacles around room
        N_obs=self.N_obstacles
        O=np.full([2,6*N_obs],np.nan)

        llo = obs_size[0]
        lhi = obs_size[1]
        orlo = obs_rotation_bounds[0]
        orhi = obs_rotation_bounds[1]

        xlo = B[0]
        xhi = B[1]
        ylo = B[2]
        yhi = B[3]
        xlo = xlo + b
        xhi = xhi - b
        ylo = ylo + b
        yhi = yhi - b

        idex_obs=0

        for idx in range(0,(6*N_obs-2),6):
            l = (lhi - llo) * np.random.rand() + llo # length
            r = (orhi - orlo) * np.random.rand() + orlo # rotation

            # obstacle rotation
            R = np.array([[math.cos(r),math.sin(r)],
                         [-math.sin(r),math.cos(r)]])

            # obstacle base
            o = np.array([[-l / 2,l / 2,l / 2,- l / 2,- l / 2],
                          [-l / 2,-l / 2,l / 2,l / 2,-l / 2]])

            d_center = 0
            ds_limit = 0.1 * max((xhi - xlo), (yhi - ylo))
            dg_limit = self.goal_radius

            while (d_center<ds_limit or d_center<dg_limit):
                c = np.array([[(xhi - xlo) * np.random.rand()  + xlo],
                              [(yhi - ylo) * np.random.rand()  + ylo]])
                ds = min(dist_point_to_points(self.start[0:2,:],c))
                dg = min(dist_point_to_points(self.goal,c))
                d_center=min(ds,dg)
            O[:, idx: idx + 5] = R.dot(o) + np.tile(c,5)

            self.obstacles_center=np.concatenate((self.obstacles_center,c),axis=1)
        self.obstacles_center=self.obstacles_center[:,1:]
        self.obstacles = O[:,0:-1]
        self.obstacles_seen=np.array([])
        self.N_obstacles = N_obs

        self.obstacles_unseen=self.obstacles

    def setup_No_rand(self):
        # get room bounds
        B=self.bounds
        xlo = B[0]
        xhi = B[1]
        ylo = B[2]
        yhi = B[3]

        # get obstacle info
        obs_size = self.obstacle_size_bounds
        obs_rotation_bounds = self.obstacle_rotation_bounds

        # generate start position on left side of room with initial
        # heading of 0, and make sure it's not too close to the walls
        b=self.buffer

        xlo = xlo + 2 * b
        xhi = xhi - 2 * b
        ylo = ylo + 2 * b
        yhi = yhi - 2 * b

        self.start = np.array([[-2], [2], [0]])
        # self.start = np.array([[-2], [1], [0]])
        self.goal=np.array([[4],[1]])

        # generate obstacles around room
        N_obs=self.N_obstacles
        O=np.full([2,6*N_obs],np.nan)

        llo = obs_size[0]
        lhi = obs_size[1]
        orlo = obs_rotation_bounds[0]
        orhi = obs_rotation_bounds[1]

        xlo = B[0]
        xhi = B[1]
        ylo = B[2]
        yhi = B[3]
        xlo = xlo + b
        xhi = xhi - b
        ylo = ylo + b
        yhi = yhi - b

        for idx in range(0,(6*N_obs-2),6):
            l = (lhi - llo) * 0.5 + llo # length
            r = (orhi - orlo) * 0.5 + orlo # rotation

            # obstacle rotation
            R = np.array([[math.cos(r),math.sin(r)],
                         [-math.sin(r),math.cos(r)]])

            # obstacle base
            o = np.array([[-l / 2,l / 2,l / 2,- l / 2,- l / 2],
                          [-l / 2,-l / 2,l / 2,l / 2,-l / 2]])

            d_center = 0
            ds_limit = 0.1 * max((xhi - xlo), (yhi - ylo))
            dg_limit = self.goal_radius

            c = np.array([[(xhi - xlo) * 0.05 * (idx+1) + xlo],
                          [(yhi - ylo) * 0.05 * (idx+1) + ylo]])
            O[:, idx: idx + 5] = R.dot(o) + np.tile(c,5)
            self.obstacles_center=np.concatenate((self.obstacles_center,c),axis=1)
        self.obstacles_center=self.obstacles_center[:,1:]
        self.obstacles = O[:,0:-1]
        self.obstacles_seen=np.array([])
        self.N_obstacles = N_obs

        self.obstacles_unseen=self.obstacles

    def setup_compare_test_osb(self):
        # get room bounds
        B=self.bounds
        xlo = B[0]
        xhi = B[1]
        ylo = B[2]
        yhi = B[3]

        # get obstacle info
        obs_size = self.obstacle_size_bounds
        obs_rotation_bounds = self.obstacle_rotation_bounds

        # generate start position on left side of room with initial
        # heading of 0, and make sure it's not too close to the walls
        b=self.buffer

        xlo = xlo + 2 * b
        xhi = xhi - 2 * b
        ylo = ylo + 2 * b
        yhi = yhi - 2 * b

        self.start = np.array([[-3.5], [-1], [0]])
        # self.start = np.array([[-2], [1], [0]])
        # self.goal=np.array([[-1],[0]])#cost1
        self.goal=np.array([[-1],[1.5]])#cost2
        # self.goal=np.array([[-2],[-1]])

        # generate obstacles around room
        N_obs=self.N_obstacles
        O=np.full([2,6*N_obs],np.nan)

        llo = obs_size[0]
        lhi = obs_size[1]
        orlo = obs_rotation_bounds[0]
        orhi = obs_rotation_bounds[1]

        xlo = B[0]
        xhi = B[1]
        ylo = B[2]
        yhi = B[3]
        xlo = xlo + b
        xhi = xhi - b
        ylo = ylo + b
        yhi = yhi - b

        for idx in range(0,(6*N_obs-2),6):
            l = (lhi - llo) * 0.5 + llo # length
            r = 0 # rotation

            # obstacle rotation
            R = np.array([[math.cos(r),math.sin(r)],
                         [-math.sin(r),math.cos(r)]])

            # obstacle base
            o = np.array([[-l / 2,l / 2,l / 2,- l / 2,- l / 2],
                          [-l / 2,-l / 2,l / 2,l / 2,-l / 2]])

            d_center = 0
            ds_limit = 0.1 * max((xhi - xlo), (yhi - ylo))
            dg_limit = self.goal_radius
            if idx == 0:
                # c = np.array([[-2],
                #               [-1]])#cost1
                # c = np.array([[0],
                #               [-1]])
                c = np.array([[-2],
                              [1]])#cost2
            else:
                c = np.array([[-0.5],
                              [1]])
            O[:, idx: idx + 5] = R.dot(o) + np.tile(c,5)
            self.obstacles_center=np.concatenate((self.obstacles_center,c),axis=1)
        self.obstacles_center=self.obstacles_center[:,1:]
        self.obstacles = O[:,0:-1]
        self.obstacles_seen=np.array([])
        self.N_obstacles = N_obs

        self.obstacles_unseen=self.obstacles

    def get_world_info(self,agent):
        zcur=agent.state[agent.position_indices,-1]
        r=agent.sensor_radius
        obs_seen_nozero=0

        O = self.obstacles_unseen
        if O.size!=0:
            N=np.ceil(O.shape[1]/6)

            indices_seen=np.array([])
            O_out = np.array([])
            for idx in range(0,(6*int(N)-2),6):

                a=copy.copy(O[:,idx:idx+5])
                dToObs = dist_point_to_polyline(zcur,a)#
                if dToObs<=r:
                    if O_out.size==0:
                        O_out = np.concatenate((O[:, idx:idx + 5], np.full((2, 1), np.nan)), axis=1)
                    else:
                        O_out=np.concatenate((O_out,O[:,idx:idx+5],np.full((2,1),np.nan)),axis=1)
                    indices_seen=np.concatenate((indices_seen,np.arange(idx,idx+6)),axis=0)

                if indices_seen.size!=0 and indices_seen[-1]>O.shape[1]-1:
                    indices_seen=indices_seen[0:-1]

            indices_seen=indices_seen.astype(int)

            # load total_obs_seen_index
            # self.index_seen=np.unique(np.concatenate((self.index_seen,indices_seen.reshape(1,-1)),axis=1)).reshape(1,-1)

            O=np.delete(O, indices_seen, axis=1)
            self.obstacles_unseen=O
            if self.obstacles_seen.size==0:
                self.obstacles_seen=O_out
            elif O_out.size>0:
                self.obstacles_seen = np.concatenate((self.obstacles_seen, O_out), axis=1)

        self.world_info['obstacles'] = self.obstacles_seen
        self.world_info['bounds'] = self.bounds
        self.world_info['start'] = self.start
        self.world_info['goal'] = self.goal
        self.world_info['dimension'] = self.dimension

        if self.obstacles_seen.size:
            N_obs_all=self.N_obstacles
            N_obs_seen=int(self.obstacles_seen.shape[1]/6)

            for idx_seen in range(0,(6*int(N_obs_seen)-2),6):
                for idx_search in range(0,(6*int(N_obs_all)-2),6):
                    if np.any(self.obstacles[:,idx_search] == self.obstacles_seen[:,idx_seen]):
                        self.index_seen = np.unique(
                            np.concatenate((self.index_seen, np.arange(idx_search,idx_search+6).reshape(1,-1)), axis=1)).reshape(1, -1)

        # delet index_seen nan
            if np.isnan(self.index_seen[0, -1]):
                self.index_seen = self.index_seen[0, 0:-1].reshape(1, -1)

            self.world_info['dis_obs_seen_agent']=np.array([])
            self.world_info['obs_seen_center']=np.full([2, 1], np.nan)
            #compute obs seen before agent
            for idx_c in range(0, np.size(self.index_seen), 6):
                #get obs before agent
                if self.obstacles_center[:, int(self.index_seen[0, idx_c] / 6)][0] >= agent.state[0, -1]:
                    obs_seen_nozero = 1

                    self.world_info['obs_seen_center'] = np.unique(np.concatenate(
                        (self.world_info['obs_seen_center'],
                         self.obstacles_center[:, int(self.index_seen[0, idx_c] / 6)].reshape(-1, 1)),
                        axis=1),axis=1)
                    #delet obs seen ceter's nan
                    if np.isnan(self.world_info['obs_seen_center'][0, -1]):
                        self.world_info['obs_seen_center'] = self.world_info['obs_seen_center'][:, 0:-1].reshape(-1, 1)
                    #get dis obs seen to agent
                    dis= np.sqrt(np.sum((self.world_info['obs_seen_center'][:,-1].reshape(-1,1)-agent.state[0:2,-1].reshape(-1,1))**2,axis=0))
                    self.world_info['dis_obs_seen_agent'] = np.append(self.world_info['dis_obs_seen_agent'],dis)

        else:
            obs_seen_nozero = 0

        return obs_seen_nozero

    def goal_check(self,agent):
        z=agent.state[0:2,:]
        dz=z-np.tile(self.goal,[1,z.shape[1]])
        out=(min(np.linalg.norm(dz,axis=0))<=self.goal_radius)
        if out:
            print('reaching goal!')
        return out


    def collision_check(self,agent,t_move,T_ref, U_ref, Z_ref):
        start = time.time()
        out=0




        zcur = agent.state[:, -1]
        t_sample = 0.01
        T = np.arange(0, t_move, t_sample)
        T = np.append(T, t_move)
        Z = odeint(agent.dynamics, zcur, T, args=(T_ref, U_ref, Z_ref))
        Z=Z.T
        # commit data
        state=np.concatenate((agent.state,Z[:,1:]),axis=1)
        T = np.concatenate((agent.time, agent.time[-1] + T[1:]), axis=0)



        pos_idx = agent.position_indices.flatten()
        h_idx = agent.heading_index
        fp = agent.footprint_vertices
        # Z = agent.state
        Z = state
        # T = agent.time

        O=np.concatenate((self.obstacles,np.full([2,1],np.nan),self.bounds_as_obstacle),axis=1)
        t_start=self.current_time

        t_log = T >= t_start

        T = T[t_log]
        Z = Z[:, t_log]

        X = Z[pos_idx,:]
        # print(X)
        N = X.shape[1]
        X = np.tile(X.flatten('F').reshape(X.shape[0]*X.shape[1],1), (1, fp.shape[1]))

        F = np.tile(fp,(N,1)) + X

        Fx = np.concatenate((F[::2].T,np.full([1,N],np.nan)))
        Fy = np.concatenate((F[1::2].T,np.full([1,N],np.nan)))

        Fx_flatten = Fx.flatten('F').reshape(1,Fx.shape[0] * Fx.shape[1])
        Fy_flatten = Fy.flatten('F').reshape(1,Fy.shape[0] * Fy.shape[1])
        F = np.concatenate((Fx_flatten,Fy_flatten))
        F = F[:,~np.isnan(F).any(axis=0)]

        F_hull = spt.ConvexHull(points=F.T)
        F_hull_points = F[:, F_hull.vertices]
        p1 = Polygon(F_hull_points.T)

        F_hull_vertices_plot = np.append(F_hull.vertices, F_hull.vertices[0])
        F_hull_points_plot = F[:,F_hull_vertices_plot]

        N_obs = self.N_obstacles
        for idx in range(0, (6 * N_obs - 2), 6):
            obs=O[:, idx: idx + 4].T
            p2 = Polygon(obs)
            out = p1.intersects(p2)
            if out:
                print('crash !!!')
                break
        # self.current_time=agent.time[-1]
        self.current_time=T[-1]



        # from shapely.geometry import Polygon
        # F1=np.array([[0,1,1],[0,1,0],[0,0,0]])
        # Obs1=np.array([[0,0,1],[2,1,2],[0,0,0]])
        #
        # p1 = Polygon(F1)
        # p2 = Polygon(Obs1)
        # print(p1.intersects(p2))
        end = time.time()

        return out,F_hull_points_plot

if __name__ == '__main__':
    a=world()
    a.bounds = np.array([-4, 5, -2.5, 2.5])
    a.N_obstacles = 2
    a.buffer = 0.38
    a.obstacle_size_bounds = np.array([0.3, 0.3])
    a.setup()
    print('ok')

