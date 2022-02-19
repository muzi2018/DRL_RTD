import difflib
import heapq
import math
import numpy as np
import matplotlib.pyplot as plt
import sympy as sy

from collections import  Counter
import sys
import matlab.engine


class Node:
    def __init__(self, x, y, cost, pind):
        self.x = x  # x position of node
        self.y = y  # y position of node
        self.cost = cost  # g cost of node
        self.pind = pind  # parent index of node

class Para:
    def __init__(self, minx, miny, maxx, maxy, xw, yw, reso, motion):
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy
        self.xw = xw
        self.yw = yw
        self.reso = reso  # resolution of grid world
        self.motion = motion  # motion set


def astar_planning(sx, sy, gx, gy, ox, oy, reso, rr):
    """
    return path of A*.
    :param sx: starting node x [m]
    :param sy: starting node y [m]
    :param gx: goal node x [m]
    :param gy: goal node y [m]
    :param ox: obstacles x positions [m]
    :param oy: obstacles y positions [m]
    :param reso: xy grid resolution
    :param rr: robot radius
    :return: path
    """

    n_start = Node(round(sx / reso), round(sy / reso), 0.0, -1)
    n_goal = Node(round(gx / reso), round(gy / reso), 0.0, -1)

    ox = [x / reso for x in ox]
    oy = [y / reso for y in oy]

    P, obsmap = calc_parameters(ox, oy, rr, reso)

    open_set, closed_set = dict(), dict()
    open_set[calc_index(n_start, P)] = n_start

    q_priority = []
    heapq.heappush(q_priority,
                   (fvalue(n_start, n_goal), calc_index(n_start, P)))

    while True:
        if not open_set:
            break

        _, ind = heapq.heappop(q_priority)
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)

        for i in range(len(P.motion)):
            node = Node(n_curr.x + P.motion[i][0],
                        n_curr.y + P.motion[i][1],
                        n_curr.cost + u_cost(P.motion[i]), ind)

            if not check_node(node, P, obsmap):
                continue

            n_ind = calc_index(node, P)
            if n_ind not in closed_set:
                if n_ind in open_set:
                    if open_set[n_ind].cost > node.cost:
                        open_set[n_ind].cost = node.cost
                        open_set[n_ind].pind = ind
                else:
                    open_set[n_ind] = node
                    heapq.heappush(q_priority,
                                   (fvalue(node, n_goal), calc_index(node, P)))

    pathx, pathy, error_flag = extract_path(closed_set, n_start, n_goal, P)

    return pathx, pathy, error_flag


def calc_holonomic_heuristic_with_obstacle(node, ox, oy, reso, rr):
    n_goal = Node(round(node.x[-1] / reso), round(node.y[-1] / reso), 0.0, -1)

    ox = [x / reso for x in ox]
    oy = [y / reso for y in oy]

    P, obsmap = calc_parameters(ox, oy, reso, rr)

    open_set, closed_set = dict(), dict()
    open_set[calc_index(n_goal, P)] = n_goal

    q_priority = []
    heapq.heappush(q_priority, (n_goal.cost, calc_index(n_goal, P)))

    while True:
        if not open_set:
            break

        _, ind = heapq.heappop(q_priority)
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)

        for i in range(len(P.motion)):
            node = Node(n_curr.x + P.motion[i][0],
                        n_curr.y + P.motion[i][1],
                        n_curr.cost + u_cost(P.motion[i]), ind)

            if not check_node(node, P, obsmap):
                continue

            n_ind = calc_index(node, P)
            if n_ind not in closed_set:
                if n_ind in open_set:
                    if open_set[n_ind].cost > node.cost:
                        open_set[n_ind].cost = node.cost
                        open_set[n_ind].pind = ind
                else:
                    open_set[n_ind] = node
                    heapq.heappush(q_priority, (node.cost, calc_index(node, P)))

    hmap = [[np.inf for _ in range(P.yw)] for _ in range(P.xw)]

    for n in closed_set.values():
        hmap[n.x - P.minx][n.y - P.miny] = n.cost

    return hmap

def check_node(node, P, obsmap):
    if node.x <= P.minx or node.x >= P.maxx or \
            node.y <= P.miny or node.y >= P.maxy:
        return False

    if obsmap[int(node.x - P.minx)][int(node.y - P.miny)]:
        return False

    return True

def u_cost(u):
    return math.hypot(u[0], u[1])


def fvalue(node, n_goal):
    return node.cost + h(node, n_goal)


def h(node, n_goal):
    return math.hypot(node.x - n_goal.x, node.y - n_goal.y)


def calc_index(node, P):
    return (node.y - P.miny) * P.xw + (node.x - P.minx)


def calc_parameters(ox, oy, rr, reso):
    minx, miny = round(min(ox)), round(min(oy))
    maxx, maxy = round(max(ox)), round(max(oy))
    xw, yw = maxx - minx, maxy - miny

    motion = get_motion()
    P = Para(minx, miny, maxx, maxy, xw, yw, reso, motion)
    obsmap = calc_obsmap(ox, oy, rr, P)

    return P, obsmap


def calc_obsmap(ox, oy, rr, P):
    obsmap = [[False for _ in range(P.yw)] for _ in range(P.xw)]

    for x in range(P.xw):
        xx = x + P.minx
        for y in range(P.yw):
            yy = y + P.miny
            for oxx, oyy in zip(ox, oy):
                if math.hypot(oxx - xx, oyy - yy) <= rr / P.reso:
                    obsmap[x][y] = True
                    break

    return obsmap


def extract_path(closed_set, n_start, n_goal, P):
    pathx, pathy = [n_goal.x], [n_goal.y]
    n_ind = calc_index(n_goal, P)
    error_flag=0

    num = 0
    while True:
        try:
            num=num+1
            node = closed_set[n_ind]
            pathx.append(node.x)
            pathy.append(node.y)
            n_ind = node.pind
            if node == n_start:
                break
        except:
            error_flag=1
            # print('high_planning_failure')
            break

    # print('extract_path_num:',num)

    pathx = [x * P.reso for x in reversed(pathx)]
    pathy = [y * P.reso for y in reversed(pathy)]
    return pathx, pathy,error_flag


def get_motion():
    motion = [[-1, 0], [-1, 1], [0, 1], [1, 1],
              [1, 0], [1, -1], [0, -1], [-1, -1]]

    return motion


def get_env(obs_points,world_x,world_y):
    ox, oy = [], []

# down
    for i in range(world_x):
        ox.append(i)
        oy.append(1.5)

#right
    for i in range(world_y):
        ox.append(world_x)
        oy.append(i)

#up
    for i in range(world_x):
        ox.append(i)
        oy.append(world_y)

#left
    for i in range(world_y+1):
        ox.append(world_x)
        oy.append(i)
#obs
    for j in range(4):
        for i in range(np.shape(obs_points[j])[0]):
            ox.append(obs_points[j][i, 0])
            oy.append(obs_points[j][i, 1])
    # for i in range(40):
    #     ox.append(20.0)
    #     oy.append(i)
    # for i in range(40):
    #     ox.append(40.0)
    #     oy.append(60.0 - i)

    return ox, oy


def main():
    sx = 10.0  # [m]
    sy = 10.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]

    robot_radius = 2.0
    grid_resolution = 1.0
    ox, oy = get_env()

#    pathx, pathy = astar_planning(sx, sy, gx, gy, ox, oy, grid_resolution, robot_radius)

    plt.plot(ox, oy, 'sk')
    # plt.plot(pathx, pathy, '-r')
    # plt.plot(sx, sy, 'sg')
    # plt.plot(gx, gy, 'sb')
    # plt.axis("equal")
    plt.show()

from spinup.my_env.rtd.simulator_files.world.world import world
from spinup.my_env.rtd.utils.geometry.geometry import *
from spinup.my_env.rtd.simulator_files.agent.agent import segway_agent
from scipy.integrate import odeint, solve_bvp, solve_ivp
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy import interpolate
# from smop.libsmop import *
import time
from spinup.my_env.rtd.utils.geometry.geometry import convert_box_obstacles_to_halfplanes
from cyipopt import minimize_ipopt
import cyipopt
from spinup.my_env.rtd.simulator_files.planner.function import *

class planner():
    def __init__(self):

        k0, k1 = sy.symbols('k0 k1')
        self.FRS=[# FRS 1
                  {'initial_x':-0.5,'initial_y':0,'distance_scale':1.7058,
                   'w_max':1,'v_range(1)':0,'v_range(2)':1,'delta_w':1,
                   'delta_v':0.5,'t_plan':0.5,'w_des':k0,'v_des':(0.5)+(0.5)*k1},

                  # FRS_2
                  {'initial_x':-0.5,'initial_y':0,'distance_scale':1.9178,
                   'w_max': 1, 'v_range(1)': 0, 'v_range(2)': 1.5, 'delta_w': 1,
                   'delta_v': 0.5,'t_plan':0.5,'w_des':k0,'v_des':(0.75)+(0.75)*k1},

                  # FRS_3
                  {'initial_x':-0.5,'initial_y':0,'distance_scale':1.9330,
                   'w_max':1,'v_range(1)':0.5,'v_range(2)':1.5,'delta_w':1,
                   'delta_v':0.5,'t_plan':0.5,'w_des':k0,'v_des':(1)+(0.5)*k1}
                 ]

        self.FRS_polynomial_structure=[
                                        # FRS_1_poly
                                        {'pows':np.array([]),'coef':np.array([]),'z_cols':np.array([]),
                                         'k_cols':np.array([]),'t_cols':np.array([])
                                        },

                                        # FRS_2_poly
                                        {'pows': np.array([]), 'coef': np.array([]), 'z_cols': np.array([]),
                                         'k_cols': np.array([]), 't_cols': np.array([])
                                         },

                                        # FRS_3_poly
                                        {'pows': np.array([]), 'coef': np.array([]), 'z_cols': np.array([]),
                                         'k_cols': np.array([]), 't_cols': np.array([])
                                         }
        ]

        self.current_plan={'T':np.array([]),'U':np.array([]),'Z':np.array([])}

        self.info={'agent_time':np.array([]),'agent_state':np.array([]),'waypoint':np.array([[],[]]),'waypoints':np.array([]),
                   'obstacles':np.array([]),'T':np.array([]),'U':np.array([]),'Z':np.array([])}


        self.agent=segway_agent()
        # self.world=world()

        self.agent_footprint = 0.38
        self.agent_max_speed = 1.5  # m / s
        self.agent_max_accel = 3.75
        self.agent_max_yaw_rate = 1
        self.agent_max_yaw_accel = 5.9
        self.agent_average_speed = 1e10  # over P.agent_average_speed_time_horizon
        self.agent_average_speed_time_horizon = 1  # s
        self.agent_average_speed_threshold = 1e-3  # m / s

        # plan handling
        self.current_waypoint=np.array([])
        self.lookahead_distance = 1.5  # meters
        self.buffer_for_HLP = 0.05

        self.bounds=np.array([])
        #obstacle handling
        self.buffer = 1e-3
        self.point_spacing=0.05

        self.current_obstacles_raw=np.array([])
        self.current_obstacles = np.array([])
        self.current_obstacles_in_FRS_coords=np.array([])

        self.bounds_as_obstacle=np.array([])

        self.t_plan = 0.5
        self.t_move = 0.5

        # rl_state and index
        self.rl_state=np.full([9,1],np.nan)

        self.deta_x_index=0
        self.deta_y_index=1
        self.v_x_index=2
        self.v_y_index=3
        self.heading_index=4

        self.obs_deta_x1_index=5
        self.obs_deta_y1_index=6
        self.obs_deta_x2_index=7
        self.obs_deta_y2_index=8

    def get_current_FRS(self,agent):
        agent_state = agent.state[:,-1]
        v_cur = agent_state[agent.speed_index]
        w_cur = agent_state[agent.yaw_rate_index]

        if v_cur >= 1:
            current_FRS_index = 2
        elif v_cur >= 0.5:
            current_FRS_index = 1
        else:
            current_FRS_index = 0

# use the fastest FRS when the average speed is super low
        if self.agent_average_speed < self.agent_average_speed_threshold:
            current_FRS_index= 2

        FRS_cur = self.FRS[current_FRS_index]
        FRS_Polynomial_cur=self.FRS_polynomial_structure[current_FRS_index]
        return FRS_cur,FRS_Polynomial_cur,current_FRS_index,v_cur,w_cur

    def get_segway_average_speed(self,T,V,t_h):
        t_f = T[-1]
        t_0 = max(T[0], t_f - t_h)
        if t_f>t_0:
            T_avg = np.linspace(t_0, t_f)
        else:
            T_avg=t_f
        if T.size>=2 and V.size>=2:
            f = interpolate.interp1d(T, V, kind='linear', fill_value="extrapolate")
            V_avg=f(T_avg)
            v_avg=float(np.mean(V_avg))
        else:
            v_avg=0
        return v_avg

    def discretize_and_scale_obstacles(self,O_world,pose,b,r,FRS):

        def buffer_polygon_obstacles(O_world,b,miterlim=2):
#            O_world = np.unique(O_world,axis = 1)
            N_obs_seen=(O_world.shape[1]-11)/6
            if N_obs_seen != 0:
                for i in range(0,(6*int(N_obs_seen)-2),6):
                    # 逆时针 0-4
                    O_world[0,i]=O_world[0,i]-b
                    O_world[1,i]=O_world[1,i]-b

                    O_world[0,i+1]=O_world[0,i+1]+b
                    O_world[1,i+1]=O_world[1,i+1]-b

                    O_world[0,i+2]=O_world[0,i+2]+b
                    O_world[1,i+2]=O_world[1,i+2]+b

                    O_world[0,i+3]=O_world[0,i+3]-b
                    O_world[1,i+3]=O_world[1,i+3]+b

                    O_world[0,i+4]=O_world[0,i+4]-b
                    O_world[1,i+4]=O_world[1,i+4]-b

                    O_world[0,i+5]=np.nan
                    O_world[1,i+5]=np.nan

                O_world=np.delete(O_world,i+6,axis=1)

            return O_world

        def interpolate_polyline_with_spacing(O_buf,r):
            dP = np.diff(O_buf)
            Pdists = np.sum(dP**2,axis=0)**0.5

            dlog = Pdists > r
            idxs = np.arange(0,dlog.shape[0])
            idxs = idxs[dlog]

            O=np.zeros([2,1])

            for idx in range(0,O_buf.shape[1]-1):
                if any(idxs==idx):
                    didx = Pdists[idx]
                    Nidx = int(np.ceil(didx / r)) + 1

                    Oidx = np.array([np.linspace(O_buf[0,idx],O_buf[0,idx+1],Nidx),
                                     np.linspace(O_buf[1,idx],O_buf[1,idx+1],Nidx)])
                else:
                    Oidx = O_buf[:,idx:idx+2]

                O = np.concatenate((O,Oidx[:,0:-1]),axis=1)
            O=O[:,1:]
            return O

        def world_to_FRS(P_world,pose,x0_FRS, y0_FRS,x_scale):
            y_scale = x_scale

            x = pose[0,0]
            y = pose[0,1]
            h = pose[0,2]

            I_mat = np.ones([P_world.shape[0], P_world.shape[1]])
            world_offset = np.array([[x,0],[0,y]])
            P_FRS = (P_world-world_offset.dot(I_mat))

            R = np.array([[math.cos(h), math.sin(h)],
                        [-math.sin(h), math.cos(h)]])
            P_FRS = R.dot(P_FRS)

            P_FRS[0, :] = (1/x_scale) * P_FRS[0, :]
            P_FRS[1, :] = (1/y_scale) * P_FRS[1, :]

            FRS_offset = np.array([[x0_FRS, 0],
                                    [0, y0_FRS]])
            P_FRS=FRS_offset.dot(I_mat)+P_FRS

            # plt.plot(P_FRS[0,:],P_FRS[1,:])
            # plt.plot(P_world[0,:],P_world[1,:])
            #
            # plt.show()

            return P_FRS

        def crop_points_outside_region(x,y,P,L):
            ref=np.array([[x],[y]])
            ref=np.tile(ref,P.shape[1])
            P_log= abs(P-ref)<L

            P_log=np.all(P_log , axis=0)


            # n=P_log[0,:]<L
            # m=P_log[1,:]<L
            # P_log=(n==m)


            P=P[:,P_log]
            return P,P_log

        def FRS_to_world(P_FRS,pose,x0_FRS, y0_FRS,x_scale):
            y_scale = x_scale

            x = pose[0, 0]
            y = pose[0, 1]
            h = pose[0, 2]

            I_mat = np.ones([P_FRS.shape[0], P_FRS.shape[1]])
            FRS_offset = np.array([[x0_FRS,0],[0,y0_FRS]])
            P_world = (P_FRS-FRS_offset.dot(I_mat))

            # plt.plot(P_FRS[0, :], P_FRS[1, :])
            # plt.plot(P_world[0, :], P_world[1, :], 'r')
            # plt.show()

            P_world[0, :] = x_scale * P_world[0, :]
            P_world[1, :] = y_scale * P_world[1, :]

            # plt.plot(P_FRS[0, :], P_FRS[1, :])
            # plt.plot(P_world[0, :], P_world[1, :], 'r')
            # plt.show()

            R = np.array([[math.cos(h), -math.sin(h)],
                        [math.sin(h), math.cos(h)]])

            P_world = R.dot(P_world)

            world_offset = np.array([[x, 0],
                                    [0, y]])
            P_world=world_offset.dot(I_mat)+P_world

            return P_world

        O_buf=buffer_polygon_obstacles(O_world,self.buffer)
        O_pts=interpolate_polyline_with_spacing(O_buf,self.point_spacing)

        # plt.plot(O_world[0,:],O_world[1,:])
        # plt.show()

        x0 = FRS['initial_x']
        y0 = FRS['initial_y']
        D = FRS['distance_scale']

#        pose=np.array([[-3],[-1],[0],[0],[0]])
        pose=pose.reshape(1,-1)
        # discard all points behind the robot
        O_FRS = world_to_FRS(O_pts,pose,x0,y0,D)
        O_FRS = O_FRS[:, O_FRS[0, :] >= x0]

        # filter out points that are too far away to be reached
        O_FRS,_=crop_points_outside_region(0,0,O_FRS,1)

        O_FRS_in_world=FRS_to_world(O_FRS,pose,x0,y0,D)

        return O_FRS,O_buf,O_FRS_in_world

    # filter out points
    def process_obstacles(self,agent,world,FRS_cur):
        O = world.world_info['obstacles']

        #O = self.world.obstacles_seen
        if O.size==0:
            O =self.bounds_as_obstacle
        else:
            O = np.concatenate((O,np.full([2,1],np.nan),self.bounds_as_obstacle),axis=1)
            # O=O
        agent_state=agent.state[:,-1]

        # eng = matlab.engine.connect_matlab()
        #
        # w_O = O.tolist()
        # a_s= agent_state.tolist()
        # eng.workspace['w_O'] = matlab.double(w_O)
        # eng.workspace['a_s'] = matlab.double(a_s)

        # plt.plot(O[0,:],O[1,:])
        # plt.show()
        # filter out points
        O_FRS,_,O_pts_in_world=self.discretize_and_scale_obstacles(O,agent_state,self.buffer,self.point_spacing,FRS_cur)

        # O_FRS_1=eng.workspace['O_FRS']
        # O_FRS_1=np.array(O_FRS_1)
        # plt.plot(O_FRS[0,:],O_FRS[1,:])
        # plt.plot(O_FRS_1[0,:],O_FRS_1[1,:])
        #
        # plt.show()

        #O: raw obs
        #O_FRS: let obs be meeted Switch to FRS frame
        #O_pts_in_world: O_FRS -> world frame
        return O,O_FRS,O_pts_in_world

    def setup(self,world):

        # 0 load FRS
        for idx in range(0,3):
            if idx==0:
                b = np.load('/home/wang/Desktop/RL_HarshTerrain_planning/spinningup-master/spinup/my_env/rtd/simulator_files/planner/FRS_1_poly.npz')
                self.FRS_polynomial_structure[idx]['pows'] = b['FRS_1_pows']
                self.FRS_polynomial_structure[idx]['coef'] = b['FRS_1_coef']

                self.FRS_polynomial_structure[idx]['z_cols'] = b['FRS_1_z_cols']
                self.FRS_polynomial_structure[idx]['z_cols'] = np.array([1, 3]).reshape(1, 2)

                self.FRS_polynomial_structure[idx]['k_cols'] = b['FRS_1_k_cols']
                self.FRS_polynomial_structure[idx]['k_cols'] = np.array([0, 2]).reshape(1, 2)

            elif idx==1:
                b = np.load('/home/wang/Desktop/RL_HarshTerrain_planning/spinningup-master/spinup/my_env/rtd/simulator_files/planner/FRS_2_poly.npz')
                self.FRS_polynomial_structure[idx]['pows'] = b['FRS_2_pows']
                self.FRS_polynomial_structure[idx]['coef'] = b['FRS_2_coef']

                self.FRS_polynomial_structure[idx]['z_cols'] = b['FRS_2_z_cols']
                self.FRS_polynomial_structure[idx]['z_cols'] = np.array([1, 3]).reshape(1, 2)

                self.FRS_polynomial_structure[idx]['k_cols'] = b['FRS_2_k_cols']
                self.FRS_polynomial_structure[idx]['k_cols'] = np.array([0, 2]).reshape(1, 2)

            elif idx==2:
                b = np.load('/home/wang/Desktop/RL_HarshTerrain_planning/spinningup-master/spinup/my_env/rtd/simulator_files/planner/FRS_3_poly.npz')
                self.FRS_polynomial_structure[idx]['pows'] = b['FRS_3_pows']
                self.FRS_polynomial_structure[idx]['coef'] = b['FRS_3_coef']

                self.FRS_polynomial_structure[idx]['z_cols'] = b['FRS_3_z_cols']
                self.FRS_polynomial_structure[idx]['z_cols'] = np.array([1, 3]).reshape(1, 2)

                self.FRS_polynomial_structure[idx]['k_cols'] = b['FRS_3_k_cols']
                self.FRS_polynomial_structure[idx]['k_cols'] = np.array([0, 2]).reshape(1, 2)



        # 1 get properties needed from agent,discretization point spacing

        # 2 set up world boundaries as an obstacle
        self.bounds=world.bounds + self.buffer * np.array([1,-1,1,-1])
        xlo = self.bounds[0]
        xhi = self.bounds[1]
        ylo = self.bounds[2]
        yhi = self.bounds[3]

        B = np.array([[xlo, xhi, xhi, xlo, xlo],
                      [ylo, ylo, yhi, yhi, ylo]])
        B=np.concatenate((B, np.full([2,1],np.nan),1.01*np.flip(B)[::-1]), axis=1)

        self.bounds_as_obstacle=B
        world.bounds_as_obstacle=B

        #3 set up high level planner, not neccessary

        #4 process the FRS polynomial, not neccessary

        #5 initialize the current plan as empty

        #6 set up info structure to save replan dat

    def process_world_info(self,world,buffer_distance):
        buffer_distance=self.agent_footprint+self.buffer_for_HLP
        # buffer_distance=0.2
        O = world.world_info['obstacles']

        O_str=convert_box_obstacles_to_halfplanes(O,buffer_distance)
        O_buf=O_str['O']
        self.current_obstacles=O_buf

        world.world_info['obstacles']=O_buf
        world.world_info['obstacles_struct']=O_str
        # 主要处理world的一些属性

    def get_waypoint(self,agent,world):

        def dist_polyline_cumulative(p):
            d=np.diff(p)
            d=np.concatenate((np.array([[0]]), np.sqrt(np.sum(d*d, axis=0)).reshape(1, -1)),
                             axis=1)
            c=np.cumsum(d, axis=1)
            return c

        #lkhd=(self.agent_average_speed + self.lookahead_distance)/2
        # lkhd=self.lookahead_distance/2
        lkhd=self.lookahead_distance/3

        # Obstacles=np.unique(world.obstacles,axis=1)
        # Obstacles=Obstacles[:, ~np.isnan(Obstacles).any(axis=0)]
        if self.current_obstacles.size!=0:
            Obstacles=np.unique(self.current_obstacles,axis=1)
            Obstacles=Obstacles[:, ~np.isnan(Obstacles).any(axis=0)]
            bounds_as_obstacle=self.bounds_as_obstacle[:,0:4]
            o=np.concatenate((Obstacles,bounds_as_obstacle),axis=1)
        else:
            o=self.bounds_as_obstacle[:,0:4]
        ox=o[0,:]
        oy=o[1,:]

        start=time.time()
        pathx, pathy,error_flag=astar_planning(agent.state[0,-1],agent.state[1,-1],
                                    world.goal[0,0], world.goal[1,0],
                                    ox, oy,
                                    reso=0.1, rr=agent.footprint-0.3)
                                    # reso=0.1, rr=agent.footprint-0.3)
        end = time.time()
        # print('high_planner:',end-start)

        pathx = np.array(pathx).reshape(1,-1)
        pathy = np.array(pathy).reshape(1,-1)
        HLP_waypoints=np.concatenate((pathx,pathy))
        HLP_waypoints=np.unique(HLP_waypoints, axis=1)

        waypoint_distance=dist_polyline_cumulative(HLP_waypoints)
        lookahead_distance = min(lkhd, waypoint_distance[:,-1])

        # plt.figure(num=3, figsize=(8, 5))
        # plt.plot(HLP_waypoints[0,:],HLP_waypoints[1,:])
        # plt.plot(agent.state[0,-1],agent.state[1,-1],'ro')
        # plt.plot(world.goal[0,0], world.goal[1,0],'bo')
        # plt.show()
        try:
            f = interpolate.interp1d(waypoint_distance.squeeze(), HLP_waypoints,
                                     kind='linear', fill_value="extrapolate")
        except:
            return self.current_waypoint,pathx,pathy,HLP_waypoints,error_flag

        self.current_waypoint=f(lookahead_distance)
        return self.current_waypoint,pathx,pathy,HLP_waypoints,error_flag

    def create_trajopt_bounds(self,w_cur,v_cur,FRS_cur):
        # create bounds for yaw rate
        w_des_lo=max(w_cur-FRS_cur['delta_w'],-FRS_cur['w_max'])
        w_des_hi=min(w_cur+FRS_cur['delta_w'],FRS_cur['w_max'])
        k_1_lo = w_des_lo / FRS_cur['w_max']
        k_1_hi = w_des_hi / FRS_cur['w_max']
        k_1_bounds = np.array([k_1_lo, k_1_hi]).reshape(1,2)

        # create bounds for speed
        v_max = FRS_cur['v_range(2)']
        v_des_lo = max(v_cur - FRS_cur['delta_v'], FRS_cur['v_range(1)'])
        v_des_hi = min(v_cur + FRS_cur['delta_v'], FRS_cur['v_range(2)'])
        k_2_lo = (v_des_lo - v_max / 2) * (2 / v_max)
        k_2_hi = (v_des_hi - v_max / 2) * (2 / v_max)
        k_2_bounds = np.array([k_2_lo, k_2_hi]).reshape(1,2)

        k_bounds=np.concatenate((k_1_bounds,k_2_bounds))
        return k_bounds

    def evaluate_FRS_polynomial_on_obstacle_points(self,pows,coef,z_cols,k_cols,O):

        # eng = matlab.engine.connect_matlab()
        N=O.shape[1]
        sub_pows = pows[:,[int(z_cols[0,0]),int(z_cols[0,1])]]
        P = sub_pows.T
        O_mat = np.tile(O,(P.shape[1],1))
        P_mat = np.tile(P.flatten('F').reshape(P.shape[0]*P.shape[1],1), (1, N))

        O_mat = O_mat ** P_mat
        O_mat = O_mat[1::2,:] * O_mat[::2,:]

        # O_mat_matlab = np.array(eng.workspace['O_mat'])
        # P_mat_matlab = np.array(eng.workspace['P_mat'])
        # print('O_mat:',(np.around(O_mat_matlab,decimals=3) == np.around(O_mat,decimals=3)).all())
        # print('P_mat:',(np.around(P_mat_matlab,decimals=3) == np.around(P_mat,decimals=3)).all())

        p_k_coef = np.tile(coef, (N, 1)) * O_mat.T
        p_k_pows = pows[:, [int(k_cols[0][0]),int(k_cols[0][1])]]
        # p_k_coef_matlab = np.array(eng.workspace['p_k_coef'])
        # p_k_pows_matlab = np.array(eng.workspace['p_k_pows'])
        # print('p_k_coef:', (np.around(p_k_coef_matlab, decimals=3) == np.around(p_k_coef, decimals=3)).all())
        # print('p_k_pows:', (np.around(p_k_pows_matlab, decimals=3) == np.around(p_k_pows, decimals=3)).all())

        # i_s=p_k_pows[:,0].argsort()
        i_s=np.lexsort((p_k_pows[:,1],p_k_pows[:,0]),axis=0)
        p_k_pows_sorted=p_k_pows[i_s]
        # i_s_matlab = np.array(eng.workspace['i_s'])
        # p_k_pows_sorted_matlab = np.array(eng.workspace['p_k_pows_sorted'])
        # print('i_s:', (np.around(i_s_matlab, decimals=3) == np.around(i_s, decimals=3)).all())
        # print('p_k_pows_sorted:', (np.around(p_k_pows_sorted_matlab, decimals=3) == np.around(p_k_pows_sorted, decimals=3)).all())

        p_k_pows_unique,i_c=np.unique(p_k_pows_sorted, axis=0, return_inverse=True)
        i_c=np.sort(i_c)
        # p_k_pows_unique_matlab = np.array(eng.workspace['p_k_pows_unique'])
        # i_c_matlab = np.array(eng.workspace['i_c'])
        # print('i_c:', (np.around(i_c_matlab, decimals=3) == np.around(i_c, decimals=3)).all())
        # print('p_k_pows_unique:', (np.around(p_k_pows_unique_matlab, decimals=3) == np.around(p_k_pows_unique, decimals=3)).all())


        p_k_coef = p_k_coef[:, i_s]

        split_dic=Counter(i_c)
        start=0
        end=0

        p_k_coef_collapsed = np.full([p_k_coef.shape[0],1],np.nan)

        for num in range(0,np.max(i_c)+1):
            count=split_dic[num]
            end = end + count

            buff = p_k_coef[:,start:end]
            buff = np.sum(buff, axis=1).reshape(-1, 1)  # 118X1
            p_k_coef_collapsed = np.concatenate((p_k_coef_collapsed, buff), axis=1)  # 118X66

            start = end
        p_k_coef_collapsed=p_k_coef_collapsed[:,1:]
        p_k_struct={}
        p_k_struct['coef']=p_k_coef_collapsed
        p_k_struct['pows']=p_k_pows_unique
        p_k_struct['N']=N
        p_k_struct['k_cols']=np.array([0,1])

        return p_k_struct

    def evaluate_FRS_polynomial_on_k(self, pows, coef, z_cols, k_cols, k):

        # eng = matlab.engine.connect_matlab()
        N=k.shape[1]
        sub_pows = pows[:,[int(k_cols[0,0]),int(k_cols[0,1])]]
        P = sub_pows.T
        O_mat = np.tile(k,(P.shape[1],1))
        P_mat = np.tile(P.flatten('F').reshape(P.shape[0]*P.shape[1],1), (1, N))

        O_mat = O_mat ** P_mat
        O_mat = O_mat[1::2,:] * O_mat[::2,:]

        p_z_coef = np.tile(coef, (N, 1)) * O_mat.T
        p_z_pows = pows[:, [int(z_cols[0][0]),int(z_cols[0][1])]]


        i_s=np.lexsort((p_z_pows[:,1],p_z_pows[:,0]),axis=0)
        p_z_pows_sorted=p_z_pows[i_s]

        p_z_pows_unique,i_c=np.unique(p_z_pows_sorted, axis=0, return_inverse=True)
        i_c=np.sort(i_c)

        p_z_coef = p_z_coef[:, i_s]

        split_dic=Counter(i_c)
        start=0
        end=0

        p_z_coef_collapsed = np.full([p_z_coef.shape[0],1],np.nan)

        for num in range(0,np.max(i_c)+1):
            count=split_dic[num]
            end = end + count

            buff = p_z_coef[:,start:end]
            buff = np.sum(buff, axis=1).reshape(-1, 1)  # 118X1
            p_z_coef_collapsed = np.concatenate((p_z_coef_collapsed, buff), axis=1)  # 118X66

            start = end
        p_z_coef_collapsed=p_z_coef_collapsed[:,1:]
        p_z_struct={}
        p_z_struct['coef']=p_z_coef_collapsed
        p_z_struct['pows']=p_z_pows_unique
        p_z_struct['N']=N
        p_z_struct['z_cols']=np.array([0,1])

        return p_z_struct

    def get_constraint_polynomial_gradient(self,cons_poly):

        coef=cons_poly['coef']
        pows=cons_poly['pows']
        N=cons_poly['N']
        Nk=cons_poly['k_cols'].size

        J_coef=np.full([1,coef.shape[1]],np.nan)
        for idx in range(0,Nk):
            J_coef=np.concatenate((J_coef,coef*np.tile(pows[:,idx].T,(N,1))))
        J_coef=J_coef[1:,:]

        J_pows=np.full([pows.shape[0],1],np.nan)
        dpdk_pows=np.full([pows.shape[0],1],np.nan)
        for idx in range(0,Nk):
            dpdk_pows=np.concatenate((dpdk_pows, (pows[:,idx].reshape(-1,1) + ((pows[:,idx] == 0) - 1).reshape(-1,1))) ,
                                     axis=1)
        dpdk_pows = dpdk_pows[:, 1:]

        for idx in range(0, Nk):
            J_pows = np.concatenate((J_pows, pows[:, 0: idx], dpdk_pows[:, idx].reshape(-1,1),
                                     pows[:, idx + 1: Nk]),axis=1)
        J_pows=J_pows[:,1:]

        p_k_struct={}
        p_k_struct['coef']=J_coef
        p_k_struct['pows']=J_pows
        p_k_struct['N']=N

        return p_k_struct


    def optimize_trajectory(self,FRS_cur,FRS_Polynomial_cur,O_FRS,agent,z_goal_world,w_cur,v_cur,k_bounds,opt_num,
                            cur_factor,xaa_factor,yaa_factor):
        error_flag = 0
        try:
            z_goal_local = world_to_local(agent.state[0:4,-1],z_goal_world)
        except:
            error_flag=1
            return error_flag
        k0, k1 = sy.symbols('k0 k1')

        def segway_symbolic_traj_prod_model(z, v_des, w_des):
            x = z[0, 0]
            y = z[1, 0]
            zd = np.array(([v_des - w_des * y], [w_des * x]))
            return zd

        build_start = time.time()
    #==============================build,cost,constrain,gradient,bounds=======================#

        ###########################################
        #              obj                        #
        ###########################################
        cost_cur=0
        cost_xaa=0
        cost_yaa=0

        # obj_build_start=time.time()

        discount_factor = np.array([[0.82], [1.0]])
        # load timing
        dt_int = 0.1
        t_plan = 0.5

        k0, k1 = sy.symbols('k0 k1')
        # z_goal_local = np.array([[1], [2]])

        w_max = 1
        v_max = 1
        w_des = w_max * k0
        v_des = (v_max / 2) * k1 + (v_max / 2)

        z = np.array([[0], [0]])

        for tidx in np.arange(0, t_plan+0.1, dt_int):
            dzdt = segway_symbolic_traj_prod_model(z, v_des, w_des)
            z = z + discount_factor * dt_int * dzdt
            # x_v = v_des * np.cos(dt_int * w_des)

        #                                        additional cost                                        #

        # v_xy
        h=np.array([])
        x_v=np.array([])
        y_v=np.array([])
        for tidx in np.arange(0, t_plan+0.1, dt_int):
            h=np.append(h,tidx*w_des)
            x_v=np.append(x_v,v_des*sy.cos((h[int(10*tidx)])))
            y_v=np.append(y_v,v_des*sy.sin((h[int(10*tidx)])))

        # x_v_sum=np.sum(x_v)
        # y_v_sum=np.sum(y_v)

        # # a_xy
        x_a=np.array([])
        y_a=np.array([])
        for tidx in np.arange(0, t_plan, dt_int):
            x_a = np.append(x_a,(x_v[int(10*tidx)+1]-x_v[int(10*tidx)])/dt_int )
            y_a = np.append(y_a,(y_v[int(10*tidx)+1]-y_v[int(10*tidx)])/dt_int )

        # x_a_sum=sy.sqrt((np.sum(x_a))**2)
        # y_a_sum=sy.sqrt(np.sum(y_a)**2)

        # # aa_xy
        x_aa=np.array([])
        y_aa=np.array([])
        for tidx in np.arange(0, t_plan-0.1, dt_int):
            x_aa = np.append(x_aa,(x_a[int(10*tidx)+1]-x_a[int(10*tidx)])/dt_int )
            y_aa = np.append(y_aa,(y_a[int(10*tidx)+1]-y_a[int(10*tidx)])/dt_int )


        # cur_cost
        for i in np.arange(0,5):
            # cost_cur = abs(x_v[i] * y_a[i] - x_a[i] * y_v[i]) / np.power((x_v[i] ** 2 + y_v[i] ** 2), 3 / 2)
            cost_cur = cost_cur+sy.sqrt((x_v[i] * y_a[i] - x_a[i] * y_v[i])**2) / ((x_v[i] ** 2 + y_v[i] ** 2)**(3 / 2))

        # xaa,yaa
        for i in np.arange(0,4):
            cost_xaa = cost_xaa+sy.sqrt(x_aa[i]**2)
            cost_yaa = cost_yaa+sy.sqrt(y_aa[i]**2)

        cur_factor=0
        xaa_factor=0
        yaa_factor=0
#######################################################################################################################

        obj = np.sum((z_goal_local - z) ** 2) + float(xaa_factor)*cost_xaa+ \
              float(yaa_factor)*cost_yaa + float(cur_factor)*cost_cur

        # obj = np.sum((z_goal_local - z) ** 2) + cur_factor*cost_cur + xaa_factor*cost_xaa + yaa_factor*cost_yaa

        # obj_build_end = time.time()
        # print('--------------obj_build_time--------------:{}s'.format(obj_build_end-obj_build_start))
        ###############################################
        #              obj grad                       #
        ###############################################
        # obj_grad_build_start = time.time()
        grad = np.array([sy.diff(obj, k0), sy.diff(obj, k1)])
        # obj_grad_build_end = time.time()
        # print('--------------obj_grad_build_time--------------:{}'.format(obj_grad_build_end - obj_grad_build_start))

        ################################################
        #              constrain                       #
        ################################################
        # constrain_build_start = time.time()
        pows = FRS_Polynomial_cur['pows']
        coef = FRS_Polynomial_cur['coef']
        z_cols = FRS_Polynomial_cur['z_cols']
        k_cols = FRS_Polynomial_cur['k_cols']

        z = O_FRS

        cons_poly=self.evaluate_FRS_polynomial_on_obstacle_points(pows,coef,z_cols,k_cols,O_FRS)
        # constrain_build_end = time.time()
        # print('--------------constrain_build_time--------------:{}'.format(constrain_build_end - constrain_build_start))

        #####################################################
        #              constrain grad                       #
        #####################################################

        # constrain_grad_build_start = time.time()
        cons_poly_grad = self.get_constraint_polynomial_gradient(cons_poly)
        # constrain_grad_build_end = time.time()
        # print('--------------constrain_grad_build_time--------------:{}'.format(constrain_grad_build_end - constrain_grad_build_start))
        #####################################################
        #              opt class                            #
        #####################################################
        # k0:w
        # k1:v

        # k_bounds = self.create_trajopt_bounds(w_cur, v_cur, FRS_cur)
        # constrain limitation
        cu=[0]*O_FRS.shape[1]
        cl=[-2e8]*O_FRS.shape[1]

        # variable limitation
        lb=[k_bounds[0,0],k_bounds[1,0]]
        ub=[k_bounds[0,1],k_bounds[1,1]]

        # k_init=[0,1]
        k_init = [(k_bounds[0,0]+k_bounds[0,1])/2, (k_bounds[0,1]+k_bounds[1,1])/2]

        class rtd_opt_online():
            def __init__(self):
                self.FRS_Polynomial_cur=FRS_Polynomial_cur

                self.O_FRS=O_FRS

                self.obj=obj
                self.grad_obj=grad

                # self.cons=cons
                # self.jacobian_cons=jacobian

        ####################### cost function ####################################
            def objective(self, k):
                # obj_complete_start = time.time()
                obj = self.obj.subs([(k0, k[0]), (k1, k[1])])

                # obj_complete_end = time.time()
                # print('--------------obj_complete_time--------------:{}s'.format(
                #     obj_complete_end - obj_complete_start))
                return obj

            ####################### cost gradient ####################################
            def gradient(self, k):
                # obj_gradient_complete_start = time.time()

                grad=np.array([])
                grad=np.append(grad,self.grad_obj[0].subs([(k0, k[0]), (k1, k[1])]))
                grad=np.append(grad,self.grad_obj[1].subs([(k0, k[0]), (k1, k[1])]))

                # obj_gradient_complete_end = time.time()
                # print('--------------obj_gradient_complete_time--------------:{}s'.format(
                #     obj_gradient_complete_end - obj_gradient_complete_start))

                return grad

            ####################### constraints ####################################
            def constraints(self,k):
                cons = cons_poly['coef'].dot(np.sum(np.tile(k, (cons_poly['pows'].shape[0], 1)) ** cons_poly['pows'], axis=1).reshape(-1, 1))
                return cons

            ####################### constraints gradient ####################################
            def jacobian(self, k):
                N_k = k.size
                N=cons_poly_grad['N']
                coef=cons_poly_grad['coef']
                pows=cons_poly_grad['pows']

                out=np.zeros([N,N_k])

                for idx in range(0,N_k):
                    a=coef[0+N*(idx):N*(idx+1),:]
                    b=(np.prod(np.tile(k,(pows.shape[0],1))**pows[:,(idx)*N_k+0:(N_k*idx+2)],axis=1)).reshape(-1,1)
                    out[:, idx]=a.dot(b).squeeze()
                return out

        nlp = cyipopt.Problem(
            n=len(k_init),
            m=len(cl),
            problem_obj=rtd_opt_online(),
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu,
        )

        nlp.add_option('tol', 1e-2)
        nlp.add_option('max_iter', 50)
        nlp.add_option('print_level',0)
        # nlp.add_option('max_cpu_time',200.0)

        build_end = time.time()
        # print('building time:',build_end-build_start)
        k,info=nlp.solve(k_init)

        # if (info['status']==0):
        #     print('--------------{}opt_num:successs--------------------\n\n'.format(opt_num))
        # else:
        #     # k = [np.nan,np.nan]
        #     print('--------------{}opt_num:failure--------------\n\n'.format(opt_num))

        return k,info['status'],error_flag

    def increment_plan(self,agent):
        T_old = self.current_plan['T']
        U_old = self.current_plan['U']
        Z_old = self.current_plan['Z']

        if T_old.size!=0:
            T_log = T_old >= self.t_move
        else:
            T_log = False

        if np.any(T_log):
            # make sure P.t_move exists in the trajectory
            T_temp=T_old[T_old<self.t_move]
            T_temp=np.append(T_temp,self.t_move)
            T_temp=np.concatenate((T_temp,T_old[T_old>self.t_move+1e-5]))

            f = interpolate.interp1d(T_old.squeeze(), U_old, kind='linear', fill_value="extrapolate")
            U = f(T_temp)

            f = interpolate.interp1d(T_old.squeeze(), Z_old, kind='linear', fill_value="extrapolate")
            Z = f(T_temp)

            T_log = T_temp >= self.t_move

            #increment the time and input
            T = T_temp[T_log] - self.t_move
            U = U[:, T_log]
            Z = Z[:, T_log]

        else:
            T = np.array([0])
            U = np.zeros([2, 1])
            Z = np.concatenate((agent.state[0:3,-1].reshape(-1,1),np.zeros([2,1])))

        return T, U, Z

    # make yaw rate for spinning in place
    def make_yaw_rate_towards_waypoint(self,z_cur,z_goal):
        z_loc=world_to_local(z_cur,z_goal)
        h_loc=np.arctan2(z_loc[1,0],z_loc[0,0])
        w_des = self.agent_max_yaw_rate * np.sign(h_loc) * np.random.rand(1)
        return w_des

    def make_segway_spin_trajectory(self,t_move,w_des):
        T=np.arange(0,t_move+0.1,0.1).reshape(1,-1)
        if T[0,-1] < t_move:
            T=np.append(T,t_move)
        N_T=T.size
        u=np.append(w_des,0).reshape((2,1))

        U=np.tile(u,[1,N_T])

        dh=u[0,0]*t_move
        Z=np.concatenate((np.zeros([2,N_T]),np.linspace(0,dh,N_T).reshape(1,-1),U))
        return T,U,Z

    def make_segway_braking_trajectory(self,t_plan,t_stop,w_des,v_des):
        t_sample = 0.01
        t_total = t_plan + t_stop
        T = np.arange(0.,t_total,t_sample)
        T = np.append(T,t_total)
        T = np.unique(T).reshape(1,-1)
        #T = np.unique(np.arange(0.,t_total,t_sample))
        T=T.astype(np.float64)

        t_log = T >= t_plan
        braking_scale_power = 4
        scale=np.ones([T.shape[0],T.shape[1]])
        scale[t_log]=((t_stop - T[t_log] + t_plan)/t_stop)**braking_scale_power

        w_traj = w_des * scale
        v_traj = v_des * scale
        U_in = np.concatenate((w_traj,v_traj))

        z0=[0,0,0]
        Z = odeint(segway_trajectory_producing_model, z0, T.squeeze(), args=(T, U_in))

        Z=np.concatenate((Z.T,w_traj,v_traj))
        U=np.zeros((2,T.size))

        return T,U,Z

    def make_plan_for_traj_opt_failure(self,agent):
        # print('Continuing previous plan!')
        # continue desire Z
        T, U, Z=self.increment_plan(agent)

        T=T.reshape(1,-1)

        w_des = self.make_yaw_rate_towards_waypoint(agent.state[0:4, -1], self.current_waypoint)
        T_spin, U_spin, Z_spin = self.make_segway_spin_trajectory(self.t_move, w_des)
        T_spin=T_spin+T[0,-1]

        Z_spin[0: 3,:] = np.tile((agent.state[0: 3, -1].reshape(-1,1)), [1, Z_spin.shape[1]])

        T = np.concatenate((T[0,0:-1].reshape(1,-1), T_spin.reshape(1,-1)),axis=1)
        U = np.concatenate((U[:,0:-1], U_spin),axis=1)
        Z = np.concatenate((Z[:,0:-1], Z_spin),axis=1)

        if T[0,-1]<self.t_move:
            T=np.append(T,[self.t_move,2*self.t_move])
            U=np.concatenate((U,np.zeros(2,1)),axis=1)
            Z_1=np.append(Z[0:3,-1],(0,0)).reshape(-1,1)
            Z=np.concatenate((Z,Z_1,Z_1),axis=1)

        return T,U,Z

    def process_traj_opt_result(self,k_opt,exit_flag,agent,FRS_cur):
        k0, k1 = sy.symbols('k0 k1')
        if exit_flag == 0:
            w_des=FRS_cur['w_des'].subs([(k0, k_opt[0]), (k1, k_opt[1])])
            v_des=FRS_cur['v_des'].subs([(k0, k_opt[0]), (k1, k_opt[1])])

            if v_des < self.agent_average_speed_threshold and self.agent_average_speed < self.agent_average_speed_threshold:
                w_des = self.make_yaw_rate_towards_waypoint(agent.state[:, -1], self.current_waypoint)
                T,U,Z=self.make_segway_spin_trajectory(self.t_move,w_des)

            else:

                t_stop = v_des / self.agent_max_accel
                T, U, Z = self.make_segway_braking_trajectory(FRS_cur['t_plan'],t_stop, w_des, v_des)

            Z[0:3,:]=local_to_world(agent.state[:,-1],Z[0:3,:])
        else:
            T, U, Z = self.make_plan_for_traj_opt_failure(agent)

        return T,U,Z

    def update_info(self,agent, waypoint, T, U, Z):
        I=self.info
        I['agent_time'] = np.append(I['agent_time'],agent.time[-1])
        I['agent_state'] = np.append(I['agent_state'],agent.state[:,-1])
        I['waypoint'] = np.concatenate((I['waypoint'],waypoint.reshape(2,1)),axis=1)

        self.info = I

    def replan(self,agent,world,cur_factor,xaa_factor,yaa_factor,opt_num):
        # 0. start a timer to enforce the planning timeout P.t_plan
        start_tic = time.time()

        # 1. determine the current FRS based on the agent

        FRS_cur, FRS_Polynomial_cur,current_FRS_index, v_cur, w_cur=self.get_current_FRS(agent)

        # also get the agent's average speed (if this average speed is
        # below the average speed threshold, we command the agent to
        # spin in place)
        self.agent_average_speed=self.get_segway_average_speed(agent.time,agent.state[agent.speed_index,:],
                                                               self.agent_average_speed_time_horizon)

        # plt.plot(world.world_info['obstacles'][0,:],world.world_info['obstacles'][1,:])
        # plt.show()

        # 2. process obstacles for RTD
            # O_pts: O_FRS->O_pts
        O,O_FRS,O_pts=self.process_obstacles(agent,world,FRS_cur)

        # plt.plot(O_FRS[0, :], O_FRS[1, :])
        # plt.show()

        # plt.plot(O[0, :], O[1, :])
        # plt.plot(O_pts[0, :], O_pts[1, :], 'r')
        # plt.xlabel('O_FRS -> O and O_discr')
        # plt.legend()
        # plt.axis([-4, 5, -2.5, 2.5])
        # plt.show()

        self.current_obstacles_raw=O # input from the world
        self.current_obstacles=O_pts # buffered and discretized
        self.current_obstacles_in_FRS_coords=O_FRS

        # plt.plot(O_pts[0, :], O_pts[1, :])
        # plt.axis([-4.5, 5.5, -3, 3])
        # plt.show()

        # 3 4 5. create the cost function for trajectory optimization
        self.process_world_info(world,self.agent_footprint+self.buffer_for_HLP)

        try:
            z_goal,pathx,pathy,HLP_waypoints,error_flag = self.get_waypoint(agent, world)
                # optimization
        except:
            error_flag=1

        if error_flag == 0:

            z_goal_copy=np.copy(z_goal)
            k_bounds = self.create_trajopt_bounds(w_cur, v_cur, FRS_cur)
            # k_bounds=np.array([[-1,1],[-1,0]])

            opt_start_time=time.time()
            # print('factor type:',type(cur_factor),type(xaa_factor),type(yaa_factor))
            # print('factor', cur_factor, xaa_factor, yaa_factor)

            cur_factor=float(cur_factor)
            xaa_factor=float(xaa_factor)
            yaa_factor=float(yaa_factor)
            k_opt,exit_flag,error_flag=self.optimize_trajectory(FRS_cur,FRS_Polynomial_cur,O_FRS,agent,z_goal_copy,w_cur,v_cur,k_bounds,opt_num,
                                                     cur_factor,xaa_factor,yaa_factor)
            opt_end_time = time.time()
            opt_totale_time=opt_end_time-opt_start_time
            # if opt_totale_time>=2.5 and opt_num>1:
            #     print('--------over time---------')
            #     exit_flag=1
            if exit_flag==0:
                print('--------------{}opt_num:successs in {}s--------------------\n\n'
                      .format(opt_num,opt_end_time-opt_start_time))
            else:
                k_opt = [np.nan,np.nan]
                print('--------------{}opt_num:failure in {}s--------------\n\n'
                      .format(opt_num,opt_end_time-opt_start_time))

            # 6. make the new plan, continue the old plan , or spin in place
            T,U,Z=self.process_traj_opt_result(k_opt,exit_flag,agent,FRS_cur)

            self.current_plan['T']=T
            self.current_plan['U']=U
            self.current_plan['Z']=Z

            #7 update the info structure
            self.update_info(agent,z_goal,T,U,Z)
        else:
            T=0
            U=0
            Z=0
            HLP_waypoints=0

        self.plot_FRS(FRS_Polynomial_cur,k_opt)
        return T,U,Z,z_goal,O,HLP_waypoints,O_pts,error_flag

    def plot_FRS(self,FRS_Polynomial_cur,k):
        FRS_poly=0
        k0, k1 = sy.symbols('k0 k1')
        x, y = sy.symbols('x y')

        pows = FRS_Polynomial_cur['pows']
        coef = FRS_Polynomial_cur['coef']
        z_cols = FRS_Polynomial_cur['z_cols']
        k_cols = FRS_Polynomial_cur['k_cols']
        k=np.array(k)

        p_z_struct=self.evaluate_FRS_polynomial_on_k(pows,coef,z_cols,k_cols,k.reshape([2,1]))

        coef=p_z_struct['coef']
        pows=p_z_struct['pows']
        N=p_z_struct['N']
        z_cols=p_z_struct['z_cols']

        N=p_z_struct['coef'].size
        for idx in range(0,N):
            FRS_poly=FRS_poly+coef[0,idx]*(x**pows[idx,0])*(y**pows[idx,1])

        Bounds = [-0.9,0.9,-0.9,0.9]
        x_vec=np.linspace(Bounds[0],Bounds[1],num=20)
        y_vec=np.linspace(Bounds[2],Bounds[3],num=20)
        F=np.zeros(shape=(x_vec.size,y_vec.size))

        for idx in range(0,x_vec.size):
            for idy in range(0, y_vec.size):
                F[idx,idy]=FRS_poly.subs([(x, x_vec[idx]), (y, x_vec[idy])])

        print('')
        pass

if __name__ == '__main__':

    # x_init = [1.0, 5.0, 5.0, 1.0]
    #
    # lb = [1.0, 1.0, 1.0, 1.0]
    # ub = [5.0, 5.0, 5.0, 5.0]
    #
    # cl = [25.0, 40.0]
    # cu = [2.0e19, 40.0]
    #
    # x0,x1,x2,x3=sy.symbols('x0 x1 x2 x3')
    #
    # obj=x0*x3*(x0+x1+x2)+x2
    # grad=np.array([sy.diff(obj,x0),sy.diff(obj,x1),sy.diff(obj,x2),sy.diff(obj,x3)])
    #
    # cons=np.array([x0*x1*x2*x3,x0*x0+x1*x1+x2*x2+x3*x3])
    # jacobian_cons1=np.array([sy.diff(cons[0],x0),sy.diff(cons[0],x1),sy.diff(cons[0],x2),sy.diff(cons[0],x3)])
    # jacobian_cons2=np.array([sy.diff(cons[1],x0),sy.diff(cons[1],x1),sy.diff(cons[1],x2),sy.diff(cons[1],x3)])
    # jacobian = np.concatenate((jacobian_cons1,jacobian_cons2))
    #
    # class HS071():
    #     def __init__(self):
    #         self.obj=obj
    #         self.grad=grad
    #         self.cons=cons
    #         self.jacobian_cons=jacobian
    #
    #     def objective(self, x):
    #         """Returns the scalar value of the objective given x."""
    #         cost=self.obj.subs([(x0, x[0]), (x1, x[1]),(x2, x[2]),(x3, x[3])])
    #         return cost
    #         # return x[0] * x[3] * np.sum(x[0:3]) + x[2]
    #
    #     def gradient(self, x):
    #         """Returns the gradient of the objective with respect to x."""
    #
    #         self.grad[0]=self.grad[0].subs([(x0, x[0]), (x1, x[1]),(x2, x[2]),(x3, x[3])])
    #         self.grad[1]=self.grad[1].subs([(x0, x[0]), (x1, x[1]),(x2, x[2]),(x3, x[3])])
    #         self.grad[2]=self.grad[2].subs([(x0, x[0]), (x1, x[1]),(x2, x[2]),(x3, x[3])])
    #         self.grad[3]=self.grad[3].subs([(x0, x[0]), (x1, x[1]),(x2, x[2]),(x3, x[3])])
    #         return self.grad
    #
    #         # return np.array([
    #         #     x[0] * x[3] + x[3] * np.sum(x[0:3]),
    #         #     x[0] * x[3],
    #         #     x[0] * x[3] + 1.0,
    #         #     x[0] * np.sum(x[0:3])
    #         # ])
    #
    #     def constraints(self, x):
    #         """Returns the constraints."""
    #         '''subject to Numerical accuracy'''
    #         cons = np.arange(0,2).reshape(2)
    #
    #         cons[0]=self.cons[0].subs([(x0, x[0]), (x1, x[1]), (x2, x[2]), (x3, x[3])])
    #         cons[1]=self.cons[1].subs([(x0, x[0]), (x1, x[1]), (x2, x[2]), (x3, x[3])])
    #         return cons
    #         # return np.array((np.prod(x), np.dot(x, x)))
    #
    #     def jacobian(self, x):
    #         """Returns the Jacobian of the constraints with respect to x."""
    #         self.jacobian_cons[0]=self.jacobian_cons[0].subs([(x0, x[0]), (x1, x[1]),(x2, x[2]),(x3, x[3])])
    #         self.jacobian_cons[1]=self.jacobian_cons[1].subs([(x0, x[0]), (x1, x[1]),(x2, x[2]),(x3, x[3])])
    #         self.jacobian_cons[2]=self.jacobian_cons[2].subs([(x0, x[0]), (x1, x[1]),(x2, x[2]),(x3, x[3])])
    #         self.jacobian_cons[3]=self.jacobian_cons[3].subs([(x0, x[0]), (x1, x[1]),(x2, x[2]),(x3, x[3])])
    #
    #         self.jacobian_cons[4]=self.jacobian_cons[4].subs([(x0, x[0]), (x1, x[1]),(x2, x[2]),(x3, x[3])])
    #         self.jacobian_cons[5]=self.jacobian_cons[5].subs([(x0, x[0]), (x1, x[1]),(x2, x[2]),(x3, x[3])])
    #         self.jacobian_cons[6]=self.jacobian_cons[6].subs([(x0, x[0]), (x1, x[1]),(x2, x[2]),(x3, x[3])])
    #         self.jacobian_cons[7]=self.jacobian_cons[7].subs([(x0, x[0]), (x1, x[1]),(x2, x[2]),(x3, x[3])])
    #         return self.jacobian_cons
    #         # return np.concatenate((np.prod(x) / x, 2 * x))
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
    #    n=len(lb),
    #    m=len(cl),
    #    problem_obj=HS071(),
    #    lb=lb,
    #    ub=ub,
    #    cl=cl,
    #    cu=cu,
    # )
    #
    # nlp.add_option('mu_strategy', 'adaptive')
    # nlp.add_option('tol', 1e-3)
    # nlp.add_option('max_iter', 100)
    # nlp.add_option('bound_frac',1e-8)
    # nlp.add_option('bound_push',1e-8)
    #
    # nlp.add_option('dual_inf_tol',1e-3)
    # nlp.add_option('constr_viol_tol',1e-3)
    # nlp.add_option('compl_inf_tol',1e-3)
    #
    # x, info = nlp.solve(x_init)


#====================================================Origin opt========================================================#
    lb = [1.0, 1.0, 1.0, 1.0]
    ub = [5.0, 5.0, 5.0, 5.0]

    cl = [25.0, 40.0]
    cu = [2.0e19, 40.0]

    x_init = [1.0, 5.0, 5.0, 1.0]

    x0,x1,x2,x3=sy.symbols('x0 x1 x2 x3')
    obj=x0*x3*(x0+x1+x2)+x2
    grad=np.array([sy.diff(obj,x0),sy.diff(obj,x1),sy.diff(obj,x2),sy.diff(obj,x3)])

    cons=np.array([x0*x1*x2*x3,x0**2+x1**2+x2**2+x3**2])
    jacobian_cons1=np.array([sy.diff(cons[0],x0),sy.diff(cons[0],x1),sy.diff(cons[0],x2),sy.diff(cons[0],x3)])
    jacobian_cons2=np.array([sy.diff(cons[1],x0),sy.diff(cons[1],x1),sy.diff(cons[1],x2),sy.diff(cons[1],x3)])
    jacobian = np.concatenate((jacobian_cons1,jacobian_cons2))


    class HS071():
        def __init__(self):
            self.obj=obj
            self.grad=grad
            self.cons=cons
            self.jacobian_cons=jacobian

        def objective(self, x):
            """Returns the scalar value of the objective given x."""
            return x[0] * x[3] * np.sum(x[0:3]) + x[2]

        def gradient(self, x):
            """Returns the gradient of the objective with respect to x."""
            return np.array([
                x[0] * x[3] + x[3] * np.sum(x[0:3]),
                x[0] * x[3],
                x[0] * x[3] + 1.0,
                x[0] * np.sum(x[0:3])
            ])

        def constraints(self, x):
            """Returns the constraints."""
            return np.array((np.prod(x), np.dot(x, x)))

        def jacobian(self, x):
            """Returns the Jacobian of the constraints with respect to x."""
            return np.concatenate((np.prod(x) / x, 2 * x))

        def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                         d_norm, regularization_size, alpha_du, alpha_pr,
                         ls_trials):
            """Prints information at every Ipopt iteration."""

            msg = "Objective value at iteration #{:d} is - {:g}"

            print(msg.format(iter_count, obj_value))



    nlp = cyipopt.Problem(
        n=len(x_init),
        m=len(cl),
        problem_obj=HS071(),
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
    )

    nlp.add_option('mu_strategy', 'adaptive')
    nlp.add_option('tol', 1e-5)
    x, info = nlp.solve(x_init)
    print(x,info['status'])
    pass
