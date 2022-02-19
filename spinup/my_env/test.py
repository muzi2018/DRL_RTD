import heapq
import time

from rtd.simulator_files.world.world import world
from rtd.simulator_files.agent.agent import segway_agent
from rtd.simulator_files.planner.planner import planner
import matplotlib.pyplot as plt
import sys
import matlab.engine
import numpy as np
from matplotlib.patches import Circle

def plot_car_mid(xx,yy,fig):
    ax = fig.add_subplot(111)
    x, y = xx[-1], yy[ -1]
    cir1 = Circle(xy=(x, y), radius=a.footprint, color='b',alpha=0.5)
    ax.add_patch(cir1)
    ax.plot(x, y, 'ro')

def plot_car(a,fig):
    ax = fig.add_subplot(111)
    x, y = a.state[0, -1], a.state[1, -1]
    cir1 = Circle(xy=(x, y), radius=a.footprint, color='b',alpha=0.5)
    ax.add_patch(cir1)
    ax.plot(x, y, 'ro')
def plot_car_reach_goal(a,fig):
    ax = fig.add_subplot(111)
    x, y = a.state[0, -1], a.state[1, -1]
    cir1 = Circle(xy=(x, y), radius=a.footprint, color='b',alpha=0.5)
    ax.add_patch(cir1)
    ax.plot(x, y, 'ro')
    print("reach goal****************")
def plot_goal_circle(w,fig):
    ax = fig.add_subplot(111)
    x, y = w.goal[0, 0], w.goal[1, 0]
    cir1 = Circle(xy=(x, y), radius=w.goal_radius, color='green',alpha=0.5)
    ax.add_patch(cir1)


import scipy.io as sio

# eng = matlab.engine.connect_matlab()
#
# a=[[1],[1],[1]]
# eng.workspace['a']=matlab.double(a)

#
# FRS_2_pows=eng.workspace['FRS_2_pows']
# FRS_2_pows=np.array(FRS_2_pows)
#
# FRS_2_coef=eng.workspace['FRS_2_coef']
# FRS_2_coef=np.array(FRS_2_coef)
#
# FRS_2_k_cols=eng.workspace['FRS_2_k_cols']
# FRS_2_k_cols=np.array(FRS_2_k_cols)
#
# FRS_2_z_cols=eng.workspace['FRS_2_z_cols']
# FRS_2_z_cols=np.array(FRS_2_z_cols)
#
#
#
# FRS_3_pows=eng.workspace['FRS_3_pows']
# FRS_3_pows=np.array(FRS_3_pows)
#
# FRS_3_coef=eng.workspace['FRS_3_coef']
# FRS_3_coef=np.array(FRS_3_coef)
#
# FRS_3_k_cols=eng.workspace['FRS_3_k_cols']
# FRS_3_k_cols=np.array(FRS_3_k_cols)
#
# FRS_3_z_cols=eng.workspace['FRS_3_z_cols']
# FRS_3_z_cols=np.array(FRS_3_z_cols)
#
# np.savez('FRS_2_poly',FRS_2_pows=FRS_2_pows,FRS_2_coef=FRS_2_coef,
#          FRS_2_k_cols=FRS_2_k_cols,FRS_2_z_cols=FRS_2_z_cols)
#
# np.savez('FRS_3_poly',FRS_3_pows=FRS_3_pows,FRS_3_coef=FRS_3_coef,
#          FRS_3_k_cols=FRS_3_k_cols,FRS_3_z_cols=FRS_3_z_cols)
#
# b2=np.load('FRS_2_poly.npz')
# b3=np.load('FRS_3_poly.npz')
# print(b2.files)
# print(b3.files)
while 1:
    print('----a new plan----')
    new_flag=1
    start=time.time()

    a = segway_agent()
    a.footprint_vertices=a.make_footprint_plot_data()

    w = world()
    # set obs
    w.setup_compare_test_osb()

    # set init position
    a.reset(w.start)

    # # get seen info
    # w.get_world_info(a)

    p = planner()
    p.setup(w)

    goal_check=False
    collision_check=False
    plot_flag=True
    # plot_flag=False
    plot_rate=0
    for indx in range(0,100):

        # get seen info
        seen_flag=w.get_world_info(a)

        indx_dis=np.argsort(w.world_info['dis_obs_seen_agent'])[0:2]
        if seen_flag:
            if np.size(w.world_info['dis_obs_seen_agent'])==0:
                p.rl_state[p.obs_deta_x1_index]=100
                p.rl_state[p.obs_deta_y1_index]=100
                p.rl_state[p.obs_deta_x2_index]=100
                p.rl_state[p.obs_deta_y2_index]=100
            elif np.size(w.world_info['dis_obs_seen_agent'])==1:
                p.rl_state[p.obs_deta_x1_index] = abs(w.world_info['obs_seen_center'][0,indx_dis[0]]-a.state[0,-1])
                p.rl_state[p.obs_deta_y1_index] = abs(w.world_info['obs_seen_center'][1,indx_dis[0]]-a.state[1,-1])
                p.rl_state[p.obs_deta_x2_index] = 100
                p.rl_state[p.obs_deta_y2_index] = 100
            elif np.size(w.world_info['dis_obs_seen_agent'])>=2:
                p.rl_state[p.obs_deta_x1_index] = abs(w.world_info['obs_seen_center'][0,indx_dis[0]]-a.state[0,-1])
                p.rl_state[p.obs_deta_y1_index] = abs(w.world_info['obs_seen_center'][1,indx_dis[0]]-a.state[1,-1])
                p.rl_state[p.obs_deta_x2_index] = abs(w.world_info['obs_seen_center'][0,indx_dis[1]]-a.state[0,-1])
                p.rl_state[p.obs_deta_y2_index] = abs(w.world_info['obs_seen_center'][1,indx_dis[1]]-a.state[1,-1])
        else:
            p.rl_state[p.obs_deta_x1_index] = 100
            p.rl_state[p.obs_deta_y1_index] = 100
            p.rl_state[p.obs_deta_x2_index] = 100
            p.rl_state[p.obs_deta_y2_index] = 100

        # T_nom,U_nom,Z_nom,z_goal,O,HLP_waypoints,O_pts,error_flag=p.replan(a, w,0.0005,0.001,0.001,opt_num=indx+1)
        # T_nom,U_nom,Z_nom,z_goal,O,HLP_waypoints,O_pts,error_flag=p.replan(a, w,0.005,0.005,0.01,opt_num=indx+1)
        T_nom,U_nom,Z_nom,z_goal,O,HLP_waypoints,O_pts,error_flag=p.replan(a, w,0,0,0,opt_num=indx+1)
        print(p.rl_state[p.obs_deta_x1_index],p.rl_state[p.obs_deta_y1_index],
              p.rl_state[p.obs_deta_x2_index],p.rl_state[p.obs_deta_y2_index])



        if error_flag==1:
            break
            # move agent
        t_move = p.t_move
        collision_check, F_hull_points_plot = w.collision_check(a,t_move, T_nom, U_nom, Z_nom)



        if (plot_flag==True and plot_rate%5==0) or goal_check :
            fig = plt.figure(num=3, figsize=(8, 5))
            plt.xlabel("x(m)")
            plt.ylabel("y(m)")
            # plt.plot(HLP_waypoints[0, :], HLP_waypoints[1, :], '--')
            plt.plot(a.state[0, :], a.state[1, :], color="black", linewidth=0.8)
            # np.save("agent_x_success1.npy", a.state[0, :])
            # np.save("agent_y_success1.npy", a.state[1, :])
            # np.save("agent_x_success2.npy", a.state[0, :])
            # np.save("agent_y_success2.npy", a.state[1, :])
            plot_car(a, fig)
            plt.plot(O[0, :], O[1, :])
            plt.plot(w.goal[0,0],w.goal[1,0],'ko')
            plot_goal_circle(w,fig)
            plt.plot(F_hull_points_plot[0,:],F_hull_points_plot[1,:],color="green")
            # np.save("F_hull_points_plot_x1.npy", F_hull_points_plot[0, :])
            # np.save("F_hull_points_plot_y1.npy", F_hull_points_plot[1, :])
            # np.save("F_hull_points_plot_x2.npy", F_hull_points_plot[0, :])
            # np.save("F_hull_points_plot_y2.npy", F_hull_points_plot[1, :])

            if p.current_obstacles.size!=0:
                # plt.plot(p.current_obstacles[0,:],p.current_obstacles[1,:],color="orange")
                pass
            plt.show()

        a.move(t_move, T_nom, U_nom, Z_nom)
        # get rl_state
        p.rl_state[p.deta_x_index]=abs(a.state[0,-1]-w.goal[0,-1])
        p.rl_state[p.deta_y_index]=abs(a.state[1,-1]-w.goal[1,-1])
        p.rl_state[p.v_x_index]=a.state[4,-1]*np.cos(a.state[2,-1])
        p.rl_state[p.v_y_index]=a.state[4,-1]*np.sin(a.state[2,-1])
        p.rl_state[p.heading_index]=np.arccos(np.dot(a.state[0:2,-1],w.goal[0:2,-1])/ \
                   (np.linalg.norm(a.state[0:2,-1])*np.linalg.norm(w.goal[0:2,-1])))

        # checking crash and goal reaching
        goal_check=w.goal_check(a)

        if goal_check or collision_check:
            plt.figure(num=3, figsize=(8, 5))
            if goal_check:
                # get seen info
                seen_flag = w.get_world_info(a)
                indx_dis = np.argsort(w.world_info['dis_obs_seen_agent'])[0:2]
                # T_nom, U_nom, Z_nom, z_goal, O, HLP_waypoints, O_pts, error_flag = p.replan(a, w, 0.0005, 0.001, 0.001,
                #                                                                             opt_num=indx + 1)
                T_nom, U_nom, Z_nom, z_goal, O, HLP_waypoints, O_pts, error_flag = p.replan(a, w, 0, 0, 0,
                                                                                            opt_num=indx + 1)

                collision_check, F_hull_points_plot = w.collision_check(a, t_move, T_nom, U_nom, Z_nom)


                fig = plt.figure(num=3, figsize=(8, 5))
                plt.xlabel("x(m)")
                plt.ylabel("y(m)")
                xx = np.load("agent_x_success1.npy")
                yy = np.load("agent_y_success1.npy")

                # xx = np.load("agent_x_success2.npy")
                # yy = np.load("agent_y_success2.npy")
                plt.plot(xx, yy, color="black", linewidth=0.8)
                plot_car_mid(xx, yy, fig)
                xx = np.load("F_hull_points_plot_x1.npy")
                yy = np.load("F_hull_points_plot_y1.npy")
                # xx = np.load("F_hull_points_plot_x2.npy")
                # yy = np.load("F_hull_points_plot_y2.npy")
                plt.plot(xx, yy, color="green")  # 可达域


                # plt.plot(HLP_waypoints[0, :], HLP_waypoints[1, :], '--')
                plt.plot(a.state[0, :], a.state[1, :], color="black", linewidth=0.8)#画出轨迹
                plt.plot(O[0, :], O[1, :])#障碍物
                plt.plot(w.goal[0, 0], w.goal[1, 0], 'ko')#目的点
                plot_goal_circle(w, fig)
                plot_car_reach_goal(a, fig)  # 画出车
                plt.plot(F_hull_points_plot[0, :], F_hull_points_plot[1, :], color="green")#可达域
                plt.show()
                sys.exit()

            else:
                plt.plot(a.state_previous[0, :], a.state_previous[1, :], color="black", linewidth=0.8)


            if p.current_obstacles.size!=0:
                # plt.plot(p.current_obstacles[0,:],p.current_obstacles[1,:],color="orange")
                pass
            plt.show()

            if collision_check:
                sys.exit()
            break











        #### plotting ###
        # if plot_flag==True:
        #
        #     fig = plt.figure(num=3, figsize=(8, 5))
        #
        #     if goal_check:
        #         plt.plot(a.state_previous[0, :], a.state_previous[1, :], color="black", linewidth=2)
        #         plot_car_reach_goal(a, fig)
        #         plt.plot(O[0, :], O[1, :])
        #         plt.plot(w.goal[0, 0], w.goal[1, 0], 'ko')
        #         plt.plot(F_hull_points_plot[0, :], F_hull_points_plot[1, :], color="green")
        #         plot_goal_circle(w,fig)
        #         plt.show()
        #         sys.exit()
        #     else:
        #         plt.plot(a.state_previous[0, :], a.state_previous[1, :], color="black", linewidth=2)
        #         plot_car(a, fig)
        #         # ax = fig.add_subplot()
        #         # plt.plot(a.state_previous[0, -1], a.state_previous[1,-1], 'ro')
        #         #
        #         # # print("a.state_previous:",a.state_previous[0, -1],a.state_previous[1, -1])
        #         #
        #         # x, y = a.state[0, -1], a.state[1, -1]
        #         #
        #         # # print("x,y:",a.state_previous[0, -1],a.state_previous[1, -1])
        #         #
        #         # cir1 = Circle(xy=(x, y), radius=a.footprint, color='k', alpha=0.5)
        #         # ax.add_patch(cir1)
        #         # ax.plot(x, y, 'bo')
        #         # plt.show()
        #
        #     plt.plot(O[0, :], O[1, :])
        #     # plt.plot(HLP_waypoints[0, :], HLP_waypoints[1, :], '--')
        #
        #     plt.plot(w.goal[0,0],w.goal[1,0],'ko')
        #     plot_goal_circle(w,fig)
        #     # plt.plot(z_goal[0], z_goal[1], 'ro')
        #
        #
        #     # plt.plot(O_pts[0], O_pts[1], 'r')
        #
        #     plt.plot(F_hull_points_plot[0,:],F_hull_points_plot[1,:],color="green")
        #
        #     if p.current_obstacles.size!=0:
        #         # plt.plot(p.current_obstacles[0,:],p.current_obstacles[1,:],color="orange")
        #         pass
        #     plt.show()
        #
        # if goal_check or collision_check:
        #     plt.figure(num=3, figsize=(8, 5))
        #     if goal_check:
        #         plt.plot(a.state[0, :], a.state[1, :], color="black", linewidth=2)
        #         plot_car_reach_goal(a, fig)
        #
        #     else:
        #         plt.plot(a.state_previous[0, :], a.state_previous[1, :], color="black", linewidth=2)
        #     plt.plot(O[0, :], O[1, :])
        #     # plt.plot(HLP_waypoints[0, :], HLP_waypoints[1, :], '--')
        #
        #     plt.plot(w.goal[0,0],w.goal[1,0],'ko')
        #     # plt.plot(z_goal[0], z_goal[1], 'ro')
        #     # plt.plot(O_pts[0], O_pts[1], 'r')
        #
        #     plt.plot(F_hull_points_plot[0,:],F_hull_points_plot[1,:],color="green")
        #     if p.current_obstacles.size!=0:
        #         # plt.plot(p.current_obstacles[0,:],p.current_obstacles[1,:],color="orange")
        #         pass
        #     plt.show()
        #
        #     if collision_check:
        #         sys.exit()
        #     break

    end = time.time()
    print('one epoch end in:',end-start)