import time

from spinup.my_env.rtd.simulator_files.world.world import world
from spinup.my_env.rtd.simulator_files.agent.agent import segway_agent
from spinup.my_env.rtd.simulator_files.planner.planner import planner
import matplotlib.pyplot as plt
import sys
import numpy as np

class segway():
    def __init__(self):
        self.a=segway_agent()
        self.a.footprint_vertices=self.a.make_footprint_plot_data()

        self.w=world()
        self.w.setup_rand()

        self.a.reset(self.w.start)

        self.w.get_world_info(self.a)

        self.p=planner()
        self.p.setup(self.w)

        self.goal_check = False
        self.collision_check = False
        self.plot_flag = True
        self.error_flag = False

        self.done = False
        self.success = 2 # not success and failure

        self.counter=0

    def one_step(self,cur_factor,xaa_factor,yaa_factor):
        # plt.close('all')

        # get seen info
        self.counter=self.counter+1
        self.success = 2
        seen_flag=self.w.get_world_info(self.a)

        if seen_flag:
            indx_dis = np.argsort(self.w.world_info['dis_obs_seen_agent'])[0:2]
            if np.size(self.w.world_info['dis_obs_seen_agent'])==0:
                self.p.rl_state[self.p.obs_deta_x1_index] = 100
                self.p.rl_state[self.p.obs_deta_y1_index] = 100
                self.p.rl_state[self.p.obs_deta_x2_index] = 100
                self.p.rl_state[self.p.obs_deta_y2_index] = 100
            elif np.size(self.w.world_info['dis_obs_seen_agent']) == 1:
                self.p.rl_state[self.p.obs_deta_x1_index] = self.w.world_info['obs_seen_center'][0, indx_dis[0]] - self.a.state[0, -1]
                self.p.rl_state[self.p.obs_deta_y1_index] = self.w.world_info['obs_seen_center'][1, indx_dis[0]] - self.a.state[1, -1]
                self.p.rl_state[self.p.obs_deta_x2_index] = 100
                self.p.rl_state[self.p.obs_deta_y2_index] = 100
            elif np.size(self.w.world_info['dis_obs_seen_agent']) >= 2:
                self.p.rl_state[self.p.obs_deta_x1_index] = abs(self.w.world_info['obs_seen_center'][0, indx_dis[0]] - self.a.state[0, -1])
                self.p.rl_state[self.p.obs_deta_y1_index] = abs(self.w.world_info['obs_seen_center'][1, indx_dis[0]] - self.a.state[1, -1])
                self.p.rl_state[self.p.obs_deta_x2_index] = abs(self.w.world_info['obs_seen_center'][0, indx_dis[1]] - self.a.state[0, -1])
                self.p.rl_state[self.p.obs_deta_y2_index] = abs(self.w.world_info['obs_seen_center'][1, indx_dis[1]] - self.a.state[1, -1])
        else:
            self.p.rl_state[self.p.obs_deta_x1_index] = 100
            self.p.rl_state[self.p.obs_deta_y1_index] = 100
            self.p.rl_state[self.p.obs_deta_x2_index] = 100
            self.p.rl_state[self.p.obs_deta_y2_index] = 100

        T_nom, U_nom, Z_nom, z_goal, O, HLP_waypoints, O_pts, self.error_flag = self.p.replan(self.a, self.w
                                                                                              ,cur_factor
                                                                                              ,xaa_factor
                                                                                              ,yaa_factor
                                                                                              ,opt_num=self.counter)
        if self.error_flag:
            self.done=True
            self.success = False
            return self.p.rl_state.reshape(1, -1), self.done, self.success

        # move agent
        t_move = self.p.t_move

        self.a.move(t_move, T_nom, U_nom, Z_nom)

        # get rl_state
        self.p.rl_state[self.p.deta_x_index] = abs(self.a.state[0, -1] - self.w.goal[0, -1])
        self.p.rl_state[self.p.deta_y_index] = abs(self.a.state[1, -1] - self.w.goal[1, -1])
        self.p.rl_state[self.p.v_x_index] = self.a.state[4, -1] * np.cos(self.a.state[2, -1])
        self.p.rl_state[self.p.v_y_index] = self.a.state[4, -1] * np.sin(self.a.state[2, -1])
        self.p.rl_state[self.p.heading_index] = np.arccos(np.dot(self.a.state[0:2, -1], self.w.goal[0:2, -1]) / \
                                                (np.linalg.norm(self.a.state[0:2, -1]) * np.linalg.norm(self.w.goal[0:2, -1])))

        # checking crash and goal reaching
        self.goal_check = self.w.goal_check(self.a)
        self.collision_check, F_hull_points_plot = self.w.collision_check(self.a)

        #### plotting ###
        if self.plot_flag == False:
            # plt.figure(num=3, figsize=(8, 5))
            if self.goal_check:
                plt.plot(self.a.state[0, :], self.a.state[1, :], color="black", linewidth=2)
            else:
                plt.plot(self.a.state_previous[0, :], self.a.state_previous[1, :], color="black", linewidth=2)
            plt.plot(O[0, :], O[1, :])
            plt.plot(HLP_waypoints[0, :], HLP_waypoints[1, :], '--')

            plt.plot(self.w.goal[0, 0], self.w.goal[1, 0], 'bo')
            plt.plot(z_goal[0], z_goal[1], 'ro')

            plt.plot(O_pts[0], O_pts[1], 'r')

            plt.plot(F_hull_points_plot[0, :], F_hull_points_plot[1, :])

            if self.p.current_obstacles.size != 0:
                plt.plot(self.p.current_obstacles[0, :], self.p.current_obstacles[1, :], color="orange")
                pass
            plt.show()

        if self.counter > 30:
            self.done = True
            self.success = False
        if self.goal_check or self.collision_check:
            # plt.figure(num=3, figsize=(8, 5))
            # if self.goal_check:
            #     plt.plot(a.state[0, :], a.state[1, :], color="black", linewidth=2)
            # else:
            #     plt.plot(a.state_previous[0, :], a.state_previous[1, :], color="black", linewidth=2)
            # plt.plot(O[0, :], O[1, :])
            # plt.plot(HLP_waypoints[0, :], HLP_waypoints[1, :], '--')
            #
            # plt.plot(w.goal[0, 0], w.goal[1, 0], 'bo')
            # plt.plot(z_goal[0], z_goal[1], 'ro')
            # plt.plot(O_pts[0], O_pts[1], 'r')
            #
            # plt.plot(F_hull_points_plot[0, :], F_hull_points_plot[1, :])
            # if p.current_obstacles.size != 0:
            #     plt.plot(p.current_obstacles[0, :], p.current_obstacles[1, :], color="orange")
            # plt.show()

            if self.collision_check :

                self.done=True
                self.success=False

            if self.goal_check:
                self.done = True
                self.success=True


        # 9 X 1
        obs_deta_x1=self.p.rl_state[self.p.obs_deta_x1_index]
        obs_deta_y1=self.p.rl_state[self.p.obs_deta_y1_index]
        obs_deta_x2=self.p.rl_state[self.p.obs_deta_x2_index]
        obs_deta_y2=self.p.rl_state[self.p.obs_deta_y2_index]

        deta_x=self.p.rl_state[self.p.deta_x_index]
        deta_y=self.p.rl_state[self.p.deta_y_index]
        v_x=self.p.rl_state[self.p.v_x_index]
        v_y=self.p.rl_state[self.p.v_y_index]
        heading=self.p.rl_state[self.p.heading_index]
        return self.p.rl_state.reshape(1,-1),self.done,self.success
