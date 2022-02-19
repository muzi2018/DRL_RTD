import numpy as np
from discrete_lidar import obeservation

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy as np
from utils.robot_plot import plot_robot_and_obstacles
from utils.create_obstacles import create_obstacles
from simulator_files.planner.planner import get_env
from simulator_files.planner.planner import astar_planning


class RobotPlayer(obeservation):
    def __init__(self, x, y, theta, v=0.1):
        self.xpos = x
        self.ypos = y
        self.theta = theta
        self.vel = v
        self.w = 0
        self.next_xpos = 0
        self.next_ypos = 0
        self.next_theta = 0
        self.state=0

        self.v_upper = 3
        self.v_lower = -1
        self.w_upper = np.pi * 30.0 / 180.0
        self.w_lower = -np.pi * 30.0 / 180.0

        obeservation.__init__ (self,angle=360,lidarRange=50,accuracy=1,beems=1080)
        self.n_distances = np.zeros(self.beems)
        self.n_intensities = np.zeros(self.beems)

    def forward(self):
        self.xpos = self.next_xpos
        self.ypos = self.next_ypos
        self.theta = self.next_theta
        # print ("car position:", self.xpos, self.ypos, self.theta)

    def try_forward(self):
        self.next_xpos = self.xpos + self.vel * np.cos(self.theta)
        self.next_ypos = self.ypos + self.vel * np.sin(self.theta)
        # self.next_theta = self.theta + self.w
        self.next_theta = self.theta
        # print ("car position:", self.xpos, self.ypos, self.theta)

    def try_forward_lidar(self, state, obs):
        self.n_distances, self.n_intensities, _, _ = obs.observe(mymap=state, location=self.nposition(),
                                                                 theta=self.next_theta)

    def set_action(self, vel, w):
        self.vel = vel
        self.w = w

    def set_angle(self, action, leftRightLimit=45, actionPossible=11):
        sideDivide = int((actionPossible - 1) // 2)
        angle = self.theta - np.pi * (leftRightLimit / 180.0) + action * (
                    float(leftRightLimit / sideDivide) / 180 * np.pi)
        if angle <= -np.pi:
            self.theta = angle + 2 * np.pi
        elif angle > np.pi:
            self.theta = angle - 2 * np.pi
        else:
            self.theta = angle

    def set_speed(self, action, init_speed):
        if action == 0:
            self.vel = 0
        if action == 1:
            self.vel = 0.2 * init_speed
        if action == 2:
            self.vel = 0.5 * init_speed
        if action == 3:
            self.vel = 0.8 * init_speed
        if action == 4:
            self.vel = 1.0 * init_speed
        if action == 5:
            self.vel = 1.2 * init_speed
        if action == 6:
            self.vel = 1.5 * init_speed

    def position(self):
        return int(self.xpos), int(self.ypos)

    def nposition(self):
        return int(self.next_xpos), int(self.next_ypos)

    def simu(self):
        obstacles = create_obstacles(1, 1)
        robot_state=np.array([[self.xpos],[self.ypos]])
        points=plot_robot_and_obstacles(robot_state,obstacles,robot_radius=0.38,world_x=9,world_y=6,num_steps=1,sim_time=0,filename=None)
        return points

if __name__ == '__main__':
    # grid_resolution = 1.0
    # robot_radius = 1.0

    robot=RobotPlayer(x=3,y=3,theta=3)
    points,sx,sy=robot.simu()
    # gx=50;gy=50
    # map_x=60;map_y=60
    # ox, oy = get_env(points)
    #
    # pathx, pathy = astar_planning(sx, sy, gx, gy, ox, oy, grid_resolution, robot_radius)
    #
    # print(pathx)
    # print(pathy)
