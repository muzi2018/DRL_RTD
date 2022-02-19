import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy as np
import time


from rtd.simulator_files.planner.planner import get_env
from rtd.simulator_files.planner.planner import astar_planning


def plot_robot_and_obstacles(robot, obstacles, robot_radius,world_x,world_y, num_steps, sim_time, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, world_x), ylim=(0, world_y))
    ax.set_aspect('equal')
    ax.grid()
    line, = ax.plot([], [], '--r')

    robot_patch = Circle((robot[0, 0], robot[1, 0]),
                         robot_radius, facecolor='green', edgecolor='black')
    obstacle_list = []
    time_start = time.time()
    for obstacle in range(np.shape(obstacles)[2]):
        obstacle = Circle((0, 0), robot_radius,
                          facecolor='aqua', edgecolor='black')
        obstacle_list.append(obstacle)

    def init():
        ax.add_patch(robot_patch)
        for obstacle in obstacle_list:
            ax.add_patch(obstacle)
        line.set_data([], [])
        return [robot_patch] + [line] + obstacle_list

    def animate(i):
        robot_patch.center = (robot[0, i], robot[1, i])

        for j in range(len(obstacle_list)):
            obstacle_list[j].center = (obstacles[0, i, j], obstacles[1, i, j])
        line.set_data(robot[0, :i], robot[1, :i])
        return [robot_patch] + [line] + obstacle_list

    init()
    step = (sim_time / num_steps)


    for i in range(num_steps):
        animate(i)
        # 得到顶点
        verts = []
        trans = []
        points = []
        for j in range(len(obstacle_list)):
            verts.append(obstacle_list[j].get_path().vertices)
            trans.append(obstacle_list[j].get_patch_transform())
            points.append(trans[j].transform(verts[j]))

        grid_resolution = 0.5

        sx, sy = robot[0, 0], robot[1, 0]
        gx, gy = 8,5

        ox, oy = get_env(points,world_x,world_y)

        pathx, pathy = astar_planning(sx, sy, gx, gy, ox, oy, grid_resolution, robot_radius)

        time_end = time.time()
        print('time cost', time_end - time_start, 's')

        print(pathx)
        plt.plot(pathx, pathy, '-r')

        for j in range(len(obstacle_list)):
            plt.plot(points[j][:, 0], points[j][:, 1])

        plt.pause(step)

    # Save animation
    if not filename:
        pass
#        return

    ani = animation.FuncAnimation(
        fig, animate, np.arange(1, num_steps), interval=200,
        blit=True, init_func=init)

#    ani.save(filename, "ffmpeg", fps=30)
    return points,robot[0, 0], robot[1, 0]

def plot_robot(robot, timestep, radius=1, is_obstacle=False):
    if robot is None:
        return
    center = robot[:2, timestep]
    x = center[0]
    y = center[1]
    if is_obstacle:
        circle = plt.Circle((x, y), radius, color='aqua', ec='black')
        plt.plot(robot[0, :timestep], robot[1, :timestep], '--r',)
    else:
        circle = plt.Circle((x, y), radius, color='green', ec='black')
        plt.plot(robot[0, :timestep], robot[1, :timestep], 'blue')

    plt.gcf().gca().add_artist(circle)

