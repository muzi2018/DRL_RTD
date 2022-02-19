from spinup.my_env.rtd.simulator_files.world.world import world
from spinup.my_env.rtd.simulator_files.agent.agent import segway_agent
from spinup.my_env.rtd.simulator_files.planner.planner import planner

def run(agents, worlds, planners):
    agents.reset(worlds.state)


    pass

if __name__ == '__main__':
    A = segway_agent()

    P = planner()

    W = world()
    #设置障碍物，起始位姿和目标点
    W.setup()
###########simulation##########
    #设置agent起始位置
    A.reset(W.start)
    #将环境的边界设置为障碍物
    P.setup(W)

    while(1):
        # get world info

        W.get_world_info(A)
        P.replan(A,W)
        pass
