from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
from spinup.my_env.rtd.utils.vert2lcon import vert2lcon


def buffer_box_obstacles(B,b,N):
    B=np.concatenate((B,np.full([2,1],np.nan)),axis=1)
    # get x and y coords
    Bx=B[0,:]
    By=B[1,:]
    Bx=Bx.reshape(1,-1)
    By=By.reshape(1,-1)

    Nx=4 #number of circles to add to corners
    t=np.linspace(0,2*np.pi,N)
    xc = b * np.cos(t)
    yc = b * np.sin(t)

    XC = np.tile(xc,(4,1))
    YC = np.tile(yc,(4,1))

    B_out = np.array([])
    for idx in range(0,B.shape[1]-1,6):
        X=np.tile(Bx[:,idx:idx+4].T,(1,13))+XC
        Y=np.tile(By[:,idx:idx+4].T,(1,13))+YC

        X=X.reshape(-1,1)
        Y=Y.reshape(-1,1)
        points=np.hstack((X,Y))
        hull=ConvexHull(points)

        k=np.append(hull.vertices,hull.vertices[0])
        Bnew=points[k].T
        if B_out.size==0:
            B_out=np.concatenate((np.full([2,1],np.nan),Bnew), axis=1)
        else:
            B_out=np.concatenate((B_out,np.full([2,1],np.nan),Bnew), axis=1)

    B_out=B_out[:,1:]
    return B_out


def convert_box_obstacles_to_halfplanes(O,b):
    O_buf = np.array([])
    A_O = np.array([])
    b_O = np.array([])

    if O.size!=0:
        if np.isnan(O[0,0]):
            O=O[:,1:]

        N_O = O.shape[1]
        N_obs =np.ceil(N_O/6)

        for idx in range(0,N_O-1,6):
            o=O[:,idx:idx+5]

            if b>0:
                o_buf=buffer_box_obstacles(o,b,13)
            else:
                o_buf=o
            if O_buf.size==0:
                O_buf = np.concatenate(( np.full([2, 1], np.nan), o_buf), axis=1)
            else:
                O_buf = np.concatenate((O_buf,np.full([2,1],np.nan),o_buf),axis=1)

            A_idx,b_idx,_,_ = vert2lcon(o_buf.T,tol=1e-10)
            if A_O.size==0:
                A_O=A_idx
            else:
                A_O = np.concatenate((A_O,A_idx),axis=0)

            if b_O.size==0:
                b_O=b_idx
            else:
                b_O = np.concatenate((b_O,b_idx),axis=0)

            N_halfplanes = b_idx.size

    else:
        N_obs = 0
        N_halfplanes = 0

    O_str = {'O': O_buf, 'A': A_O, 'b': b_O, 'N_obs':N_obs,'N_halfplanes':N_halfplanes}
    return O_str

def dist_point_to_points(p,P):
    if P.shape[1]>0:
        d=np.sqrt(np.sum((abs(P - np.tile(p,(1,P.shape[1]))))**2,0))
    else:
        d=np.Inf
    return d

def dist_point_to_polyline(p,P):
    N=P.shape[0]

    Pa = P[:,0:-1]
    Pb = P[:,1:]

    dP=Pb-Pa
    P2=np.sum(dP**2,axis=0)

    P2p=np.tile(p,(1,P.shape[1]-1))-Pa

    t=np.sum(P2p*dP,axis=0)/P2

    tlog= np.logical_and(t>0,t<1)

    if any(tlog):
        Pa[:,tlog]=Pa[:,tlog]+np.tile(t[tlog],(N,1))*dP[:,tlog]
        Pall=np.concatenate((Pa,P[:,-1].reshape(-1,1)),axis=1)
    else:
        Pall = P

    # get the distance from p to the closest point on P

    d_out = dist_point_to_points(p, Pall)
    d_min = min(d_out)
    return d_min


def world_to_local(robot_pose, P_world):
        robot_pose=robot_pose.reshape(-1,1)
        x = robot_pose[0,0]
        y = robot_pose[1,0]
        h = robot_pose[2,0]

        P_world = P_world.reshape(-1,1)
        P_out = np.copy(P_world)
        N_rows=P_world.shape[0]
        N_cols=P_world.shape[1]

        # shift all the world points to the position of the robot
        P_out[0: 2,:]=P_world[0:2,:] - np.tile(robot_pose[0:2,:],N_cols)

        R = np.array([[np.cos(h), np.sin(h)],
                      [-np.sin(h), np.cos(h)]])

        P_local = R.dot(P_out[0: 2,:])
        return P_local


def local_to_world(robot_pose, P_local):
    x = robot_pose[0]
    y = robot_pose[1]
    h = robot_pose[2]

    P_out = P_local
    N_rows=P_local.shape[0]
    N_cols=P_local.shape[1]

    R=np.array([np.cos(h),-np.sin(h),np.sin(h),np.cos(h)]).reshape(2,2)
    P_out[0: 2,:] = R.dot(P_out[0: 2,:])

    if N_rows>2:
        P_out[2,:] = P_out[2,:] + h

    robot_pose=robot_pose.reshape(-1,1)

    P_out[0: 2,:] = P_out[0: 2,:] + np.tile(robot_pose[0:2,:], (1, N_cols))

    return P_out
