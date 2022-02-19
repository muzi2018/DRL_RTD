import numpy as np
from scipy import linalg
from scipy.spatial import ConvexHull, convex_hull_plot_2d
# V: 17 X 2
def vert2lcon(V,tol):

    def rownormalize(A,b):
        if np.size(A)==0:
            return
        normsA=np.sqrt(np.sum(A**2,axis=1))
        normsA=normsA.reshape(-1,1)
        idx = normsA>0
        idx = np.where(idx == True)
        idx = idx[0]

        A[idx,:]=A[idx,:]/normsA[idx]
        b[idx]=b[idx]/normsA[idx]
        return A,b

    def lindep(X,tol):
        #Extract a linearly independent set of columns of a given matrix X
        if np.any(np.nonzero(X))==False:
            Xsub = np.array([])
            idx = np.array([])
            r=0
            return r
        tol=1e-10

        Q, R, E = linalg.qr(X, mode='economic', pivoting=True)
        diagr = abs(np.diag(R))

        # rank estimation
        r=np.where((diagr >= tol * diagr[0])==1)
        r=r[0][-1]+1
        return r


    def vert2con(V,tol):
        hull=ConvexHull(V)
        k=hull.simplices
        c=np.mean(V[np.unique(k),:], axis=0)

        V=V-c
        A=np.full([k.shape[0],V.shape[1]],np.nan)

        dim=V.shape[1]
        ee=np.ones([k.shape[1],1])
        rc=-1

        for ix in range(0,k.shape[0]):
            F=V[k[ix,:],:]
            if lindep(F,tol)==dim:
                rc = rc + 1
                A[rc,:]=np.linalg.solve(F,ee).squeeze()



        A = A[0:rc+1,:]
        b=np.ones([A.shape[0],1])
        b=b+A.dot(c.T).reshape(-1,1)

        # eliminate duplicate constraints:
        A,b=rownormalize(A,b)
        A_b=np.concatenate((A,b),axis=1)
        discard,I=np.unique((A_b*1e6).astype(int),axis=0,return_index=True)

        A=A[I,:]
        b=b[I]
        return A,b

######################################################
    tol=1e-10
    M=V.shape[0]
    N=V.shape[1]

    if M==1:
        A=np.array([])
        b=np.array([])
        Aeq=np.identity(N)
        beq = V.reshape(1,-1).T
        return A,b,Aeq,beq

    p=V[0,:].reshape(1,-1).T
    X=V.T-p

    if M>N:
        Q, R, E = linalg.qr(X, mode='economic', pivoting=True)
    else:
        print('M<=N')

    diagr = abs(np.diag(R))

    if np.any(np.nonzero(diagr)):
        #r=find(diagr >= tol * diagr(1)
        r=np.where((diagr >= tol * diagr[0])==1)
        r=r[0][-1]+1

        iE=np.arange(0,E.size).reshape(1,E.size)
        iE=E

        Rsub = R[0:r,iE].T

        if r>0:
            A,b=vert2con(Rsub,tol)
        elif r==0:
            A=np.array([[1],[-1]])
            b=np.append(np.amax(Rsub,axis=0).reshape(1,-1),-np.amin(Rsub,axis=0).reshape(1,-1),axis=0)

        A=A.dot(Q[:,0:r+1].T)
        b=b+A.dot(p)

        if r<N:
            Aeq = Q[:, r + 1: ].T
            beq = Aeq.dot(p)
        else:
            Aeq = np.array([])
            beq = np.array([])
    else:
        A=np.array([])
        b=np.array([])
        Aeq=np.eye(N)
        beq=p
    return A,b,Aeq,beq
