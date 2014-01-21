# lsqmodel.py
# Define an abstract class to represent a general
# least-squares problem.
#
# M.Dehghani, Montreal  2012.

from Others.toolslsq import *
from pykrylov.linop import *
from nlpy.krylov import SimpleLinearOperator, ReducedLinearOperator, SymmetricallyReducedLinearOperator
from nlpy.model.nlp import NLPModel
from pysparse.sparse.pysparseMatrix import PysparseMatrix as sp
from pysparse.sparse import PysparseIdentityMatrix as eye

import numpy as np

__docformat__ = 'restructuredtext'


class LSQModel(NLPModel):
    """
    LSQModel creates an instance of least-squares problem.
           minimize    c'x + 1/2|Qx-d|^2

           subject to  LCon <= Bx <= UCon,                       (LSQP)
                       LVar <=  x <= UVar,

    Where `Q`  is an (p x n) and `B`  is an (m x n) matrices.
    """

    def __init__(self, Q, B, d, c, Lcon, Ucon, Lvar, Uvar, **kwargs):

        NLPModel.__init__(self, n=Q.shape[1], m=B.shape[0], Lcon=Lcon, \
                          Ucon=Ucon, Lvar=Lvar, Uvar=Uvar, **kwargs)

        #Initialize the parameters of least squares
        self.Q = Q
        self.B = B
        "Return a contiguous array in memory (C order)."
        self.d = np.ascontiguousarray(d, dtype=float)
        self.c = np.ascontiguousarray(c, dtype=float)

    def close(self):
        """
        Not needed anymore. Here for backward compatibility with
        original LSQModel interface.
        """
        pass

    def obj(self, x):
        """
        Evaluate objective function value at x for the least squares
        problem and returns a floating-point number.
        """
        x = np.ascontiguousarray(x, dtype=float)
        Qx = self.Q * x
        lsq_term = Qx - self.d
        f = np.dot(self.c, x) + 0.5 * np.dot(lsq_term, lsq_term)
        return f

    def A(self):
        " To obtain constraint matrix of the least squares problem."
        Bmat = self.B
        return  Bmat

    def jac(self, x):
        """
        Evaluate Jacobian of constraints.
        In the linear case of constraints this is
        equal to coefficient matrix of the constraints.
        """
        J = self.A()
        return J

    def hess(self, x=None, z=None, obj_num=0):
        "Return the Hessian of the problem which is equal to Q'Q."
        H = self.Q.T*self.Q
        return H

    def cost(self):
        "Return a Numpy array as cost vector."
        return  self.c

    def cons(self, x):
        "Evaluate vector of constraints at x and returns a Numpy array."
        x = np.ascontiguousarray(x, dtype=float)
        Z = ZeroOperator()
        B = Z.ReducedLinearOperator(self.B, x.size, x.size)
        Bx = B * x
        return Bx

    def grad(self, x):
        "Evaluate objective gradient at x and returns a Numpy array."
        x = np.ascontiguousarray(x, dtype=float)
        Q = self.Q
        Qx = Q * x
        Qxd = Qx - self.d
        Qxd = Q.T * Qxd
        cx = self.c + Qxd
        return cx


class LSQRModel(LSQModel):
    """
    LSQRModel creates an instance of least-squares problem
           minimize    c'x + 1/2 |r|^2
           subject to  Qx + r = d,
                       LCon <= Bx <= UCon,                       (LSQR)
                       LVar <=  x <= UVar,

    Where `Q`  is a (p x n) matrix and `B`  is an (m x n) matrix.
    The variables are assumed to be ordered as (x,r).
    """

    def __init__(self, Q, B, d, c, Lcon, Ucon, Lvar, Uvar, **kwargs):

        p, n = Q.shape
        m = B.shape[0]
        _Lcon = np.concatenate((d, Lcon))
        _Ucon = np.concatenate((d, Ucon))
        _Lvar = np.empty(n+p)
        _Lvar[:n] = Lvar
        _Lvar[n:] = -np.inf
        _Uvar = np.empty(n+p)
        _Uvar[:n] = Uvar
        _Uvar[n:] = np.inf

        NLPModel.__init__(self, n=n+p, m=m+p, Lcon=_Lcon, \
                          Ucon=_Ucon, Lvar=_Lvar, Uvar=_Uvar, **kwargs)

        # Initialize the parameters of least squares
        self.Q = Q
        self.B = B
        self.nx = n
        self.nr = p
        self.mx = m
        "Return a contiguous array in memory (C order)."
        self.d = np.ascontiguousarray(d, dtype=float)
        self.c = np.ascontiguousarray(c, dtype=float)

    def obj(self, xr):
        nx, nr = self.nx, self.nr
        x = xr[:nx]
        r = xr[nx:]
        return np.dot(self.c, x) + 0.5 * np.dot(r, r)

    def grad(self, xr):
        nx = self.nx
        r = xr[nx:]
        return np.concatenate((self.c, r))

    #def hess(self, xr= None, *args, **kwargs):
        #nx, nr = self.nx, self.nr
        #H =  PysparseLinearOperator(eye(self.nvar))
        #return H

    def cons(self, xr):
        nx, nr = self.nx, self.nr
        x = xr[:nx]
        r = xr[nx:]
        c = np.empty(self.m)
        c[:nr] = self.Q * x + r
        c[nr:] = self.B * x
        return c

    def jprod(self,x,u):
        """
          nx  nr
         [Q   I][u]  [Qu + v]   nr=4
                   = 
         [B   0][v]  [Bu    ]   mx=3
        """
        p = np.zeros(self.nr + self.mx)
        p[:self.nr] = self.Q*u[:self.nx] + u[self.nx:]
        p[self.nr:] = self.B*u[:self.nx]
        return p

    def jtprod(self,x,u):
        """
         [Q^T   B^T][u]  [Q^Tu + B^Tv]
                       = 
         [I       0][v]  [u          ]
        """
        
        p = np.zeros(self.nr + self.nx)
        p[:self.nx]= self.Q.T*u[:self.nr] + self.B.T*u[self.nr:]
        p[self.nx:]= u[:self.nr]
        return p
    
    def jac(self, x, **kwargs):
        return SimpleLinearOperator(self.n, self.m, symmetric=False,
                         matvec=lambda u: self.jprod(x,u,**kwargs),
                         matvec_transp=lambda u: self.jtprod(x,u,**kwargs))
    # Evaluate matrix-vector product between
    # the Hessian of the Lagrangian and a vector
    def hprod(self, x, y, v, **kwargs):
        return v
    
    def hess(self, x, z, **kwargs):
        return SimpleLinearOperator(self.nx+self.nr, self.nx+self.nr, symmetric=True,
                         matvec=lambda u: self.hprod(x,z,u,**kwargs))

def FormEntireMatrix(on,om,Jop):
    J = np.zeros([on,om])
    for j in range(0,om):
        v = np.zeros(om)
        v[j] = 1.
        J[:,j] = Jop * v
    return J

if __name__ == '__main__':
    from slack_nlp import SlackFrameworkNLP
    from lsq_testproblem import *
    from Others.toolslsq import as_llmat
    #lsqpr = exampleliop()
    ls = l1_ls_class_tp(2,3)
    #print ls.c.size
    #print ls.obj(range(12))
    


    #A = LinearOperator(nargin=3, nargout=3,matvec=lambda v: 2*v, symmetric=True)
    
    #B = LinearOperator(nargin=4, nargout=3, matvec=lambda v: v[:3],
                    #matvec_transp=lambda v: np.concatenate((v,np.zeros(1))))
    
    #C = LinearOperator(nargin=2, nargout=3, matvec=lambda v: np.concatenate((v,np.zeros(1))),
                    #matvec_transp=lambda v: v[:2])
    
    #D = LinearOperator(nargin=2, nargout=4, matvec=lambda v: np.concatenate((v,np.zeros(2))),
                    #matvec_transp=lambda v: v[:2])
    
    #E = LinearOperator(nargin=4, nargout=4, matvec=lambda v: -v, symmetric=True)
    #F = LinearOperator(nargin=2, nargout=2, matvec=lambda v: -v, symmetric=True)
    #print FormEntireMatrix(3,4,B)
    ## Build [A  B  C]
    ##       [B' E  D].
    ##       [C' D' F]
    #print A.symmetric,E.symmetric,F.symmetric
    #print A.shape,B.shape,C.shape
    #print B.T.shape,E.shape,D.shape
    #print C.T.shape,D.T.shape,F.shape
    #K4 = BlockLinearOperator([[A, B,C], [E,D],[F]], symmetric=True)
    ##print FormEntireMatrix(3,2,C)
    #W = K4*2
    #print FormEntireMatrix(9,9,W)
    ##[[A,B,C], [D,E], [F]]
 
    
    remove_type_file()
    #path_ = '/Users/Mohsen/Documents/nlpy_mohsen/lsq/test_problems_pkl/'
    #remove_type_file(type='.npz', path=path_)
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    #def jac(self, xr):
        
        #SimpleLinearOperator()
        
        #"""
        #[Q   I]
        #[B   0]
        #"""
        ## 
        #nx, nr, mx = self.nx, self.nr, self.mx
        #Q = self.Q
        #B = self.B
        #Zero = LinearOperator(nargin=nr, nargout=mx,
                       #matvec=lambda v: 0*v, symmetric=True)
        #I =  IdentityOperator(nr)#PysparseLinearOperator(eye(nr))
        ## Build [Q  I]
        ##       [B  0]
        #J = BlockLinearOperator([[Q, I], [B, Zero]])
        #return J