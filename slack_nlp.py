"""
A framework for converting a general nonlinear program into a form with
(possibly nonlinear) equality constraints and bounds only, by adding slack
variables.
"""

__docformat__ = 'restructuredtext'

import numpy
import numpy as np
from nlpy.model.nlp import NLPModel
from nlpy.tools import List
from pysparse.sparse.pysparseMatrix import PysparseMatrix as sp
from nlpy.krylov import SimpleLinearOperator
from nlpy.tools import List

class SlackFrameworkNLP( NLPModel ):
    """
    General framework for converting a nonlinear optimization problem to a
    form using slack variables.

    In the latter problem, the only inequality constraints are bounds on
    the slack variables. The other constraints are (typically) nonlinear
    equalities.

    The order of variables in the transformed problem is as follows:

    1. x, the original problem variables.

    2. sL = [ sLL | sLR ], sLL being the slack variables corresponding to
       general constraints with a lower bound only, and sLR being the slack
       variables corresponding to the 'lower' side of range constraints.

    3. sU = [ sUU | sUR ], sUU being the slack variables corresponding to
       general constraints with an upper bound only, and sUR being the slack
       variables corresponding to the 'upper' side of range constraints.

    4. tL = [ tLL | tLR ], tLL being the slack variables corresponding to
       variables with a lower bound only, and tLR being the slack variables
       corresponding to the 'lower' side of two-sided bounds.

    5. tU = [ tUU | tUR ], tUU being the slack variables corresponding to
       variables with an upper bound only, and tLR being the slack variables
       corresponding to the 'upper' side of two-sided bounds.

    This framework initializes the slack variables sL, sU, tL, and tU to
    zero by default.

    Note that the slack framework does not update all members of AmplModel,
    such as the index set of constraints with an upper bound, etc., but
    rather performs the evaluations of the constraints for the updated
    model implicitly.
    """

    def __init__(self, model, **kwargs):

        NLPModel.__init__(self, n=model.n, m=model.m, Lcon=model.Lcon, \
                          Ucon=model.Ucon, Lvar=model.Lvar, Uvar=model.Uvar,
                          name= model.name, **kwargs)
        
        self.model = model
        # Save number of variables and constraints prior to transformation
        self.original_n = self.n
        self.original_m = self.m
        self.original_nbounds = self.nbounds
        
        # Number of slacks for inequality constraints with a lower bound
        n_con_low = self.nlowerC + self.nrangeC ; self.n_con_low = n_con_low

        # Number of slacks for inequality constraints with an upper bound
        n_con_up = self.nupperC + self.nrangeC ; self.n_con_up = n_con_up

        # Number of slacks for variables with a lower bound
        n_var_low = self.nlowerB + self.nrangeB ; self.n_var_low = n_var_low

        # Number of slacks for variables with an upper bound
        n_var_up = self.nupperB + self.nrangeB ; self.n_var_up = n_var_up

        # Update effective number of variables and constraints
        self.n  = self.original_n + n_con_low + n_con_up + n_var_low + n_var_up
        self.m  = self.original_m + self.nrangeC + n_var_low + n_var_up

        # Redefine primal and dual initial guesses
        self.original_x0 = self.x0[:]
        self.x0 = numpy.zeros(self.n)
        self.x0[:self.original_n] = self.original_x0[:]

        self.original_pi0 = self.pi0[:]
        self.pi0 = numpy.zeros(self.m)
        self.pi0[:self.original_m] = self.original_pi0[:]
        return

    def InitializeSlacks(self, val=0.0, **kwargs):
        """
        Initialize all slack variables to given value. This method may need to
        be overridden.
        """
        self.x0[self.original_n:] = val
        return
    
    def obj(self, x):
            """
            Return the value of the objective function at `x`. This function is
            specialized since the original objective function only depends on a
            subvector of `x`.
            """
            return self.model.obj(x[:self.original_n])
        
    def grad(self,x):
        "Evaluate objective gradient at x and returns a Numpy array."
        return self.model.grad(x[:self.original_n])

    def cons(self, x):

        """
        Evaluate the vector of general constraints for the modified problem.
        Constraints are stored in the order in which they appear in the
        original problem. If constraint i is a range constraint, c[i] will
        be the constraint that has the slack on the lower bound on c[i].
        The constraint with the slack on the upper bound on c[i] will be stored
        in position m + k, where k is the position of index i in
        rangeC, i.e., k=0 iff constraint i is the range constraint that
        appears first, k=1 iff it appears second, etc.

        Constraints appear in the following order:

        1. [ c  ] general constraints in origninal order
        2. [ cR ] 'upper' side of range constraints
        3. [ b  ] linear constraints corresponding to bounds on original problem
        4. [ bR ] linear constraints corresponding to 'upper' side of two-sided
                  bounds
        """
        n = self.n ; on = self.original_n
        m = self.m ; om = self.original_m
        equalC = self.equalC
        lowerC = self.lowerC ; nlowerC = self.nlowerC
        upperC = self.upperC ; nupperC = self.nupperC
        rangeC = self.rangeC ; nrangeC = self.nrangeC
        
        mslow = on + self.n_con_low
        msup  = mslow + self.n_con_up
        s_low = x[on:mslow]    # len(s_low) = n_con_low
        s_up  = x[mslow:msup]  # len(s_up)  = n_con_up

        c = numpy.empty(m)
        c[:om] = self.model.cons(x[:on])
        c[om:om+nrangeC] = c[rangeC]

        c[equalC] -= self.Lcon[equalC]
        c[lowerC] -= self.Lcon[lowerC] ; c[lowerC] -= s_low[:nlowerC]

        c[upperC] -= self.Ucon[upperC] ; c[upperC] *= -1
        c[upperC] -= s_up[:nupperC]

        c[rangeC] -= self.Lcon[rangeC] ; c[rangeC] -= s_low[nlowerC:]

        c[om:om+nrangeC] -= self.Ucon[rangeC]
        c[om:om+nrangeC] *= -1
        c[om:om+nrangeC] -= s_up[nupperC:]

        # Add linear constraints corresponding to bounds on original problem
        lowerB = self.lowerB ; nlowerB = self.nlowerB ; Lvar = self.Lvar
        upperB = self.upperB ; nupperB = self.nupperB ; Uvar = self.Uvar
        rangeB = self.rangeB ; nrangeB = self.nrangeB

        nt = on + self.n_con_low + self.n_con_up
        ntlow = nt + self.n_var_low
        t_low = x[nt:ntlow]
        t_up  = x[ntlow:]

        b = c[om+nrangeC:]

        b[:nlowerB] = x[lowerB] - Lvar[lowerB] - t_low[:nlowerB]
        b[nlowerB:nlowerB+nrangeB] = x[rangeB] - Lvar[rangeB] - t_low[nlowerB:]
        b[nlowerB+nrangeB:nlowerB+nrangeB+nupperB] =\
         Uvar[upperB] - x[upperB] - t_up[:nupperB]
        b[nlowerB+nrangeB+nupperB:] = Uvar[rangeB] - x[rangeB] - t_up[nupperB:]

        return c


    def Bounds(self, x):
        """
        Evaluate the vector of equality constraints corresponding to bounds
        on the variables in the original problem.
        """
        lowerB = self.lowerB ; nlowerB = self.nlowerB
        upperB = self.upperB ; nupperB = self.nupperB
        rangeB = self.rangeB ; nrangeB = self.nrangeB

        n  = self.n ; on = self.original_n
        mslow = on + nrangeC + self.n_con_low
        msup  = mslow + self.n_con_up
        nt = self.original_n + self.n_con_low + self.n_con_up
        ntlow = nt + self.n_var_low

        t_low  = x[msup:ntlow]
        t_up   = x[ntlow:]

        b = numpy.empty(n + nrangeB)
        b[:n] = x[:]
        b[n:] = x[rangeB]

        b[lowerB] -= self.Lvar[lowerB] ; b[lowerB] -= t_low[:nlowerB]

        b[upperB] -= self.Uvar[upperB] ; b[upperB] *= -1
        b[upperB] -= t_up[:nupperB]

        b[rangeB] -= self.Lvar[rangeB] ; b[rangeB] -= t_low[nlowerB:]
        b[n:]     -= self.Uvar[rangeB] ; b[n:] *= -1
        b[n:]     -= t_up[nupperB:]

        return b
    
    def jprod(self, x, v, **kwargs):

        nlp = self.model
        on = self.original_n
        om = self.original_m
        n = self.n
        m = self.m
    

        # List() simply allows operations such as 1 + [2,3] -> [3,4]
        lowerC = List(nlp.lowerC) ; nlowerC = nlp.nlowerC
        upperC = List(nlp.upperC) ; nupperC = nlp.nupperC
        rangeC = List(nlp.rangeC) ; nrangeC = nlp.nrangeC
        lowerB = List(nlp.lowerB) ; nlowerB = nlp.nlowerB
        upperB = List(nlp.upperB) ; nupperB = nlp.nupperB
        rangeB = List(nlp.rangeB) ; nrangeB = nlp.nrangeB
        nbnds  = nlowerB + nupperB + 2*nrangeB
        nSlacks = nlowerC + nupperC + 2*nrangeC

        p = np.zeros(m)

        p[:om] = nlp.jprod(x[:on], v[:on])
        p[upperC] *= -1.0
        p[om:om+nrangeC] = p[rangeC]
        p[om:om+nrangeC] *= -1.0
        
        # Insert contribution of slacks on general constraints
        bot = on;       p[lowerC] -= v[bot:bot+nlowerC]
        bot += nlowerC; p[rangeC] -= v[bot:bot+nrangeC]
        bot += nrangeC; p[upperC] -= v[bot:bot+nupperC]
        bot += nupperC; p[om:om+nrangeC] -= v[bot:bot+nrangeC]

        # Insert contribution of bound constraints on the original problem
        bot = om+nrangeC; p[bot:bot+nlowerB] += v[lowerB]
        bot += nlowerB;  p[bot:bot+nrangeB] += v[rangeB]
        bot += nrangeB;  p[bot:bot+nupperB] -= v[upperB]
        bot += nupperB;  p[bot:bot+nrangeB] -= v[rangeB]

        ## Insert contribution of slacks on the bound constraints
        #bot = om+nrangeC; p[bot:bot+nlowerB] -= v[self.tLL]
        #bot += nlowerB;  p[bot:bot+nrangeB] -= v[self.tLR]
        #bot += nrangeB;  p[bot:bot+nupperB] -= v[self.tUU]
        #bot += nupperB;  p[bot:bot+nrangeB] -= v[self.tUR]

        return p


    def jtprod(self, x, v, **kwargs):

        nlp = self.model
        on = self.original_n
        om = self.original_m
        n = self.n
        m = self.m

        # List() simply allows operations such as 1 + [2,3] -> [3,4]
        lowerC = List(nlp.lowerC) ; nlowerC = nlp.nlowerC
        upperC = List(nlp.upperC) ; nupperC = nlp.nupperC
        rangeC = List(nlp.rangeC) ; nrangeC = nlp.nrangeC
        lowerB = List(nlp.lowerB) ; nlowerB = nlp.nlowerB
        upperB = List(nlp.upperB) ; nupperB = nlp.nupperB
        rangeB = List(nlp.rangeB) ; nrangeB = nlp.nrangeB
        nbnds  = nlowerB + nupperB + 2*nrangeB
        nSlacks = nlowerC + nupperC + 2*nrangeC

        p = np.zeros(n)
        vmp = v[:om].copy()
        vmp[upperC] *= -1.0
        vmp[rangeC] -= v[om:]

        p[:on] = nlp.jtprod(x[:on], vmp)

        # Insert contribution of slacks on general constraints
        bot = on;       p[on:on+nlowerC]    = -v[lowerC]
        bot += nlowerC; p[bot:bot+nrangeC]  = -v[rangeC]
        bot += nrangeC; p[bot:bot+nupperC]  = -v[upperC]
        bot += nupperC; p[bot:bot+nrangeC]  = -v[om:om+nrangeC]

        #if self.keep_variable_bounds==False:
            # Insert contribution of bound constraints on the original problem
        bot = om+nrangeC; p[lowerB] += v[bot:bot+nlowerB]
        bot += nlowerB;  p[rangeB] += v[bot:bot+nrangeB]
        bot += nrangeB;  p[upperB] -= v[bot:bot+nupperB]
        bot += nupperB;  p[rangeB] -= v[bot:bot+nrangeB]

        # Insert contribution of slacks on the bound constraints
        #bot = om+nrangeC; p[self.tLL] -= v[bot:bot+nlowerB]
        #bot += nlowerB;  p[self.tLR] -= v[bot:bot+nrangeB]
        #bot += nrangeB;  p[self.tUU] -= v[bot:bot+nupperB]
        #bot += nupperB;  p[self.tUR] -= v[bot:bot+nrangeB]

        return p
    


    def A(self):
        """
        Return the constraint matrix if the problem is a linear program. See the
        documentation of :meth:`jac` for more information.
        """
        return self.jac([0])

    def hprod(self, x, y, v, **kwargs):
            on = self.original_n ; om = self.original_m
            Hv = np.zeros(self.n)
            Hv[:on] = self.model.hprod(x[:on], y[:om], v[:on], **kwargs)
            return Hv
 
    #def ghivprod(self, g, v, **kwargs):
            #on = self.original_n
            #return self.nlp.ghivprod(g[:on], v[:on], **kwargs)

    def jac(self, x, **kwargs):
        return SimpleLinearOperator(self.n, self.m, symmetric=False,
                         matvec=lambda u: self.jprod(x,u,**kwargs),
                         matvec_transp=lambda u: self.jtprod(x,u,**kwargs))
    
    def hess(self, x, z, **kwargs):
        return SimpleLinearOperator(self.n, self.n, symmetric=True,
                         matvec=lambda u: self.hprod(x,z,u,**kwargs))

def FormEntireMatrix(on,om,Jop):
    J = np.zeros([om,on])
    for i in range(0,on):
        v = np.zeros(on)
        v[i] = 1.
        J[:,i] = Jop * v
    return J

if __name__ == '__main__':
    from lsqmodel import LSQModel
    #from lsq import lsq
    #from mfnlp import *
    from Others.lsq_testproblem import *
    import numpy as np
    lsqpr  =  exampleliop()
    
    slack = SlackFrameworkNLP( lsqpr ) 
    j = slack.jac(slack.x0)
    A = slack.A()
    print as_llmat(FormEntireMatrix(j.shape[1],j.shape[0],j))
    print as_llmat(FormEntireMatrix(A.shape[1],A.shape[0],A))
    #print p.shape
    #print as_llmat(FormEntireMatrix(p.shape[1],p.shape[0],p))
    #print slack.InitializeSlacks()
    #print slack.display_basic_info()
    #print ls2.display_basic_info()

