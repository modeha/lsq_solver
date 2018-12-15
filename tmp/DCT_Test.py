from pykrylov.linop import LinearOperator
import logging
import sys
from dctt import *
    

# Create root logger.
log = logging.getLogger('blk-ops')
log.setLevel(logging.DEBUG)
fmt = logging.Formatter('%(name)-8s %(levelname)-8s %(message)s')
hndlr = logging.StreamHandler(sys.stdout)
hndlr.setFormatter(fmt)
log.addHandler(hndlr)

A = LinearOperator(nargin=6, nargout=6,
                   matvec=lambda v: 2*v, symmetric=True)

C = LinearOperator(nargin=3, nargout=6, matvec=lambda v: v[:6],
                   matvec_transp=lambda v: np.concatenate((v, np.zeros(1))))
D = LinearOperator(nargin=4, nargout=3, matvec=lambda v: v[:3],
                   matvec_transp=lambda v: np.concatenate((v, np.zeros(2))))
m=4;n=6
#m=3;n=4
B = LinearOperator(nargin=m, nargout=n, matvec=lambda v: Ax(n,m,v),
                       matvec_transp=lambda v: ATx(m,n,v))

B = LinearOperator(nargin=m, nargout=n, matvec=lambda v: v[:n],
                   matvec_transp=lambda v: np.concatenate((v, np.zeros(1))))



print "A",A.shape, A.T.shape
print "B",B.shape, B.T.shape
print "C",C.shape, C.T.shape
print "D",D.shape, D.T.shape


# Build [A  B].
#K1 = BlockLinearOperator([[A, B]], logger=log)


# Build [A  B]
#       [C  D]
K2 = BlockLinearOperator([[A, B], [C.T, D]], logger=log)

x = np.ones(K2.shape[1])
K2x = K2 * x
print 'K2*e = ', K2x

y = np.ones(K2.shape[0])
K2Ty = K2.T * y
print 'K2.T*e = ', K2Ty

# Build [A  B]
#       [B' E]
K3 = BlockLinearOperator([[A, B], [E]], symmetric=True, logger=log)
y = np.ones(K3.shape[0])
K3y = K3 * y
print 'K3*e = ', K3y
K3Ty = K3.T * y
print 'K3.T*e = ', K3T