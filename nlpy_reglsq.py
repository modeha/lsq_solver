from lsq_testproblem import npz_to_lsqobj, exampleliop
from lsqmodel import LSQModel, LSQRModel
from nlpy import __version__
from slack_nlp import SlackFrameworkNLP as SlackFramework
from lsqp import RegLSQPInteriorPointSolver
from lsqp import RegLSQPInteriorPointSolver3x3
from lsqp import RegLSQPInteriorPointSolver4x4
from lsqp import RegLSQPInteriorPointIterativeSolver4x4
from nlpy.tools.norms import norm2
from nlpy.tools.timing import cputime
from optparse import OptionParser
import numpy
import os
import sys
import logging
from lsq_testproblem import nnz_elements

# Create root logger.
log = logging.getLogger('cqp')
log.setLevel(logging.INFO)
fmt = logging.Formatter('%(name)-10s %(levelname)-8s %(message)s')
hndlr = logging.StreamHandler(sys.stdout)
hndlr.setFormatter(fmt)
log.addHandler(hndlr)

# Configure the solver logger.
sublogger = logging.getLogger('cqp.solver')
sublogger.setLevel(logging.INFO)
sublogger.addHandler(hndlr)
sublogger.propagate = False

usage_msg = """%prog [options] problem1 [... problemN]
where problem1 through problemN represent convex quadratic programs."""

# Define formats for output table.
hdrfmt = '%-15s  %5s  %15s  %7s  %7s  %7s  %6s  %6s  %4s'
hdr = hdrfmt % ('Name', 'Iter', 'Objective', 'pResid', 'dResid',
                'Gap', 'Setup', 'Solve', 'Stat')
fmt = '%-15s  %5d  %15.8e  %7.1e  %7.1e  %7.1e  %6.2f  %6.2f  %4s'

# Define allowed command-line options
parser = OptionParser(usage=usage_msg, version='%prog version ' + __version__)

# File name options
parser.add_option("-i", "--iter", action="store", type="int", default=None,
        dest="maxiter",  help="Specify maximum number of iterations")
parser.add_option("-t", "--tol", action="store", type="float", default=None,
        dest="tol", help="Specify relative stopping tolerance")
parser.add_option("-p", "--regpr", action="store", type="float", default=None,
        dest="regpr", help="Specify initial primal regularization parameter")
parser.add_option("-d", "--regdu", action="store", type="float", default=None,
        dest="regdu", help="Specify initial dual regularization parameter")
parser.add_option("-S", "--no-scale", action="store_true",
        dest="no_scale", default=False, help="Turn off problem scaling")
parser.add_option("-3", "--3x3", action="store_true",
        dest="sys5x5", default=False, help="Use 5x5 block linear system")
parser.add_option("-4", "--4x4", action="store_true",
        dest="sys4x4", default=False, help="Use 4x4 block linear system")

parser.add_option("-I", "--iterSolver", action="store",
        dest="iterSolve", default=None, help="Use 4x4 block linear system")

parser.add_option("-l", "--long-step", action="store_true", default=False,
        dest="longstep", help="Use long-step method")
parser.add_option("-f", "--assume-feasible", action="store_true",
        default=False, dest="assume_feasible",
        help="Deactivate infeasibility check")
parser.add_option("-V", "--verbose", action="store_true", default=False,
        dest="verbose", help="Set verbose mode")

# Parse command-line options
(options, args) = parser.parse_args()
Probname = ''
# Decide which class to instantiate.
if options.no_scale:
    Probname +='Scaled'
else:
    Probname +='UnScaled'
if options.sys5x5:
    print 'Brave man! Using 3x3 block system!'
    Probname +='_3x3'
    Solver = RegLSQPInteriorPointSolver3x3
elif options.sys4x4:
    print 'Brave man! Using 4x4 block system!'
    Probname +='_4x4'
    Solver = RegLSQPInteriorPointSolver4x4   # RegLSQInteriorPointSolver4x4
    if options.iterSolve:
	print 'Brave man! Using Iterative Solver!'
	Probname +='_Iterative'
	Solver = RegLSQPInteriorPointIterativeSolver4x4
else:
    print 'Brave man! Using example block system!'
    Probname +='_4x4'
    Solver = RegLSQPInteriorPointIterativeSolver4x4   # RegLSQInteriorPointSolver4x4

    #Probname +='_2x2'
    #Solver = RegLSQPInteriorPointSolver


opts_init = {}
if options.regpr is not None:
    opts_init['regpr'] = options.regpr
if options.regdu is not None:
    opts_init['regdu'] = options.regdu
if options.iterSolve is not None:
    opts_init['iterSolve'] = options.iterSolve
    
opts_solve = {}
if options.maxiter is not None:
    opts_solve['itermax'] = options.maxiter
if options.tol is not None:
    opts_solve['tolerance'] = options.tol

# Set printing standards for arrays.
numpy.set_printoptions(precision=3, linewidth=70, threshold=10, edgeitems=2)

multiple_problems = len(args) > 1

if not options.verbose:
    log.info(hdr)
    log.info('-'*len(hdr))
args = ['example']    
for probname in args:
    print probname
    

    t_setup = cputime()
    if options.sys4x4:
        lsqp = npz_to_lsqobj(probname[:-4],Model=LSQRModel)
	
    else :
        lsqp = exampleliop()#npz_to_lsqobj(probname[:-4],Model=LSQModel)
	
    t_setup = cputime() - t_setup

    # isqp() should be implemented in the near future.
    #if not qp.isqp():
    #    log.info('Problem %s is not a quadratic program\n' % probname)
    #    qp.close()
    #    continue

    # Pass problem to RegQP.
    regqp = Solver(lsqp,
                   scale=not options.no_scale,
                   verbose=options.verbose,
                   **opts_init)

    regqp.solve(PredictorCorrector=not options.longstep,
                check_infeasible=not options.assume_feasible,
                **opts_solve)

    # Display summary line.
    probname=os.path.basename(probname)
    if probname[-4:] == '.npz': probname = probname[:-4]

    if not options.verbose:

        sys.stdout.write(fmt % (probname, regqp.iter, regqp.obj_value,
                                regqp.pResid, regqp.dResid, regqp.rgap,
                                t_setup, regqp.solve_time, regqp.short_status))
        if regqp.short_status == 'degn':
            sys.stdout.write(' F')  # Could not regularize sufficiently.
        sys.stdout.write('\n')
    lsqp.close()

log.info('-'*len(hdr))

if not multiple_problems:
    x0,x,nnz,nnz0 = nnz_elements()
    #path = '/Users/Mohsen/Documents/nlpy_mohsen/lsq/output/MATLAB/'
    #strname = probname
    #list = strname.split('_')
    #p = int(list[0])
    #n = int(list[1])/2
    #import os,glob
    #os.chdir(path)
    #a =[]
    #for r,d,f in os.walk(path):
	    #a.append(d)
    #tmp = str(p)+'_'+str(n)
    #for i in a[0]:
	#lis = i.split('_')
	#if lis[0] !='tmp':
	    #if  tmp in str(lis[0])+'_'+str(int(lis[1])/2):
		#namef = i
    #tol = 1e-4
    #x0 = (numpy.genfromtxt(path+ namef +'/x0.txt'))
    #xmax0 = tol*numpy.linalg.norm(x0,numpy.inf)
    #nnz0 = len(x0[abs(x0)>xmax0])*1.0/len(x0)*100.
    #x = regqp.x[:n]
    x_x0 = x-x0
    #x = regqp.x[:n]
    #xmax = tol*numpy.linalg.norm(x,numpy.inf)  
    #nnz = len(x[x> xmax])*1.0/len(x)*100
    #for i in range(n):
	    #print '%9.4f' % x[i]
	 
    
    log.info('Problem name %10s'%probname[:-6])
    
    log.info('Non zero elements in original signal %6.f'%nnz0)
    log.info('Non zero elements in minimizer %6.f'%nnz)
    log.info('Defferent between the original and minimizer %6.6f'%norm2(x.T-x0))
    
    
    log.info('Final x: %s, |x| = %7.1e' % (repr(regqp.x),norm2(regqp.x)))
    log.info('Final y: %s, |y| = %7.1e' % (repr(regqp.y),norm2(regqp.y)))
    log.info('Final z: %s, |z| = %7.1e' % (repr(regqp.z),norm2(regqp.z)))

    log.info(regqp.status)
    log.info('#Iterations: %-d' % regqp.iter)
    log.info('RelResidual: %7.1e' % regqp.kktResid)
    log.info('Final cost : %21.15e' % regqp.obj_value)
    log.info('Setup time : %6.2fs' % t_setup)
    log.info('Solve time : %6.2fs' % regqp.solve_time)
    str_len = len(Probname)
    Probname += probname[:-6]
    Probname = Probname[str_len:]+'_'+Probname[:str_len]
    if not os.path.isfile('/Users/Mohsen/Documents/nlpy_mohsen/lsq/Results.txt'):
	f =open ('/Users/Mohsen/Documents/nlpy_mohsen/lsq/Results.txt','a+')
	path =6*' '+'Name'+ 23*' '+'Iter'+13*' '+'Cost'+8*' '+'RelResidual'\
	     +4*' '+'Time'+4*' '+'Nnz_Orgigin'+4*' '+'Nnz_Minimizer'+2*' '\
	     +'norm(x_x0)'
	f.write(path)
	f.write('\n')
	f.write('-'*(86+18+27+4))
	f.write('\n')
    f =open ('/Users/Mohsen/Documents/nlpy_mohsen/lsq/Results.txt','a+')
    #
    #f.close()
    #f =open ('Results.txt','a+')
    #f.write('\n')
    f.write('%-30s      ' % Probname)
    f.write('%-5d       ' % regqp.iter)
    f.write('%3e   ' % regqp.obj_value)
    f.write('%3e   ' % regqp.kktResid)
    f.write('%1.3f   ' % regqp.solve_time)

    f.write('%3e   ' % nnz0)
    f.write('%3e   ' % nnz)
    f.write('%3e   ' % norm2(x_x0))
    f.write('\n')
    f.close()
    #for i in range(4):
	#    print '%9.4f' % regqp.x[:4][i]
    # Plot linear system statistics.
    #import matplotlib.pyplot as plt
    #fig = plt.figure()
    #ax = fig.gca()
    #ax.semilogy(regqp.lres_history)
    #ax.set_title('LS Relative Residual')
    #fig2 = plt.figure()
    #ax2 = fig2.gca()
    #ax2.semilogy(regqp.derr_history)
    #ax2.set_title('Direct Error Estimate')
    #fig3 = plt.figure()
    #ax3 = fig3.gca()
    #ax3.semilogy([cond[0] for cond in regqp.cond_history], label='K1')
    #ax3.semilogy([cond[1] for cond in regqp.cond_history], label='K2')
    #ax3.legend(loc='upper left')
    #ax3.set_title('Condition number estimates of Arioli, Demmel, Duff')
    #fig4 = plt.figure()
    #ax4 = fig4.gca()
    #ax4.semilogy([berr[0] for berr in regqp.berr_history], label='bkwrd err1')
    #ax4.semilogy([berr[1] for berr in regqp.berr_history], label='bkwrd err2')
    #ax4.legend(loc='upper left')
    #ax4.set_title('Backward Error Estimates of Arioli, Demmel, Duff')
    #fig5 = plt.figure()
    #ax5 = fig5.gca()
    #ax5.semilogy([nrm[0] for nrm in regqp.nrms_history], label='Matrix norm')
    #ax5.semilogy([nrm[1] for nrm in regqp.nrms_history], label='Solution norm')
    #ax5.legend(loc='upper left')
    #ax5.set_title('Infinity Norm Estimates')
    #plt.show()
