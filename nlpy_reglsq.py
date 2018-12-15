from dctt import partial_DCT,Random
from lsq_testproblem import *
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
from time import gmtime, strftime
import numpy
import os
import sys
import logging
import subprocess


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

#parser.add_option("-n", "--n_dim", action="store", type="int", default=None,
        #dest="n_dim", help="Specify relative dimention")

#parser.add_option("-m", "--m_dim", action="store", type="int", default=None,
        #dest="m_dim", help="Specify relative dimention")

parser.add_option("-r", "--delta_size", action="store", type="float", default=None,
        dest="delta_size", help="Specify relative norm 1")


parser.add_option("-p", "--regpr", action="store", type="float", default=None,
        dest="regpr", help="Specify initial primal regularization parameter")

parser.add_option("-d", "--regdu", action="store", type="float", default=None,
        dest="regdu", help="Specify initial dual regularization parameter")
 
parser.add_option("-l", "--long-step", action="store_true", default=False,
        dest="longstep", help="Use long-step method")

parser.add_option("-f", "--assume-feasible", action="store_true",
        default=False, dest="assume_feasible",
        help="Deactivate infeasibility check")

parser.add_option("-V", "--verbose", action="store_true", default=False,
        dest="verbose", help="Set verbose mode")

parser.add_option("-w", "--problem", action="store", dest="problem", help="problem")


# Parse command-line options
(options, args) = parser.parse_args()
#print 'Brave man! Using example block system!'
Solver = RegLSQPInteriorPointIterativeSolver4x4   # RegLSQInteriorPointSolver4x4

opts_init = {}
if options.regpr is not None:
    opts_init['regpr'] = options.regpr
if options.regdu is not None:
    opts_init['regdu'] = options.regdu
    
opts_solve = {}
if options.maxiter is not None:
    opts_solve['itermax'] = options.maxiter
if options.tol is not None:
    opts_solve['tolerance'] = options.tol

#if options.m_dim is not None:
    #m = options.m_dim
    
#if options.n_dim is not None:
    #n = options.n_dim

if options.delta_size is not None:
    delta = options.delta_size
else:
    delta=1.0e-06
# Set printing standards for arrays.

if options.problem == 'DCT' or options.problem is None:
    problem = partial_DCT
    tol = [1e+3,1e+3,1e+3,1e-3,1e-5,1e-6,1e-7,1e-8]
    
else:
    #m=4;n=6
    #lsqp = partial_DCT(160,60,delta)#partial_
    problem = Random
    tol = [1e+3,1e+3,1e-8,1e-15,1e-15,1e-15,1e-15]
    #lsqp = exampleliop(n,m)
    
numpy.set_printoptions(precision=3, linewidth=70, threshold=10, edgeitems=2)



#if not options.verbose:
    #log.info(hdr)
    #log.info('-'*len(hdr)) 

string_results =  ''
j = 0
for i in range(len(args)/2):
    try:
	m,n = int(args[i+j]),int(args[i+j+1])
    except ValueError:
	print "Oops!  That was no valid number.  Try again..."
    j +=1
    print "Dimention of test Problem is: ","m=",m," and n=",n
    t_setup = cputime()
    
    lsqp = problem(m,n,delta)    
    t_setup = cputime() - t_setup
    
    # Pass problem to RegQP.
    regqp = Solver(lsqp,
                   verbose=options.verbose,*tol,
                   **opts_init)

    regqp.solve(PredictorCorrector=not options.longstep,
                check_infeasible=not options.assume_feasible,
                **opts_solve)

    # Display summary line.
    probname=os.path.basename('cqp')
    if not options.verbose:
        format1  = '%-4d  %9.2e'
        format1 += '  %-8.2e' * 6
        format2  = '  %-7.1e  %-4.2f  %-4.2f'
        format2 += '  %-8.2e' * 8
	#output_line = format1 % (regqp.iter, regqp.obj_value, regqp.pResid,
                                                  #regqp.dResid, regqp.cResid, regqp.rgap, regqp.qNorm,
                                                  #regqp.wNorm)
	#output_line += format2 % (regqp.mu, regqp.alpha_p, regqp.alpha_d,
                                                   #regqp.nres, regqp.regpr, regqp.regdu, regqp.rho_q,
                                                   #regqp.del_w, regqp.mins, regqp.minz, regqp.maxs)
	#log.info(output_line)
	log.info('-' * len(regqp.header))

        #sys.stdout.write(fmt % (probname, regqp.iter, regqp.obj_value,
                                #regqp.pResid, regqp.dResid, regqp.rgap,
                                #t_setup, regqp.solve_time, regqp.short_status))
        #if regqp.short_status == 'degn':
            #sys.stdout.write(' F')  # Could not regularize sufficiently.
        #sys.stdout.write('\n')
	x = regqp.x[0:n]
	x0 = 0.1*np.ones([1,n])
	x0[0] = -0.1
	norm_x0_x = norm2(x0-x)
    
	nnz = nnz_elements(x,1e-3)
	Probname = str(m)+'--'+str(n)
    
	numpy.set_printoptions(threshold=5)
	numpy.set_printoptions(threshold='nan')
	print x[0:10]
        log.info('difference between initialize point and  minimizer %6.f'%norm_x0_x)
    
	
	x = regqp.x[0:n]
	log.info('#Iterations: %-d' % regqp.iter)
	log.info('RelResidual: %7.1e' % regqp.kktResid)
	log.info('Final cost : %21.15e' % regqp.obj_value)
	log.info('Setup time : %6.2fs' % t_setup)
	log.info('Solve time : %6.2fs' % regqp.solve_time)
	print '#LSMR Iterations:',regqp.Niter_lsmr[2:-1]
	log.info('#LSMR Iterations: %-d' % sum(regqp.Niter_lsmr[2:-1]))
	log.info('%-d  %7.1e  %7.1e  %7.1e' % (regqp.iter,regqp.obj_value,regqp.kktResid,regqp.solve_time))
	#if not os.path.isfile('/Users/Mohsen/Documents/nlpy_mohsen/lsq/Results.txt'):
		#f =open ('/Users/Mohsen/Documents/nlpy_mohsen/lsq/LSQ_Results.txt','a+')
		#path =6*' '+'Name'+ 23*' '+'Iter'+13*' '+'Cost'+8*' '+'RelResidual'\
		     #+4*' '+'Time'+4*' '
		#f.write(path)
		#f.write('\n')
		#f.write('-'*(86+27))
		#f.write('\n')
	f =open ('/Users/Mohsen/Documents/nlpy_mohsen/lsq/LSQ_Results.txt','a+')
	#
	#f.close()
	#f =open ('Results.txt','a+')
	#f.write('\n')
	f.write('%-15s&' % Probname)
	f.write('%-5d&' % regqp.iter)
	f.write('%-15.1e&' % regqp.obj_value)
	f.write('%-15.1e&' % regqp.kktResid)
	f.write('%-15.2f&' % regqp.solve_time)
	f.write('%-5d\\\ ' % sum(regqp.Niter_lsmr[2:-1]))
	#f.write('%-5d       ' % nnz)
	#f.write('%3e   ' % norm_x0_x)
	f.write('\n')
	f.close()

	string_results +=  '%-15s\&' % Probname
	string_results += '%-5d\&' % regqp.iter
	string_results += '%-15.1e\&' % regqp.obj_value
	string_results += '%-15.1e\&' % regqp.kktResid
	string_results += '%-15.2f\&' % regqp.solve_time
	string_results +=  '%-5d\\\\' % sum(regqp.Niter_lsmr)
	string_results += '\\\\'
t = strftime("%d-%H-%M-%S", gmtime())
os.system("sed -e s/TABLE_CONTENTS/'"+string_results+"'/g template.tex > "+t+"result_table.tex")
 
#f = os.getcwd()+'/'
#p = "/usr/texbin/pdflatex  "
#print [p + f+'result_table.tex']
#subprocess.Popen([p + f+'result_table.tex'], shell=True)
##sts = os.waitpid(p.pid, 0)[1] 
#h = 'open '+f+'result_table.pdf'
#subprocess.Popen([h],shell=True)    
