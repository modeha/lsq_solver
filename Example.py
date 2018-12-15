
from __future__ import division
from sympy import *
from scipy.optimize import fsolve, fmin,golden,basinhopping

#yprime = y.diff(x)


#F = R.T*R

#phi = F[0,0]

#
#(x1+alpha*p1)**2+(10*(x1+alpha*p1)/(x1+alpha*p1+0.1)+2*(x2+alpha*p2)**2)**2
#pprint (F[0,0])  

def Res(x,y):
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    R = Matrix([[x1],[10*x1/(x1+0.1)+2*x2*x2]]) 
    return Matrix([[x1],[10*x1/(x1+0.1)+2*x2*x2]]).subs(x1,x).subs(x2,y)

def Jacob(x,y):
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    R = Matrix([[x1],[10*x1/(x1+0.1)+2*x2*x2]])         
    R.jacobian(Matrix([x1,x2]))
    return R.jacobian(Matrix([x1,x2])).subs(x1,x).subs(x2,y)

def phi():
    alpha = Symbol('alpha') 
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    p1 = Symbol('p1')
    p2 = Symbol('p2')    
    return (x1**2+(10*x1/(x1+0.1)+2*x2**2)**2).subs(x1,x1+alpha*p1).subs(x2,x2+alpha*p2)

def linear_system(J,R):
    return J.LUsolve(-R)

def Diff(x, y, p1, p2):
    a=2*p1*(alpha*p1 + x) + (10*(alpha*p1 + x)/(alpha*p1 + x + 0.1) +\
                             2*(alpha*p2 + y)**2)*(-20*p1*(alpha*p1 + x)/(alpha*p1 + x + 0.1)**2 \
                                                   + 20*p1/(alpha*p1 + x + 0.1) + 8*p2*(alpha*p2 + y))

    #(diff((x1**2+(10*x1/(x1+0.1)+2*x2**2)**2).subs(x1,x1+alpha*p1).subs(x2,x2+alpha*p2),alpha))
    return a
def f(x,y,p_1,p_2,a):
    alpha = Symbol('alpha') 
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    p1 = Symbol('p1')
    p2 = Symbol('p2')    
    g = phi().subs(x1,x).subs(x2,y).subs(p1,p_1).subs(p2,p_2)
    return g.subs(alpha,a)

def line_search(x1,x2,n):
    x = Symbol('x'); y = Symbol('y')
    p1 = Symbol('p1'); p2 = Symbol('p2')
    alpha = Symbol('alpha') 
    x_k = Matrix([[x1], [x2]])
    for i in range(n):

        R =  Res(x_k[0],x_k[1])
        J =  Jacob(x_k[0],x_k[1])
        p =  linear_system(J,R)
        #a = Diff(x_k[0],x_k[1],p[0],p[1])
        #pprint (a)
        
        #y = solve(a,alpha)
        #print y
        #total = []
        #for root in y:
            #if "I" not in str(root):
                #total.append(root.evalf())
        #alpha = min(total)
        alpha = golden(lambda z: f(x_k[0],x_k[1],p[0],p[1],z))
        #alpha = basinhopping(lambda z: f(x_k[0],x_k[1],p[0],p[1],z),0)
        x = x_k + p*alpha
        f_ = R.T*R
        #print 'x_k[0],x_k[1],f[0],alpha'
        print  i,x_k[0],x_k[1],f_[0],alpha
        x_k = x
    return       
    

if __name__ == "__main__":
    x = Symbol('x'); y = Symbol('y',)
    p1 = Symbol('p1'); p2 = Symbol('p2')
    alpha = Symbol('alpha')    
    #A =  Res(x,y).subs(y,2).subs(x,1)
    #pprint (A)
    #B =  J(Symbol('x'), Symbol('y'))
    #pprint(A)
    #pprint(B)
    #C = phi(x, y, p1, p2)
    #pprint (C.subs(y,-1).subs(x,2).subs(p1,.2).subs(p2,.1))

    #pprint (Diff(x,y,p1,p2).subs(y,-1).subs(x,2).subs(p1,.2).subs(p2,.1))
    #x1 = 3; x2 = 1
    #R =  Res(x,y).subs(x,x1).subs(y,x2)
    #J =  Jacob(x,y).subs(x,x1).subs(y,x2)
    #p =  linear_system(J,R)   
    #print
    #pprint (R)
    #pprint (J)
    #pprint (p)
    #print
    #print
    #y = solve(Diff(x,y,p1,p2).subs(y,-1).subs(x,2).subs(p1,.2).subs(p2,.1),alpha)
    #total = []
    #for root in y:
        #if "I" not in str(root):
            #total.append(root.evalf())
    #print min(total)
    
    line_search(3,1,6000)
    




    