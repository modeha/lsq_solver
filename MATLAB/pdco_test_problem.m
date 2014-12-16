function [x,IterN,realTol,objtrue,time,Niter_lsmr]= ...
    pdco_test_problem(n,m,test_problem,epsilon,gamma)

if n<m 
    error('n must be grater than m!')
end
delta = 1.0000e-6; % regularization parameter

if strcmp(test_problem,'DCT')
    obj = @(mode,M,N,x) AXfunc_l1_ls(mode,M,N,x);
     M = m+2*n;
     N = 4*n+m;
     c = vertcat(zeros(n,1),delta*ones(n,1),zeros(m,1),zeros(2*n,1));
     A = partialDCT(m,n); % A
     At = A'; % transpose of A
     d = parameterD(m,epsilon);
     y = A*d;
     y = y*(sqrt(2)/2);
%     lambda = 1.0000e-6; % regularization parameter
%      rel_tol = 1.0000e-06; % relative target duality gap
     b   = vertcat(y,zeros(2*n,1));
     tol = [1e+3,1e+3,1e+3,1e-3,1e-5,1e-6,1e-7,1e-8];

elseif strcmp(test_problem,'PRNG')
    [obj,y] = random_test_PDCO(m,n,epsilon);
     M = m+2*n;
     N = 4*n+m;
     c = vertcat(zeros(n,1),delta*ones(n,1),zeros(m,1),zeros(2*n,1));
     b   = vertcat(y,zeros(2*n,1)); 
    tol = [1e+3,1e+3,1e-8,1e-15,1e-15,1e-15,1e-15,1e-15];

% elseif strcmp(test_problem,'least_squares')
%      M = m;
%      N = n; 
%      c = zeros(n,1);
%      obj = rand(n,m);
%      I = eye(m);
%      I(1,1) = -2;
%      y = obj*I(:,1);
%      b   = y;
else
     error('!')
end

%rel_tol = 1.0000e-06; % relative target duality gap 

 
 x0    = 0.5*ones(N,1);     % Initial x
 y0    = zeros(M,1);        % Initial y
 z0    = ones(N,1);         % Initial z
 xsize = norm(x0,Inf);
  % Estimate of norm(x,inf) at solution
  % Estimate of norm(z,inf) at solution
  A = obj;
  %Anorm = normest(A, 1.0e-3);
  %zsize = max(normest(A*A')+sqrt(n)*Anorm,1)
  zsize = max(m,n)/min(m,n);
  


  
  options = pdcoSet;
  options.StepTol = .1;
  options.FeaTol       =  1e-6;
  options.OptTol       =  1e-6;
  options.MaxIter = 900;
  options.mu0       = 1e-0;  % An absolute value
  options.tol = []
  %options.tol = 100*tol;
  options.LSMRatol1 = 1e-12;  % For LPs, LSQR must solve quite accurately
  options.wait      = 0;     % Allow options to be reviewed before solve
  options.Print     = 1;
  options.Method    = 3;  % Will change to 1 or 3
                          %1=Cholesky  2=QR  3=LSMR  4=MINRES  21=SQD
  %options.LSMRatol1    = 1e-12;
  options.LSMRatol2    = 1e-12;  % 
  %options.LSMRconlim   = 1e+8;  % 
 
  em      = ones(M,1);
  zn      = zeros(n,1);
  on      = ones(n,1);
  om      = ones(m,1);
  bigbnd  = 1e+30;
  inf = bigbnd*on;
  infm = bigbnd*om;
  
  
  bl      = [zn;-inf;-infm;zn;zn]; 
  bu      = [inf;inf;infm;inf;inf];

  %gamma   = .19;%  'PRNG' .19 for DCT 19     % Primal regularization.
  delta   = 1;       % 1e-3 or 1e-4 for LP;  1 for Least squares.
  d1      = gamma;      % Scalar for this test.
  d2      = em;         % D2 = I
  nc = N/10;
  d2(M-nc+1:M) = delta; % Satisfy last nc rows more accurately.
 lsmr_solver = 1;
%[Q,~,~,~,~]= partialDCT_generate(n,m);
[x,IterN,realTol,objtrue,time,Niter_lsmr] = ...
    pdco(c,obj,b,bl,bu,d1,d2,options,x0,y0,z0,xsize,zsize,lsmr_solver);
  fprintf('\n IterN =%d Objective=%-2.2e realTol =%-2.2e time =%-.2f Niter_lsmr = %dSolver =%-1.0f zsize=%-5g\n',...
      IterN,objtrue,realTol, time,sum(Niter_lsmr), lsmr_solver,zsize);

end

function [A,y] = random_test_PDCO(m,n,epsilon)
 M = m+2*n;
 N = 4*n+m;
 rand('twister', 1919);
 Q = sqrt(2)/2*rand(m,n);
 d = parameterD(n,epsilon);
 %d(2,1) = 1;
 y = (sqrt(2)/2)*(Q*d);

 I = eye(n);
 Im = eye(m);
 A = zeros(M,N);

 A(1:m,1:n)             = Q;
 A(m+1:m+n,1:n)         = -I;
 A(m+n+1:end,1:n)       = I;
 
 A(m+1:m+n,n+1:2*n)     = -I;
 A(m+n+1:end,n+1:2*n)     = -I;
 
 A(1:m,2*n+1:2*n+m)         = -Im;
 
 A(m+1:m+n,2*n+m+1:3*n+m)   = I; 
 A(m+n+1:end,3*n+m+1:end) = I;
end

function [y] = AXfunc_l1_ls(mode,M,N,x)
%
m = 2*M-N;
n = (N-M)/2;
Q = partialDCT(m,n);

if mode ==1
    x1 = x(1:n);
    x2 = x(n+1:2*n);
    x3 = x(2*n+1:2*n+m);
    x4 = x(2*n+m+1:3*n+m);
    x5 = x(3*n+m+1:end);
    y =[Q*x1-x3;-x1-x2+x4;x1-x2+x5];
else
    x1 = x(1:m);
    x2 = x(m+1:n+m);
    x3 = x(m+n+1:end);
    y = [Q'*x1-x2+x3;-x2-x3;x1;x2;x3];
end
end
