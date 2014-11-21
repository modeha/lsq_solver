function [x,IterN,realTol,objtrue,time] = PDCOTest(m,n,k)

%Example of use:
% PDCOTest(128,1024,1)
% PDCOTest(128,1024,2)
[A,~,y,~,lambda]= partialDCT_generate( m,n );
 %[A,y]= randPDCO(m,n);
 
 
 
  options = pdcoSet;
  options.StepTol = 0.1;
  options.FeaTol       =  1e-6;
  options.OptTol       =  1e-6;
  options.MaxIter = 400;
  options.mu0       = 1e-0;  % An absolute value
  options.LSMRatol1 = 1e-6;  % For LPs, LSQR must solve quite accurately
  options.wait      = 0;     % Allow options to be reviewed before solve
  options.Print     = 1;
  options.Method    = 3;  % Will change to 1 or 3
                          %1=Cholesky  2=QR  3=LSMR  4=MINRES  21=SQD



  x0    = ones(n,1);     % Initial x
  y0    = zeros(m,1);        % Initial y
  xsize = norm(x0,inf);                 % Estimate of norm(x,inf) at solution
  Anorm = normest(A, 1.0e-3);
  zsize = max(normest(A*A')+sqrt(n)*Anorm,1);% Estimate of norm(z,inf) at solution
  %zsize = max(m,n)/min(m,n);%.93
  %z0    = zsize*ones(n,1);         % Initial z

  c       = zeros(n,1);
  b       = y;
  em      = ones(m,1);
  zn      = zeros(n,1);
  bigbnd  = 1e+30;
  bl      = zn;         % x > 0
  bu      = bigbnd;%10*en;      % x < 10 where en      = ones(n,1);
%   bl(1)   = - bigbnd;   % Test "free variable" (no bounds)
%   bu(1)   =   bigbnd;
  nc = m;
  gamma   = 1;       % Primal regularization.
  delta   = 1;       % 1e-3 or 1e-4 for LP;  1 for Least squares.
  d1      = gamma;      % Scalar for this test.
  d2      = em;         % D2 = I
  d2(m-nc+1:m) = delta; % Satisfy last nc rows more accurately.
  

  lsmr_solver = k;
  % lsmr_solver = 1 use PDCO lsmr, lsmr_solver = 2 use lsmr_spot 
  [x,IterN,realTol,objtrue,time] = ...
  pdco(c,A,b,bl,bu,d1,d2,options,x0,y0,z0,xsize,zsize,lsmr_solver);
  %realTol = max([Pinf,Dinf,Cinf,atol])
  fprintf('\n zsize=%-9g realTol =%-9g time =%-5f Solver =%-5.0f\n',zsize, realTol, time,lsmr_solver);
  
  
%dictInfo(x,b,IterN,realTol,objtrue,time,Name);
end