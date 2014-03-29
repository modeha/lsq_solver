% keyboard
function [x,IterN,realTol,objtrue,time,Name,b]= PDCO_partialDCT( m,n )
% Private function below
% D  = sum(A,1);   D(find(D==0)) = 1;
% D  = sparse( 1:n, 1:n, 1./D, n, n );
% A  = A*D;                                % Normalize cols of A



%rand('state',0);randn('state',0); %initialize (for reproducibility)

 n = 1024; % signal dimension
 m = 128; % number of measurements
 nc = 10;
nc = m;
m2n = [m,n];
x = textscan(num2str(m2n),'%s');
Name = strcat(char(x{1}{1}),char('101'),char(x{1}{2}));
Name = str2num(Name)


 J = randperm(n); J = J(1:m); % m randomly chosen indices

 % generate the m*n partial DCT matrix whose m rows are
 % the rows of the n*n DCT matrix at the indices specified by J
 % see files at @partialDCT/
 A = partialDCT(n,m,J); % A
 At = A'; % transpose of A

 % spiky signal generation
 T = 10; % number of spikes
 x0 = zeros(n,1);
 q = randperm(n);
 x0(q(1:T)) = sign(randn(T,1));

 % noisy observations
 sigma = 0.01; % noise standard deviation
 y = A*x0 + sigma*randn(m,1);

 lambda = 0.01; % regularization parameter
 rel_tol = 0.0001; % relative target duality gap

%run the l1-regularized least squares solver
% tic;
% [iter,primobj,gap,x] = l1_ls(A,At,m,n,y,lambda,rel_tol);
% time = toc;
%dictInfo(x,y,iter,gap,primobj,time,Name);
 
 
 
  options = pdcoSet;

  x0    = 0.5*ones(n,1);     % Initial x
  y0    = zeros(m,1);        % Initial y
  z0    = ones(n,1);         % Initial z
  xsize = 1;                 % Estimate of norm(x,inf) at solution
  zsize = 1;                 % Estimate of norm(z,inf) at solution

  options.mu0       = 1e-0;  % An absolute value
  options.LSMRatol1 = 1e-6;  % For LPs, LSQR must solve quite accurately
  options.wait      = 0;     % Allow options to be reviewed before solve
  options.Method    = 3;  % Will change to 1 or 3
                          %1=Cholesky  2=QR  3=LSMR  4=MINRES  21=SQD

  
  c       = zeros(n,1);
  b       = y;
  em      = ones(m,1);
  en      = ones(n,1);
  zn      = zeros(n,1);
  bigbnd  = 1e+30;
  bl      = zn;         % x > 0
  bu      = bigbnd;%10*en;      % x < 10
%   bl(1)   = - bigbnd;   % Test "free variable" (no bounds)
%   bu(1)   =   bigbnd;

  gamma   = 1e-3;       % Primal regularization.
  delta   = 1e-3;       % 1e-3 or 1e-4 for LP;  1 for Least squares.
  d1      = gamma;      % Scalar for this test.
  d2      = em;         % D2 = I
  d2(m-nc+1:m) = delta; % Satisfy last nc rows more accurately.
  
%   rand('state',0);
%   density = 0.25;
%   rc      = 1e-1;
  %A       = sprand(m,n,density,rc);
  %x       = en;
  %b       = full(A*x) + d2;  % Plausible b, as if dual vector y = ones(m,1).
  %c       = rand(n,1);
   options
 
  %operator  =  isa(A,'partialDCT')
%   [x,IterN,realTol,objtrue,time] = ...
    pdco(c,A,b,bl,bu,d1,d2,options,x0,y0,z0,xsize,zsize);
%dictInfo(x,b,IterN,realTol,objtrue,time,Name);
end