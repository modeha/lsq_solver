function [ntiter,opt,gap,time,x,history] = l1ls_test_problem(n,m,test_problem)
if n<m 
    error('n must be grater than m!')
end
if strcmp(test_problem,'random')
   rand('twister', 1919);
    A = rand(n,m);
    At = A';
    I = eye(m);
    epsilon = 0.1;
    d = epsilon*I(:,1);%ones(m,1);
    y = sqrt(2)/2*(A*d);
    lambda = 1.0000e-19; % regularization parameter
    rel_tol = 1.0000e-06; % relative target duality gap
    tmp = n;
    n = m;
    m = tmp;
    tic;
    [ntiter,opt,gap,x,history] =l1_ls_nonneg(A,At,m,n,y,lambda,rel_tol);
    time = toc;
    %[~,~,y,~,~]= partialDCT_generate(n,m);
elseif strcmp(test_problem,'partial_DCT_norm1')
    %Example l1ls_partialdct(1024,128)
    [A,At,y,rel_tol,lambda,y1]= partialDCT_generate(n,m);
    tic;
    [ntiter,opt,gap,x,history] =l1_ls_nonneg(A,At,n,m,y,lambda,rel_tol);
    time = toc;
%run the l1-regularized least squares solver
end
%  tic;
%  [ntiter,opt,gap,x] = l1_ls(A,At,m,n,y,lambda,rel_tol);
%  time = toc;

 %[ntiter,opt,gap,x,history] =l1_ls_nonneg(A,At,m,n,y,lambda,rel_tol);
 %[ntiter,opt,gap,x,history] =l1_ls_nonneg(A,At,n,m,y,lambda,rel_tol);
 
   fprintf('\n IterN =%d Objective=%-9g realTol =%-9g time =%-5f pcg_iters=%d \n',...
      ntiter,opt,gap, time, sum(history(5,:))) 
  
end
