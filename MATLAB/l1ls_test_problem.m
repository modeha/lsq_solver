function [ntiter,opt,gap,time,x,history,status] = l1ls_test_problem(n,m,test_problem,epsilon)
if n<m 
    error('n must be grater than m!')
end
if strcmp(test_problem,'PRNG')
    rand('twister', 1919);
    A = sqrt(2)/2*rand(n,m);
    At = A';
    d = parameterD(m,epsilon);
    %d(2,1) = 1;
    y = sqrt(2)/2*(A*d);
    lambda = 1.0e-06; % regularization parameter
    rel_tol = 1.0e-06; % relative target duality gap
    tmp = n;
    n = m;
    m = tmp;
    tic;
    [ntiter,opt,gap,x,history,status] =l1_ls_nonneg(A,At,m,n,y,lambda,rel_tol);
    time = toc;
    %[~,~,y,~,~]= partialDCT_generate(n,m);
elseif strcmp(test_problem,'DCT')
    % [A,At,y,rel_tol,lambda,y1]= partialDCT_generate( n,m ) 
    % Normalize cols of A
    % Matlab script for solving the sparse signal recovery problem
    % using the object-oriented programming feature of Matlab.
    % The three m files in ./@partialDCT/ implement the partial
    %DCT class with the multiplication and transpose operators overloaded.
    A = partialDCT(n,m); % A
    At = A'; % transpose of A
    I = eye(m);
    d = parameterD(m,epsilon);
    y = A*d;
    y = y*(sqrt(2)/2);
    lambda = 1.e-6; % regularization parameter
    rel_tol = 1.e-06; % relative target duality gap
    tic;
    [ntiter,opt,gap,x,history,status] =l1_ls_nonneg(A,At,n,m,y,lambda,rel_tol);
    time = toc;
end
fprintf('\n IterN =%d Objective=%-9g realTol =%-9g time =%-5f pcg_iters=%d \n',...
      ntiter,opt,gap, time, sum(history(5,:))) 
  
end
