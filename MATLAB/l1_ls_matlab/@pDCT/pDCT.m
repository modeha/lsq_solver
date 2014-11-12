function  res = pDCT(n,m,J)
if n<m 
    error('n must be grater than m!')
end
res.adjoint = 0;
res.n = n;
res.m = m;
res.J = J(1:m);
res.M = m+2*n;
res.N = 4*n+m;

% m = 2*M-N;
% n = (N-M)/2;
% 
%res.nnz = nnz(J(1:m));
% Register this variable as a partialDCT class
res = class(res,'pDCT');
