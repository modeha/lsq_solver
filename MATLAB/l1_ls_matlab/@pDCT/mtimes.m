function res = mtimes(A,x)
m = A.m;
n = A.n;
if A.adjoint == 0 %A*x
    x1 = x(1:n);
    x_1 = dct(x1);
    x2 = x(n+1:2*n);
    x3 = x(2*n+1:2*n+m);
    x4 = x(2*n+m+1:3*n+m);
    x5 = x(3*n+m+1:end);
   res =[x_1-x3;-x1-x2+x4;x1-x2+x5];
   res = res(A.J);
else %At*x
    % dimention of x and J must be the same
    z = zeros(A.n,1);
    z(A.J) = x
    res = idct(z);
%     z = zeros(A.n,1);
%     x1 = x(1:m);
%     z(A.J) = x1;
%     res1 = idct(z);    
%     x2 = x(m+1:n+m);
%     x3 = x(m+n+1:end);
%     res = [res1-x2+x3;-x2-x3;x1;x2;x3];
end
