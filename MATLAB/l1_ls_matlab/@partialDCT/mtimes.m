function Ax = mtimes(A,x)
%A(m,n)
size_x = size(x);
n = A.n;
m = A.m;
k = abs(m-n);

if A.adjoint == 0 %A*x
%     if size_x(1) ~= A.n
%     error('Inner matrix dimensions must agree.Size x should be the same as column of A')
%     end
    if n < m 
    Ax = vertcat(sqrt(2)/2*dct(x),zeros(k,1));
    else 
        Ax = sqrt(2)/2*dct(x(1:m));
    end
    %res = res(A.J);
else %At*x
    if size_x(1) ~= A.m
    error('Inner matrix dimensions must agree.Size x should be the same as row of A')
    end
    if m < n
        Ax = vertcat(sqrt(2)/2*idct(x), zeros(k,1));
    else
        
        Ax = sqrt(2)/2*idct(x(1:n));
    end

%     z = zeros(A.n,1);
%     z(A.J) = x;
   
end
