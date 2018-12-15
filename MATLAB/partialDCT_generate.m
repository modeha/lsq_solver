function [A,At,y,rel_tol,lambda,y1]= partialDCT_generate( n,m )                               % Normalize cols of A
% Matlab script for solving the sparse signal recovery problem
 % using the object-oriented programming feature of Matlab.
 % The three m files in ./@partialDCT/ implement the partial DCT class
 % with the multiplication and transpose operators overloaded.
%  if n<m
%      error('n must be grater than m')
%  end

 A = partialDCT(n,m); % A
 At = A'; % transpose of A
 I = eye(m);
 %I(1,1) = 0.3;
 %I(end,1) = -1;
 %y = A*I(:,1);
 epsilon = 1;
 d = epsilon*ones(m,1);
 d(1,1) = 0.1;
 y = A*d;
 y = y*(sqrt(2)/2);
 y1 = y(1);

 lambda = 1.0000e-06; % regularization parameter
 rel_tol = 1.0000e-06; % relative target duality gap
end