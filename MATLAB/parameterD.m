function d = parameterD(m,epsilon)
rand('twister', 1919);
%d =  epsilon*ones(m,1);
% a = 0; b = .05;
% d = a + (b-a).*rand([m,1])
d = epsilon*rand([m,1]);
% I = eye(m);
% d = epsilon*I(:,1);
d(1,1)=  -0.1;
%d(2,1) = 1;
end