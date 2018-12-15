% Example test program to solve
% [ M   A'] [r]   [b]
% [ A  -N ] [x] = [0]
% by applying LSMR to min ||A'x - b||^2_inv(M) + ||x||^2_N.
%
% 03 Feb 2014: First version: Dominique Orban <dominique.orban@gerad.ca>
% This test requires http://math.nist.gov/MatrixMarket/mmio/matlab/mmread.m

problem = 'mosarqp1';
M      = mmread(['/Users/Mohsen/Documents/MATLAB/spot-sol/data/', problem, '/M.mtx']);
opts.M = opChol(M);
N      = mmread(['/Users/Mohsen/Documents/MATLAB/spot-sol/data/', problem, '/N.mtx']);
opts.N = opChol(N);

% Set up least-squares problem.
A  = opMatrix(mmread(['/Users/Mohsen/Documents/MATLAB/spot-sol/data/', problem, '/A.mtx']));
r  = ones(size(M,1),1);  % Optimal residual.
xe = opts.N * A * r;     % Exact solution.
b  = M * r + A' * xe;

% Set options.
opts.show = true;
opts.atol = 0;
opts.btol = 0;     % Force a stop on the energy norm of the direct error.
opts.sqd  = true;  % Indicates that N acts as regularization.

% Solve with LSMR.
[x, flags, stats] = lsmr_spot(A', b, opts);
flags
stats
semilogy(stats.err_lbnds, 'b-'); hold on;   % Monotonically decreasing.
semilogy(stats.Aresvec, 'r-');              % Also monotonic in LSMR.
title('History');
legend('Error lower bound', 'Optimality residual');

% Compute relative error in energy norm.
NE1 = N + A * opts.M * A';        % Normal equations of the first kind.
W   = NE1 * opts.N * NE1;
err = sqrt((x - xe)' * W * (x - xe));
fprintf('Absolute error in energy norm = %7.1e\n', err);
fprintf('Relative error in energy norm = %7.1e\n', err / sqrt(xe' * W * xe));
