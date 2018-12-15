function [x, flags, stats] = lsmr_spot(A, b, opts)

%        [x, flags, stats] = lsmr_spot(A, b, opts);
%
% Spot version of LSMR developed by Dominique Orban.
% All optional input arguments go into the `opts` structure with the same name
% as in the original LSMR. All original output arguments go into the `stats`
% structure with the same name as in the original LSMR.
%
% Preconditioners M and N may be provided via `opts.M` and `opts.N` and are assumed
% to be symmetric and positive definite. If `opts.sqd` is set to `true`, we solve
% the symmetric and quasi-definite system
% [ E   A' ] [ r ]   [ b ]
% [ A  -F  ] [ x ] = [ 0 ],
% where E = inv(M) and F = inv(N).
%
% If `opts.sqd` is set to `false` (the default), we solve the symmetric and
% indefinite system
% [ E   A' ] [ r ]   [ b ]
% [ A   0  ] [ x ] = [ 0 ].
% In this case, `opts.N` can still be specified and inv(N) indicates the norm
% in which `x` should be measured.
%
% A is a linear operator.
%
% opts.M is a linear operator representing the inverse of E.
% More precisely, the product M*v should return the solution of the system
% Ey=v. By default, opts.M is the identity.
%
% 03 Feb 2014: Spot version created by Dominique Orban <dominique.orban@gerad.ca>
% Spot may be obtained from https://github.com/mpf/spot
%-----------------------------------------------------------------------

% function [x, istop, itn, normr, normAr, normA, condA, normx]...
%   = lsmr(A, b, lambda, atol, btol, conlim, itnlim, localSize, show)
%
% LSMR   Iterative solver for least-squares problems.
%   X = LSMR(A,B) solves the system of linear equations A*X=B. If the system
%   is inconsistent, it solves the least-squares problem min ||b - Ax||_2.
%   A is a rectangular matrix of dimension m-by-n, where all cases are
%   allowed: m=n, m>n, or m<n. B is a vector of length m.
%   The matrix A may be dense or sparse (usually sparse).
%
%   X = LSMR(AFUN,B) takes a function handle AFUN instead of the matrix A.
%   AFUN(X,1) takes a vector X and returns A*X. AFUN(X,2) returns A'*X.
%   AFUN can be used in all the following syntaxes.
%
%   X = LSMR(A,B,LAMBDA) solves the regularized least-squares problem
%      min ||(B) - (   A    )X||
%          ||(0)   (LAMBDA*I) ||_2
%   where LAMBDA is a scalar.  If LAMBDA is [] or 0, the system is solved
%   without regularization.
%
%   X = LSMR(A,B,LAMBDA,ATOL,BTOL) continues iterations until a certain
%   backward error estimate is smaller than some quantity depending on
%   ATOL and BTOL.  Let RES = B - A*X be the residual vector for the
%   current approximate solution X.  If A*X = B seems to be consistent,
%   LSMR terminates when NORM(RES) <= ATOL*NORM(A)*NORM(X) + BTOL*NORM(B).
%   Otherwise, LSMR terminates when NORM(A'*RES) <= ATOL*NORM(A)*NORM(RES).
%   If both tolerances are 1.0e-6 (say), the final NORM(RES) should be
%   accurate to about 6 digits. (The final X will usually have fewer
%   correct digits, depending on cond(A) and the size of LAMBDA.)
%   If ATOL or BTOL is [], a default value of 1.0e-6 will be used.
%   Ideally, they should be estimates of the relative error in the
%   entries of A and B respectively.  For example, if the entries of A
%   have 7 correct digits, set ATOL = 1e-7. This prevents the algorithm
%   from doing unnecessary work beyond the uncertainty of the input data.
%
%   X = LSMR(A,B,LAMBDA,ATOL,BTOL,CONLIM) terminates if an estimate
%   of cond(A) exceeds CONLIM. For compatible systems Ax = b,
%   conlim could be as large as 1.0e+12 (say).  For least-squares problems,
%   conlim should be less than 1.0e+8. If CONLIM is [], the default value
%   is CONLIM = 1e+8. Maximum precision can be obtained by setting
%   ATOL = BTOL = CONLIM = 0, but the number of iterations may then be
%   excessive.
%
%   X = LSMR(A,B,LAMBDA,ATOL,BTOL,CONLIM,ITNLIM) terminates if the
%   number of iterations reaches ITNLIM.  The default is ITNLIM = min(m,n).
%   For ill-conditioned systems, a larger value of ITNLIM may be needed.
%
%   X = LSMR(A,B,LAMBDA,ATOL,BTOL,CONLIM,ITNLIM,LOCALSIZE) runs LSMR
%   with reorthogonalization on the last LOCALSIZE v_k's (v-vectors
%   generated by the Golub-Kahan bidiagonalization). LOCALSIZE = 0 or []
%   runs LSMR without reorthogonalization. LOCALSIZE = Inf specifies
%   full reorthogonalization of the v_k's.  Reorthogonalizing only u_k or
%   both u_k and v_k are not an option here, because reorthogonalizing all
%   v_k's makes the u_k's close to orthogonal. Details are given in the
%   submitted SIAM paper.
%
%   X = LSMR(A,B,LAMBDA,ATOL,BTOL,CONLIM,ITNLIM,LOCALSIZE,SHOW) prints an
%   iteration log if SHOW=true. The default value is SHOW=false.
%
%   [X,ISTOP] = LSMR(A,B,...) gives the reason for termination.
%      ISTOP  = 0 means X=0 is a solution.
%             = 1 means X is an approximate solution to A*X = B,
%                 according to ATOL and BTOL.
%             = 2 means X approximately solves the least-squares problem
%                 according to ATOL.
%             = 3 means COND(A) seems to be greater than CONLIM.
%             = 4 is the same as 1 with ATOL = BTOL = EPS.
%             = 5 is the same as 2 with ATOL = EPS.
%             = 6 is the same as 3 with CONLIM = 1/EPS.
%             = 7 means ITN reached ITNLIM before the other stopping
%                 conditions were satisfied.
%
%   [X,ISTOP,ITN] = LSMR(A,B,...) gives ITN = the number of LSMR iterations.
%
%   [X,ISTOP,ITN,NORMR] = LSMR(A,B,...) gives an estimate of the residual
%   norm: NORMR = norm(B-A*X).
%
%   [X,ISTOP,ITN,NORMR,NORMAR] = LSMR(A,B,...) gives an estimate of the
%   residual for the normal equation: NORMAR = NORM(A'*(B-A*X)).
%
%   [X,ISTOP,ITN,NORMR,NORMAR,NORMA] = LSMR(A,B,...) gives an estimate of
%   the Frobenius norm of A.
%
%   [X,ISTOP,ITN,NORMR,NORMAR,NORMA,CONDA] = LSMR(A,B,...) gives an estimate
%   of the condition number of A.
%
%   [X,ISTOP,ITN,NORMR,NORMAR,NORMA,CONDA,NORMX] = LSMR(A,B,...) gives an
%   estimate of NORM(X).
%
%   LSMR uses an iterative method requiring matrix-vector products A*v
%   and A'*u.  For further information, see
%      D. C.-L. Fong and M. A. Saunders,
%      LSMR: An iterative algorithm for sparse least-squares problems,
%      SIAM J. Sci. Comput., submitted 1 June 2010.
%      See http://www.stanford.edu/~clfong/lsmr.html.
%
% 08 Dec 2009: First release version of LSMR.
% 09 Apr 2010: Updated documentation and default parameters.
% 14 Apr 2010: Updated documentation.
% 03 Jun 2010: LSMR with local and/or full reorthogonalization of v_k.
% 10 Mar 2011: Bug fix in reorthgonalization. (suggested by David Gleich)

% David Chin-lung Fong            clfong@stanford.edu
% Institute for Computational and Mathematical Engineering
% Stanford University
%
% Michael Saunders                saunders@stanford.edu
% Systems Optimization Laboratory
% Dept of MS&E, Stanford University.
%-----------------------------------------------------------------------

  % Initialize.

  [m, n] = size(A);
  minDim = min([m n]);

  % Retrieve input arguments.
  lambda = 0;
  atol = 1.0e-6;
  btol = 1.0e-6;
  etol = 1.0e-6;
  conlim = 1.0e+8;
  itnlim = minDim;
  localSize = 0;
  show = false;
  wantvar = false;
  resvec = [];
  Aresvec = [];

  M = opEye(m);
  M_given = false;
  N = opEye(n);
  N_given = false;
  window = 5;
  x_energy_norm2 = 0;              % Squared energy norm of x.
  x_energy_norm = 0;               % Energy norm of x.
  err_vector = zeros(window,1);   % Lower bound on direct error in energy norm.
  err_lbnds = [];                  % History of values of err_lbnd.
  err_lbnd_small = false;

  if nargin > 2
    if isfield(opts, 'damp')
      lambda = opts.damp;
    end
    if isfield(opts, 'atol')
      atol = opts.atol;
    end
    if isfield(opts, 'btol')
      btol = opts.btol;
    end
    if isfield(opts, 'etol')
      etol = opts.etol;
    end
    if isfield(opts, 'conlim')
      conlim = opts.conlim;
    end
    if isfield(opts, 'itnlim')
      itnlim = opts.itnlim;
    end
    if isfield(opts, 'localSize')
      localSize = opts.localSize;
    end
    if isfield(opts, 'show')
      show = opts.show;
    end
    if isfield(opts, 'wantvar')
      wantvar = opts.wantvar;
    end
    if isfield(opts, 'M')
      M = opts.M;
      M_given = true;
    end
    if isfield(opts, 'N')
      N = opts.N;
      N_given = true;
    end
    if isfield(opts, 'window')
      window = opts.window;
    end
    if isfield(opts, 'sqd')
      if opts.sqd & M_given & N_given
        lambda = 1.0;
      end
    end
  end

  if wantvar, var = zeros(n,1); end

  msg = ['The exact solution is  x = 0                              '
         'Ax - b is small enough, given atol, btol                  '
         'The least-squares solution is good enough, given atol     '
         'The estimate of cond(Abar) has exceeded conlim            '
         'Ax - b is small enough for this machine                   '
         'The least-squares solution is good enough for this machine'
         'Cond(Abar) seems to be too large for this machine         '
         'The iteration limit has been reached                      '
         'The truncated direct error is small enough, given etol    '];

  hdg1 = '   itn      x(1)       norm r    norm A''r';
  hdg2 = ' compatible   LS      norm A   cond A';
  pfreq  = 20;   % print frequency (for repeating the heading)
  pcount = 0;    % print counter

  % Form the first vectors u and v.
  % These satisfy  beta*u = b,  alpha*v = A'u.

  Mu   = b;
  u    = M * Mu;
  beta = realsqrt(dot(u, Mu));
  if beta > 0
    u  = u/beta;
    Mu = Mu/beta;
  end

  Nv = A' * u;
  v  = N * Nv;

  if show
    fprintf('\n\nLSMR            Least-squares solution of  Ax = b')
    fprintf('\nVersion 1.11                          09 Jun 2010')
    fprintf('\nThe matrix A has %8g rows  and %8g cols', m,n)
    fprintf('\nlambda = %16.10e', lambda )
    fprintf('\natol   = %8.2e               conlim = %8.2e', atol,conlim)
    fprintf('\nbtol   = %8.2e               itnlim = %8g'  , btol,itnlim)
  end

  alpha = realsqrt(dot(v, Nv));
  if alpha > 0
    v  = (1/alpha)*v;
    Nv = (1/alpha)*Nv;
  end

  % Initialization for local reorthogonalization.

  localOrtho = false;
  if localSize > 0
    localPointer    = 0;
    localOrtho      = true;
    localVQueueFull = false;

    % Preallocate storage for the relevant number of latest v_k's.

    localV = zeros(n, min([localSize minDim]));
  end

  % Initialize variables for 1st iteration.

  itn      = 0;
  zetabar  = alpha*beta;
  alphabar = alpha;
  rho      = 1;
  rhobar   = 1;
  cbar     = 1;
  sbar     = 0;

  h    = v;
  hbar = zeros(n,1);
  x    = zeros(n,1);

  % Initialize variables for estimation of ||r||.

  betadd      = beta;
  betad       = 0;
  rhodold     = 1;
  tautildeold = 0;
  thetatilde  = 0;
  zeta        = 0;
  d           = 0;

  % Initialize variables for estimation of ||A|| and cond(A).

  normA2  = alpha^2;
  maxrbar = 0;
  minrbar = 1e+100;

  % Items for use in stopping rules.
  normb  = beta;
  istop  = 0;
  ctol   = 0;         if conlim > 0, ctol = 1/conlim; end;
  normr  = beta;

  resvec = [resvec ; normr];

  % Exit if b=0 or A'b = 0.

  normAr = alpha * beta;
  Aresvec = [Aresvec ; normAr];
  if normAr == 0, disp(msg(1,:)); return, end

  % Heading for iteration log.

  if show
    test1 = 1;
    test2 = alpha/beta;
    fprintf('\n\n%s%s'      , hdg1 , hdg2   )
    fprintf('\n%6g %12.5e'  , itn  , x(1)   )
    fprintf(' %10.3e %10.3e', normr, normAr )
    fprintf('  %8.1e %8.1e' , test1, test2  )
  end


  %------------------------------------------------------------------
  %     Main iteration loop.
  %------------------------------------------------------------------
  while itn < itnlim
    itn = itn + 1;

    % Perform the next step of the bidiagonalization to obtain the
    % next beta, u, alpha, v.  These satisfy the relations
    %      beta*M*u  =  A*v  - alpha*M*u,
    %      alpha*N*v  =  A'*u - beta*N*v.

    Mu = A*v    - alpha * Mu;
    u  = M * Mu;
    beta = realsqrt(dot(u, Mu));

    if beta > 0
      u = (1/beta)*u;
      Mu = (1/beta)*Mu;
      if localOrtho
        localVEnqueue(v);    % Store old v for local reorthogonalization of new v.
      end
      Nv = A' * u - beta * Nv;
      v  = N * Nv;

      if localOrtho
        v = localVOrtho(v);  % Local-reorthogonalization of new v.
      end
      alpha  = realsqrt(dot(v, Nv));
      if alpha > 0
        v  = (1/alpha)*v;
        Nv = (1/alpha)*Nv;
      end
    end

    % At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.

    % Construct rotation Qhat_{k,2k+1}.

    alphahat = norm([alphabar lambda]);
    chat     = alphabar/alphahat;
    shat     = lambda/alphahat;

    % Use a plane rotation (Q_i) to turn B_i to R_i.

    rhoold   = rho;
    rho      = norm([alphahat beta]);
    c        = alphahat/rho;
    s        = beta/rho;
    thetanew = s*alpha;
    alphabar = c*alpha;

    % Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar.

    rhobarold = rhobar;
    zetaold   = zeta;
    thetabar  = sbar*rho;
    rhotemp   = cbar*rho;
    rhobar    = norm([cbar*rho thetanew]);
    cbar      = cbar*rho/rhobar;
    sbar      = thetanew/rhobar;
    zeta      =   cbar*zetabar;
    zetabar   = - sbar*zetabar;

    x_energy_norm2 = x_energy_norm2 + zeta * zeta;

    % Update h, h_hat, x.

    hbar      = h - (thetabar*rho/(rhoold*rhobarold))*hbar;
    x         = x + (zeta/(rho*rhobar))*hbar;
    h         = v - (thetanew/rho)*h;

    % See if lower bound on direct error has converged.

    err_vector(mod(itn,window)+1) = zeta;
    if itn >= window
      err_lbnd = norm(err_vector);
      err_lbnds = [err_lbnds ; err_lbnd];
      err_lbnd_small = (err_lbnd <= etol * sqrt(x_energy_norm2));
    end

    % Estimate of ||r||.

    % Apply rotation Qhat_{k,2k+1}.
    betaacute =   chat* betadd;
    betacheck = - shat* betadd;

    % Apply rotation Q_{k,k+1}.
    betahat   =   c*betaacute;
    betadd    = - s*betaacute;

    % Apply rotation Qtilde_{k-1}.
    % betad = betad_{k-1} here.

    thetatildeold = thetatilde;
    rhotildeold   = norm([rhodold thetabar]);
    ctildeold     = rhodold/rhotildeold;
    stildeold     = thetabar/rhotildeold;
    thetatilde    = stildeold* rhobar;
    rhodold       =   ctildeold* rhobar;
    betad         = - stildeold*betad + ctildeold*betahat;

    % betad   = betad_k here.
    % rhodold = rhod_k  here.

    tautildeold   = (zetaold - thetatildeold*tautildeold)/rhotildeold;
    taud          = (zeta - thetatilde*tautildeold)/rhodold;
    d             = d + betacheck^2;
    normr         = realsqrt(d + (betad - taud)^2 + betadd^2);

    resvec = [resvec ; normr];

    % Estimate ||A||.
    normA2        = normA2 + beta^2;
    normA         = realsqrt(normA2);
    normA2        = normA2 + alpha^2;

    % Estimate cond(A).
    maxrbar       = max(maxrbar,rhobarold);
    if itn>1
      minrbar     = min(minrbar,rhobarold);
    end
    condA         = max(maxrbar,rhotemp)/min(minrbar,rhotemp);

    % Test for convergence.

    % Compute norms for convergence testing.
    normAr  = abs(zetabar);
    Aresvec = [Aresvec ; normAr];
    normx   = norm(x);

    % Now use these norms to estimate certain other quantities,
    % some of which will be small near a solution.

    test1   = normr /normb;
    test2   = normAr/(normA*normr);
    test3   =      1/condA;
    t1      =  test1/(1 + normA*normx/normb);
    rtol    = btol + atol*normA*normx/normb;

    % The following tests guard against extremely small values of
    % atol, btol or ctol.  (The user may have set any or all of
    % the parameters atol, btol, conlim  to 0.)
    % The effect is equivalent to the normAl tests using
    % atol = eps,  btol = eps,  conlim = 1/eps.

    if itn >= itnlim,   istop = 7; end
    if err_lbnd_small,  istop = 8; end
    if 1 + test3  <= 1, istop = 6; end
    if 1 + test2  <= 1, istop = 5; end
    if 1 + t1     <= 1, istop = 4; end

    % Allow for tolerances set by the user.

    if  test3 <= ctol,  istop = 3; end
    if  test2 <= atol,  istop = 2; end
    if  test1 <= rtol,  istop = 1; end

    % See if it is time to print something.

    if show
      prnt = 0;
      if n     <= 40       , prnt = 1; end
      if itn   <= 10       , prnt = 1; end
      if itn   >= itnlim-10, prnt = 1; end
      if rem(itn,10) == 0  , prnt = 1; end
      if test3 <= 1.1*ctol , prnt = 1; end
      if test2 <= 1.1*atol , prnt = 1; end
      if test1 <= 1.1*rtol , prnt = 1; end
      if istop ~=  0       , prnt = 1; end

      if prnt
      	if pcount >= pfreq
      	  pcount = 0;
                fprintf('\n\n%s%s'    , hdg1 , hdg2  )
      	end
      	pcount = pcount + 1;
        fprintf('\n%6g %12.5e'  , itn  , x(1)  )
        fprintf(' %10.3e %10.3e', normr, normAr)
        fprintf('  %8.1e %8.1e' , test1, test2 )
        fprintf(' %8.1e %8.1e'  , normA, condA )
      end
    end

    if istop > 0, break, end
  end % iteration loop

  % Print the stopping condition.

  if show
    fprintf('\n\nLSMR finished')
    fprintf('\n%s', msg(istop+1,:))
    fprintf('\nistop =%8g    normr =%8.1e'     , istop, normr )
    fprintf('    normA =%8.1e    normAr =%8.1e', normA, normAr)
    fprintf('\nitn   =%8g    condA =%8.1e'     , itn  , condA )
    fprintf('    normx =%8.1e\n', normx)
  end

  % Collect statistics.
  stats.istop = istop;
  stats.msg = msg(istop+1,:);
  stats.normr = normr;
  stats.normAr = normAr;
  stats.normA = normA;
  stats.condA = condA;
  stats.normx = normx;
  stats.resvec = resvec;
  stats.Aresvec = Aresvec;
  stats.err_lbnds = err_lbnds;
  stats.x_energy_norm = sqrt(x_energy_norm2);

  flags.solved = (istop >= 1 & istop <= 3) | (istop >= 5 & istop <= 6) | istop == 8;
  flags.niters = itn;

% end function lsmr

%---------------------------------------------------------------------
% Nested functions.
%---------------------------------------------------------------------

  function localVEnqueue(v)

  % Store v into the circular buffer localV.

    if localPointer < localSize
      localPointer = localPointer + 1;
    else
      localPointer = 1;
      localVQueueFull = true;
    end
    localV(:,localPointer) = v;

  end % nested function localVEnqueue

%---------------------------------------------------------------------

  function vOutput = localVOrtho(v)

  % Perform local reorthogonalization of V.

    vOutput = v;
    if localVQueueFull
      localOrthoLimit = localSize;
    else
      localOrthoLimit = localPointer;
    end
    for localOrthoCount = 1:localOrthoLimit
      vtemp   = localV(:, localOrthoCount);
      vOutput = vOutput - (vOutput'*N*vtemp)*vtemp;
    end

  end % nested function localVOrtho

end % function lsmr
