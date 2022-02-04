% Quadratic interpolation line search with backtracking fallback;
% Uses Strong Wolfe termination conditions; returns distance along
% direction (alpha) and objective (obj) and gradient (dx) at
% x0+alpha.*direction
%
% function [alpha,obj,dx] = cgLineSearch(x0,obj0,dx0,direction,ogfun,ogparams,varargin)
% x0 - initial parameter value [d,1]
% obj0 - objective value at x0 [scalar]
% dx0 - gradient at x0 [d,1]
% direction - direction along which to find minimum [d,1]
% ogfun - pointer to objective/gradient calculation function
% function [obj,dx] = ogfun(x,params)
%     x - parameter value at which to calculate obj/grad [d,1]
%     params - additional parameters
%     obj - objective value at x [scalar]
%     dx - gradient at x [d,1]
% alpha - approximate minimum along direction is found at x0+alpha.*direction
%
% Additional parameters given by name/value pairs via varargin.
% E.g. cgLineSearch(w,obj0,dx0,d,objfn,'c2',1e-2,'verbose',0)
%
% Secant Line Search algorithm taken from "An introduction to the
% conjugate gradient method without the agonizing pain" by Jonathan
% Shewchuk (1994).
%
% Written by Jason Rennie, February 2005
% Last modified: Wed Sep  6 18:53:39 2006
%[alpha,obj,dx,ogc] = lsfun(x,obj,dx,d,'alpha0',alpha,parameter);
%function [alpha,obj,dx,ogcalls] = cgLineSearch(x0,obj0,dx0,direction,ogfun,ogparams,varargin)
function [alpha,obj,dx,ogcalls] = cgLineSearch(x0,obj0,dx0,direction,alpha, parameter)
  if nargin < 6
    error('Insufficient parameters');
  end
  % Parameters that can be set via varargin
  seciter = 5; % maximum number of quadratic interpolation iterations
  %alpha0 = 1e-10;
  alpha0 = alpha;
  c1 = 1e-4; % required decrease in objective (relative to gradient)
  %c2 = 1e-1; % required decrease in directional derivative
  c2 = parameter.c2;
  digits = 12; % digits of precision to use for objective comparisons
  gamma = 10;
  ogfun = parameter.objGrad;
  % Check initial values
  if alpha0 <= 0
    error('alpha0 must be greater than zero');
  end

  %%%%% Begin line search %%%%%
  %hessdiag = 0; % bogus variable, used in some subroutines shared with pccg
  obj = obj0;
  etazero = full(dx0'*direction);
  etaprev = etazero;
  alpha = alpha0;
  ogcalls = 0;
  findNonZeroAlpha;
  [obj,dx] = ogfun(x0+alpha.*direction,parameter);
  ogcalls = ogcalls + 1;
  saveAlpha;
  % Make sure current step yields decrease in objective
  backtrack;
  beta = alpha;
  eta = full(dx'*direction);
  i = 0;
  % Try to find a point with lower directional derivative
  % Stop if too many iterations, large objective, no change, or small alpha
%   if verbose >= 4
%     %fprintf(1,'abs(eta) > c2.*abs(etazero)=%d\n',abs(eta) > c2.*abs(etazero));
%     %fprintf(1,'i < seciter=%d\n',i < seciter);
%    % fprintf(1,'obj <= obj0=%d\n',obj <= obj0);
%    % fprintf(1,'etaprev ~= eta=%d\n',etaprev ~= eta);
%    % fprintf(1,'sum(x0+alpha.*direction ~= x0)>0=%d\n',sum(x0+alpha.*direction ~= x0)>0);
%   end
  while abs(eta) > c2.*abs(etazero) & i < seciter & pround(obj,digits) <= pround(obj0,digits) & etaprev ~= eta & sum(x0+alpha.*direction ~= x0)>0
    % Want newBeta/(eta-0) = oldBeta/(etaprev-eta)
%     if verbose >= 4
%         %  fprintf(1,'eta=%.4e etaprev=%.4e beta=%.4e\n',eta,etaprev,beta);
%     end
    beta = eta.*beta./(etaprev-eta);
    saveAlpha;
    alpha = alpha + beta;
%     if verbose >= 4
%      % fprintf(1,'alpha=%.4e\n',alpha);
%     end
    % Negative alpha could be a sign that initial parameter vector is bad
    if alpha <= 0
%       if verbose >= 3
%         %fprintf('Negative Alpha: alpha=%.4e\n',alpha);
%       end
      alpha = 1;
    end
    etaprev = eta;
    i = i + 1;
    [obj,dx] = ogfun(x0+alpha.*direction,parameter);
    ogcalls = ogcalls + 1;
    eta = full(dx'*direction);
  end
  findNonZeroAlpha;
  backtrack;
  checkConditions;
  
%%%%%%% Need to ensure that loop will terminate if negative alpha %%%%%%%

% ChangeLog
% 7/25/06 - Ignore differences of less than 12 significant digits (double
%           only has 16 digits of precision; I've seen spurious effects at the
%           14th significant digit)
% 7/24/06 - Remove all pccg/hess(ian) references from code; move pccg code to 
%           "old" directory; pccg code isn't sufficiently useful to warrant
%           the upkeep effort
% 7/21/06 - if non-linear approximation doesn't work, set alpha=1 and let 
%           backtrack select appropriate step size (alpha)
% 11/3/05 - make objective/gradient function call counting work
% 6/20/05 - ensure that alpha is >= 0
% 6/3/05 - rename to cgLineSearch
% 6/3/05 - farm-out job of finding non-zero alpha
% 6/3/05 - farm-out backtrack routine to separate script
% 5/18/05 - ensure that initial alpha produces obj <= obj0
% 5/18/05 - increased epsilon to 1e-30, set secstep to 1e-10
% 5/17/05 - make sure eta is full() in all calculations
% 5/8/05 - Place limit on # of quadratic interpolations
% 3/20/05 - Return alpha=0 if it appears we can't find decrease
% 3/20/05 - Don't check for increasing objective until 2nd iteration
