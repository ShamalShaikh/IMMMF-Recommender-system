% Calculation of gradient and objective for All-Threshold SHC
% MMMF.
%
% function [obj,grad,lossobj,regobj] = m3fshc(v,Y,lambda,l,varargin)
% v - vector of parameters [n*p+m*p+n*(l-1),1]
% Y - rating matrix (labels) [n,m]
% lambda - regularization parameter [scalar]
% l - # of unique rating values (1..l)
% obj - value of objective at v [scalar]
% grad - gradient at v [n*p+m*p+n*(l-1),1]
% lossobj - loss component of objective [scalar]
% regobj - regularization component of objective [scalar]
% 
% Written by Jason Rennie, January 2005
% Last modified: Tue Jan 16 09:25:25 2007

%function [obj,grad,lossobj,regobj] = m3fshc(v,Y,lambda,l,varargin)
function [obj,grad,lossobj,regobj] = m3fshc(v,parameter)
  fn = mfilename;
  if nargin < 2
    error('insufficient parameters')
  end
  % Parameters that can be set via varargin
  %verbose = 1;
  % Process varargin
  %paramgt;
  Y = parameter.Y;
  lambda = parameter.lambda;
  l = parameter.l;
  
  t0 = clock;
  [n,m] = size(Y);
  p = (length(v)-n.*(l-1))./(n+m);
  if p ~= floor(p) | p < 1
    error('dimensions of v and Y don''t match l');
  end
  U = reshape(v(1:n*p),n,p);
  V = reshape(v(n*p+1:n*p+m*p),m,p);
  theta = reshape(v(n*p+m*p+1:n*p+m*p+n*(l-1)),n,l-1);
  clear v;

  X = U*V';
  Ygt0 = Y>0;
  BX = X.*Ygt0;
  %reallyzero = (X==0).*Ygt0;
  %fprintf(1,'m3fshc: sum(sum(reallyzero))=%d\n',full(sum(sum(reallyzero))));
  clear X;
  %lambda is regularized value that we are passing from weak.m
  dU = lambda.*U; % [n,p]
  dV = lambda.*V; % [m,p]
  dtheta = zeros(n,l-1); % [n,l-1]
  regobj = lambda.*(sum(U(:).^2)+sum(V(:).^2))./2; % [scalar]
  lossobj = 0;
  for k=1:l-1
    S = Ygt0-2.*(Y>k); %S is T in the paper
    % Next line is the memory bottleneck
    BZ = (theta(:,k)*ones(1,m)).*S - BX.*S; % [n,m] (sparse)
    lossobj = lossobj + sum(sum(h(BZ)));
    tmp = hprime(BZ).*S; % [n,m]
    clear BZ S;
    dU = dU - tmp*V; % [n,p]
    dV = dV - tmp'*U; % [m,p]
    dtheta(:,k) = tmp*ones(m,1);
    clear tmp;
  end
  obj = regobj + lossobj; % obj is the objective function that we need to minimize
  grad = [dU(:); dV(:); dtheta(:)];
%   if verbose
%     fprintf(1,'%s: lambda=%.2e obj=%.2e grad''*grad=%.2e time=%.1f\n',fn,lambda,obj,grad'*grad,etime(clock,t0));
%   end

% ret = (z>0).*(z<1).*((1-z).^2)/2.0 + (z<=0).*(0.5-z);
function [ret] = h(z)
  zin01 = (z>0)-(z>=1);
  zle0 = z<0;
  ret = zin01./2 - zin01.*z + zin01.*z.^2./2 + zle0./2 - zle0.*z;

% ret = (z>0).*(z<1).*(z-1) - (z<=0);
function [ret] = hprime(z)
  zin01 = (z>0)-(z>=1);
  zle0 = z<0;
  ret = zin01.*z - zin01 - zle0;

% ChangeLog
% 12/8/06 - remove reallyzero (print warning if it would be necessary)
% 3/23/05 - made calcultions take better advantage of sparseness
% 3/18/05 - fixed bug in objective (wasn't squaring fro norms)
% 3/1/05 - added objective calculation
% 2/23/05 - fixed bug in hprime()
