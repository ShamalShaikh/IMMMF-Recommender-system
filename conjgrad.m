% Conjugate Gradients minimization routine.  Uses gradient magnitude
% for termination condition.
%
% function [x] = conjgrad(x0,lsfun,lsparams,ogfun,ogparams,varargin)
% x0 - initial value of the solution [d,1]
% lsfun - pointer to line search function
% function [alpha] = lsfun(x0,direction,ogfun,params)
%         x0 - location to start line search [d,1]
%         direction - direction from x0 along which to search [d,1]
%         ogfun - poitner to objective/gradient function
%         params - cell array of parameters to pass to ogfun
%         alpha - appx. location of minimum along direction
%                 appx. minimium is: x0+alpha.*direction
% lsparams - cell array of parameters to pass to lsparams
% ogfun - pointer to objective/gradient calculation function
% function [obj,grad] = ogfun(x,params)
%     x - parameter value at which to calculate obj/grad [d,1]
%     params - additional parameters
%     obj - objective value at x [scalar]
%     grad - gradient at x [d,1]
% ogparams - cell array of parameters to pass to ogfun
% x - final value of the solution (appxroximate local minimum) [d,1]
%
% Additional parameters given by name/value pairs via varagin.
% E.g. conjgrad(x0,lsfun,params,ogfun,'verbose',0)
%
% This is an implementation of the Polak-Ribiere Nonlinear Conjugate
% Gradients, as described by "An introduction to the conjugate
% gradient method without the agonizing pain" by Shewchuk (1994) and
% "Numerical Optimization" by Nocedal and Wright.
%
% Note: this code automatically uses the last alpha as initial value for
% line search.  This can backfire, causing lots of backtracking, if the
% optimization surface isn't nice.  To avoid this, pass a value of alpha0
% as part of lsparams (e.g. 'alpha0',1e-6)
%
% Written by Jason Rennie, February 2005
% Last modified: Mon Feb  5 21:06:51 2007

%function [x,numiter,ogcalls] = conjgrad(x0,lsfun,lsparams,ogfun,ogparams,varargin)
function [x,numiter,ogcalls, J] = conjgrad(x0,parameter)
  J = [];

  fn = mfilename;
  if nargin() < 2
    error('insufficient parameters')
  end
  temp = 0;
  tol = parameter.tol; % geometric decrease in gradient magnitude to declare minimum
  maxiter = parameter.maxiter; % stop after this many iterations (if no minimum found)
  nu = 0.1;
  abstol = 0; % stop if gradient magnitude goes below this
  allowNonDecrease = 0; % don't stop if line search fails to find decrease
  digits = 12; % digits of precision to use for objective comparisons
  ogfun = parameter.objGrad;
  lsfun = parameter.lineSearchFun;

  t0 = clock;
  t1 = t0;
  ogcalls = 0;
  x = x0;
  numiter = 0;
  j = 0;
  alpha = 1e-10;

  ogcalls = ogcalls + 1;
  [obj,dx] = ogfun(x,parameter);
  r = -dx;
  s = r;
  d = s;
  deltanew = full(r'*d);
  deltazero = deltanew;
%   if verbose >= 2
%    % fprintf(1,'Begin deltazero=%1.1e tol=%.0e obj=%.1f\n',deltazero,tol,obj);
%   end
  while numiter < maxiter & abs(deltanew) > tol.*tol.*abs(deltazero) & abs(deltanew) > abstol
    numiter = numiter + 1;
    j = j + 1;
    fprintf(1,'\n%.4f',obj);
    J = [J;obj];
    prevobj = obj;
    if alpha < 1e-10
      alpha = 1e-10;
    end
    %checkgrad2(ogfun,x,ogparams);
    [alpha,obj,dx,ogc] = lsfun(x,obj,dx,d,alpha,parameter);
    ogcalls = ogcalls + ogc;
    temp=temp+alpha;
    %count=count+1;
    %fprintf('%.4f   %f\n',temp,count);
    x = x + alpha.*d;
    if j==1 & pround(obj,digits) > pround(prevobj,digits) & ~allowNonDecrease
      %fprintf(1,'Line search could not find decrease in direction of negative gradient (bug in objective/gradient code?) dx''*dx=%.4e\n',full(dx'*dx));
      %if verbose >= 2
       % fprintf(1,'%s i=%d delta=%1.3e obj=%1.3e  Time: %.1f s  Calls: %d\n',fn,numiter,deltanew,obj,etime(clock,t0),ogcalls);
      %end
      return
    end
    r = -dx;
    deltaold = deltanew;
    deltamid = full(r'*s);
    deltanew = full(r'*r);
%     if verbose >= 2
%       lastt = t1;
%       t1 = clock;
%       %fprintf(1,'i=%d x=%d alpha=%1.1e delta=%1.1e dobj=%1.1e time=%.1f obj=%.1f\n',numiter,ogc,alpha,deltanew,obj-prevobj,etime(t1,lastt),obj);
%     end
    beta = (deltanew-deltamid)./deltaold;
    d = r + max(0,beta).*d;
    if deltamid./deltanew >= nu | d'*dx >= 0
%       if verbose >= 3 & length(x0) > 1
%       %  fprintf(1,'RESET\n');
%       end
      d = r;
      j = 0;
    end
    s = r;
  end
  numiter;
  %pause(1);
%   if verbose
%     %fprintf(1,'%s i=%d delta=%1.3e obj=%1.3e  Time: %.1f s  Calls: %d\n',fn,numiter,deltanew,obj,etime(clock,t0),ogcalls);
%   end

% ChangeLog
% 7/25/06 - Only print RESET if dimensionality is greater than 1.
% 7/25/06 - Only break loop if line search returns larger objective
%           (accouting for precision issues)
% 5/31/05 - return number of iterations
% 5/18/05 - pass lsparams to lsfun after 'secstep'
% 5/17/05 - default abstol to zero
% 5/17/05 - make sure dx'*dx is full() in warning
% 5/1/05 - Check for d'*dx >= 0
% 3/22/05 - Add abstol parameter/stopping condition
% 3/19/05 - Tell line search to use previous iteration alpha as 1st
% guess; many line searches complete after one obj/grad call,
% especially when close to the solution
