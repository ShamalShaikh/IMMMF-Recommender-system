% Ensures either (1) Armijo lower objective, or (2) infinitesimal alpha
doBacktrack = 0;
preAlpha = alpha;
while (pround(obj,digits) > pround(obj0 + c1.*alpha.*etazero,digits)) & sum(x0+alpha.*direction ~= x0)>0
  if ogcalls > 1
    if oldalpha > alpha./gamma & oldobj < obj0 + c1.*oldalpha.*etazero
      restoreAlpha;
      doBacktrack = 1;
      break;
    end
  end
  alpha = alpha ./ gamma;
  [obj,dx] = ogfun(x0+alpha.*direction,parameter);
  ogcalls = ogcalls + 1;
  doBacktrack = 1;
end

% if doBacktrack & verbose >= 3
%   fprintf(1,'Backtracking preAlpha=%.2e alpha=%.2e\n',preAlpha,alpha);
% end

% ChangeLog
% 7/7/06 - separate pre-conditioned CG and regular CG backtrack code
