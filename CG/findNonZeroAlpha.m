% Make sure alpha isn't smaller than level of precision
if x0+alpha.*direction == x0
  preAlpha = alpha;
  while x0+alpha.*direction == x0
    alpha = alpha .* gamma;
  end
  [obj,dx] = ogfun(x0+alpha.*direction,parameter);
  ogcalls = ogcalls + 1;
%   if verbose > 3
%    % fprintf(1,'FindNonZeroAlpha preAlpha=%.2e alpha=%.2e\n',preAlpha,alpha);
%   end
end
  
