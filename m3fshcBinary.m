function [obj,grad] = m3fshcBinary(v,parameter)
Y = parameter.Y;
lambda = parameter.lambda;
p = parameter.p;

[n,m] = size(Y);
U = reshape(v(1:n*p),n,p);
V = reshape(v(n*p+1:n*p+m*p),m,p);
clear v;
X = U*V';
dU = lambda.*U;
dV = lambda.*V;
regobj = lambda.*( sum(U(:).^2) + sum(V(:).^2) )./2; % [scalar]
BZ =  Y .* X ;
clear X
lossobj =  sum(sum(h(BZ) .* (Y ~= 0)));
tmp = hprime(BZ) ;
clear BZ;
dU = dU + ( (tmp .* Y) * V );
dV = dV + ( (tmp' .* Y') * U );
clear U V tmp
obj = regobj + lossobj;
grad = [dU(:) ; dV(:)];
clear dU dV regobj;

function [ret] = h(z)
zin01 = (z>0)-(z>=1);
zle0 = z<0;
ret = ( ( (zin01./2 - zin01.*z ) + zin01.*z.^2./2 ) + zle0./2 ) - zle0.*z;
clear zin01 zle0


function [ret] = hprime(z)
zin01 = (z>0)-(z>=1);
zle0 = z<0;
ret = zin01.*z - zin01 - zle0;
clear zin01 zle0



%{
function [ret] = h(z)
  zle1 = z < 1;
  ret =  1 .* zle1 - z .* zle1  ;
  

function [ret] = hprime(z)
  zle1 = z < 1;
  ret =  - zle1;
%}