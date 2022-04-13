%function [obj,grad] = m3fshcBinaryFixedV(v,Y,V,lambda,p)
function [obj,grad] = m3fshcBinaryFixedV(v,parameter)
Y = parameter.Y;
lambda = parameter.lambda;
p = parameter.p;
V = parameter.V;
[n, ~] = size(Y);
U = reshape(v(1:n*p),n,p);
clear v;
X = U*V';
dU = lambda.*U;

regobj = lambda.*( sum(U(:).^2) ); % [scalar]
BZ =  Y .* X ;
clear X
lossobj =  sum(sum(h(BZ) .* (Y ~= 0)));
tmp = hprime(BZ) ;
clear BZ;
dU = dU + ( (tmp .* Y) * V );
clear U V tmp
obj = regobj + lossobj;
grad = dU(:);
clear dU regobj;

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