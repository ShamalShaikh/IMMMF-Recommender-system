function [y,U,V,theta] = mmmfWeak(weakTrn,par)

par.Y = weakTrn;
p = par.p;
l = par.l;
[n, m] = size(weakTrn);
if exist('par.v0')
     v0 = par.v0;
else
    v0 = randn(n*p+m*p+n*(l-1),1);
end
[v, numiter, ogcalls, J] = conjgrad(v0,par);
fprintf('total conjugate gradient iteration in MMMF = %d\n',numiter);
fprintf('total number of line search call in MMMF = %d\n',ogcalls);
U = reshape(v(1:n*p),n,p);
V = reshape(v(n*p+1:n*p+m*p),m,p);
theta = reshape(v(n*p+m*p+1:n*p+m*p+n*(l-1)),n,l-1);
X = U*V';
%clear v U V;
y = m3fSoftmax(X,theta);
end
