function [x, num_iter, J] = graddesc(x0,parameter, fs)
%GRADDESC Summary of this function goes here
%   Detailed explanation goes here
J           = [];
[n,m]       = size(parameter.Y);
x           = x0;
eta         = parameter.eta;
p           = parameter.p;
l           = parameter.l;
objfun      = parameter.objGrad;
maxiter     = parameter.maxiter;
num_iter    = 0;
error       = 1e+10;
threshold   = 0.3;
Ytrn        = parameter.Y;
m3fSoftmax   = parameter.softmax;

while num_iter < maxiter & error > threshold
    num_iter = num_iter + 1;
    %num_iter
    [obj,dx] = objfun(x,parameter);
    J = [J; obj];
    d = -dx;
    
    U = reshape(x(1:n*p),n,p);
    V = reshape(x(n*p+1:n*p+m*p),m,p);
    theta = reshape(x(n*p+m*p+1:n*p+m*p+n*(l-1)),n,l-1);
    
    dU = reshape(d(1:n*p),n,p);
    dV = reshape(d(n*p+1:n*p+m*p),m,p);
    dtheta = reshape(d(n*p+m*p+1:n*p+m*p+n*(l-1)),n,l-1);
    
    U = U + eta.*dU;
    V = V + eta.*dV;
    theta = theta + eta.*dtheta;
    
    x = [U(:); V(:); theta(:)];
    
    X = U*V';
    
    YPred = m3fSoftmax(X, theta);
    
    error = RMSE(YPred, Ytrn);
    fprintf(fs,'Error:\tRMSE = %.4f \n',error);
    %error
end

