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

prevobj = realmax('double');

while num_iter < maxiter & error > threshold
    num_iter = num_iter + 1;
    %num_iter
    [obj,dx] = objfun(x,parameter);
    
    d = -dx;
    
    U = reshape(x(1:n*p),n,p);
    V = reshape(x(n*p+1:n*p+m*p),m,p);
    theta = reshape(x(n*p+m*p+1:n*p+m*p+n*(l+1)),n,l+1);
    
    dU = reshape(d(1:n*p),n,p);
    dV = reshape(d(n*p+1:n*p+m*p),m,p);
    dtheta = reshape(d(n*p+m*p+1:n*p+m*p+n*(l+1)),n,l+1);
    if obj >= prevobj
        %fprintf(fs,'--Iter: = %d \t Eta: = %.4f \n', num_iter, eta);
        tp = linspace(eta, 0, 8);
        eta_arr = tp(1:length(tp)-1);
        flag = 0;
        for id=2:length(eta_arr)
            obj_h = objective(Ytrn,U,V,theta,dU,dV,dtheta,parameter.lambda,l,eta_arr(id));
            if obj_h < prevobj
                J = [J; obj_h];
                U = U + eta_arr(id).*dU;
                V = V + eta_arr(id).*dV;
                theta = theta + eta_arr(id).*dtheta;
                flag = 1;
                prevobj = obj_h;
                fprintf(fs,'Eta_arr = %.4f \n',eta_arr(id));
                break;
            end
        end
        if flag == 0
            break;
        end
    else
        
        J = [J; obj];

        U = U + eta.*dU;
        V = V + eta.*dV;
        theta = theta + eta.*dtheta;
        prevobj = obj;
    end
    
    x = [U(:); V(:); theta(:)];
    %size(x)
    X = U*V';
    
    YPred = m3fSoftmax(X, theta);
    
    error = RMSE(YPred, Ytrn);
    
    if(mod(num_iter, 1) == 0)
        fprintf(fs,'Iter: = %d \t Obj: = %.4f \n',num_iter,obj);
    end
    %error
end

    