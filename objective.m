function [obj] = objective(Y, U,V,theta,dU,dV,dtheta,lambda,l,et)
    [n,m] = size(Y);
    U = U + et.*dU;
    V = V + et.*dV;
    theta = theta + et.*dtheta;
    
    regobj = lambda.*(sum(U(:).^2)+sum(V(:).^2))./2;
    lossobj = 0;
    X = U*V';
    Ygt0 = Y>0;
    BX = X.*Ygt0;
    for k=1:l+1
        S = Ygt0-(2 .*(Y > (k-1))); %S is T in the paper
        % Next line is the memory bottleneck
        BZ = (theta(:,k)*ones(1,m)).*S - BX.*S; % [n,m] (sparse)
        lossobj = lossobj + sum(sum(h(BZ)));
    end
    obj = regobj + lossobj;

function [ret] = h(z)
  zin01 = (z>0)-(z>=1);
  zle0 = z<0;
  ret = zin01./2 - zin01.*z + zin01.*z.^2./2 + zle0./2 - zle0.*z;