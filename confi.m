function [conf] = confi(Y, XY, theta, l)
    %CONFI Summary of this function goes here
    %   Detailed explanation goes here
    [n,m] = size(Y);
    TBY = zeros(n,m);
    conf = zeros(n,m);
    BY = zeros(n,m);
    T = zeros(n,m);
    
    for k=1:l-1
        tmp = theta(:,k) * ones(1,m);
        
        BY = (Y==k | Y==k+1);
        TBY = TBY + BY;
        tmp = abs((tmp - XY) .* BY);
        % Minmax normalization
        ma = full(max(max(tmp)));
        mi = full(min(min(tmp(tmp>0))));
        
        tmp = ((tmp-mi)./(ma-mi)) .* (tmp>0);
        conf = conf + tmp;
        clear tmp;
    end
    conf = conf ./ TBY;
end

