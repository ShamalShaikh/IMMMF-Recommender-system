function [error] = RMSE(X,Y)
tmp = ( (Y - X).* (Y ~= 0) ) .^ 2 ;
error = full(sqrt( sum( sum(tmp) ) / sum( sum(Y~=0) ) ));
end