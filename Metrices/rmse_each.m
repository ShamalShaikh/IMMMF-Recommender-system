function [error] = rmse_each(X,Y,r)
tmp = ( (Y - X).* (Y == r) ) .^ 2 ;
error = full(sqrt( sum( sum(tmp) ) / sum( sum(Y==r) ) ));
end