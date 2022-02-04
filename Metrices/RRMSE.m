function [error] = RRMSE(X,Y)
% X predicted
% Y Original
% tmp = ( (Y - X).* (Y > 0) ) .^ 2 ;
% error = full(sqrt( sum( sum(tmp) ) )./  sqrt(sum(sum(Y.^2 .* (Y > 0)))));

tmp = ( (Y - X).* (Y ~=0 ) ) .^ 2 ;
error = full(sqrt( sum( sum(tmp) ) )./  sqrt(sum(sum(Y.^2 .* (Y ~=0)))));