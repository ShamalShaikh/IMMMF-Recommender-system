function [error] = averageRMSE(X, Y)
[~, m] = size(Y);
tmp = ( (Y - X).* (Y ~= 0) ) .^ 2 ;
error = 0;
 for i=1:m
     if sum(Y(:,i)~=0) ~=0
        error = error + sqrt( sum(tmp(:,i)) / sum(Y(:,i)~=0)  );
     end
 end
error = full(error / m);