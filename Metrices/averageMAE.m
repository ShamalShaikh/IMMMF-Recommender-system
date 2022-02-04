function [ error ] = averageMAE(X, Y)
[~, m] = size(Y);
tmp = abs( (Y - X).* (Y ~= 0) );
error = 0;
 for i=1:m
     if sum(Y(:,i)~=0) ~=0
        error = error + sum(tmp(:,i)) / sum(Y(:,i)~=0) ;
     end
 end
error = full(error / m);
end