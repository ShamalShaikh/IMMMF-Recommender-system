function [weakR, strongR] = divideIntoWeakStrong(R,weakMax)
[n, ~] = size(R);
idx = randperm(n);
weakR = sparse(R(idx(1:weakMax),:));
strongR = sparse(R(idx(weakMax+1:end),:));
end