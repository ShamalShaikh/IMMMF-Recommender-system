function [Y,flagDelUser] = delUser(Y)
L = max(Y(:));
[n,~] = size(Y);
flagDelUser = false(n,1);
for i=1:L
    flagDelUser(sum(Y == i,2) == 0) = 1;
end
Y(flagDelUser,:) = [];
end