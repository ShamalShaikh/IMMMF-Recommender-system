function [weakTrn, weakTst, strongTrn, strongTst] = allBut1Division(weakR, strongR)
%%
[n, m] = size(weakR);
weakTst = zeros(n, m);
for i=1:n
    idx = find(weakR(i,:));
    idx = idx( randperm( size(idx,2) ) );
    weakTst(i,idx(1,1)) = weakR(i, idx(1,1) );
    weakR(i, idx(1,1) ) = 0;
end
weakTrn = weakR;
clear weakR;

%%
[n, m] = size(strongR);
strongTst = zeros(n, m);
for i=1:n
    idx = find(strongR(i,:));
    idx = idx( randperm( size(idx,2) ) );
    strongTst(i,idx(1,1)) = strongR(i, idx(1,1) );
    strongR(i, idx(1,1) ) = 0;
end
strongTrn = strongR;
clear strongR;

end