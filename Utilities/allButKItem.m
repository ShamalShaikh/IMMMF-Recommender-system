function [YTrn, YTst, groupWiseTestItems] = allButKItem(Y, UGidx, NoTstIFromG, minUserPerForSel)
%%

flagItemRemainFromG = 1;
[n, m] = size(Y);
K1     = length(unique(UGidx));
YTst   = zeros(n,m);

noUserInG = histcounts(UGidx,'BinMethod','integers');

% Item will be selected in a group only if it is rated by minUserForSeleI of users
minUserForSeleI= ceil( (noUserInG.*minUserPerForSel)./100); 

groupWiseTestItems = cell(K1,1);
for k=1:K1
    groupWiseTestItems{k} = [];
end

while (flagItemRemainFromG)
    
    for k=1:K1
        idx         = setdiff(1:m,groupWiseTestItems{k});
        %idx         = setdiff(idx,find(sum(Y(UGidx==k,:),1)==0));
        idxNotRatedByPperU = find(sum(Y(UGidx==k,:),1)<minUserForSeleI(k));        
        idx         = setdiff(idx,idxNotRatedByPperU);
        
        
        randpermIdx = idx(randperm(length(idx)));
        
        %itemsInTest     = randpermIdx(1:(NoTstIFromG - length(groupWiseTestItems{k})));
        
        itemsInTest     = randpermIdx(1:min(NoTstIFromG - length(groupWiseTestItems{k}), length(randpermIdx)) );
        
        YTst(UGidx==k,itemsInTest) = Y(UGidx==k,itemsInTest);
        groupWiseTestItems{k} = [groupWiseTestItems{k},itemsInTest];
    end
    YTrn = Y.*(YTst==0);
    unObsIdx = find(sum(YTrn,1)==0);
    YTrn(:,unObsIdx) = YTst(:,unObsIdx);
    YTst(:,unObsIdx) = 0;
    flagItemRemainFromG      = length(unObsIdx)>0;
    
    if ~isempty(unObsIdx) >0
        for k=1:K1
            groupWiseTestItems{k}     = setdiff(groupWiseTestItems{k},unObsIdx);
        end
    end
end



%%
%Items to be restricted to only one group
% [n, m] = size(Y);
% noUserAlloweForI = floor(m*perUserAlloweForI/100);
%
% K1     = length(unique(UGidx));
%
% I2GAss = unidrnd(K1,[1,m]);
% YTst   = zeros(n,m);
%
% for k=1:K1
%     idx     = find(I2GAss==k);
%     randIdx = idx(randperm(length(idx)));
%
%     YTst(UGidx==k,randIdx(1:min(NoTstIFromG,length(idx)))) = Y(UGidx==k,randIdx(1:min(NoTstIFromG,length(idx))));
% end
%
% YTrn = Y;
% YTrn = YTrn.*(YTst==0);
end