function r = IsBetterThanBefore(Result,CurrentResult)
% 1 MAE
% 2 RMSE
% 3 AUC
% 4 PRECISION@1 ...  23 PRECISION@20 ...
% 24 RECALL@1   ...  43 RECALL@20 ...
% 44 F1@1       ...  63 F1@20 ...
% 64 NDCG@1     ...  83 NDCG@20 ...

CurrentResult(isnan(CurrentResult))=0;
% a =   CurrentResult(3,1) + sum(CurrentResult(4:13))  + sum(CurrentResult(24:33))...
%       + sum(CurrentResult(44:53)) + sum(CurrentResult(64:73));
% b =   Result(3,1) + sum(Result(4:13))  + sum(Result(24:33))...
%       + sum(Result(44:53)) + sum(Result(64:73));
  
% if a >= b
%     r = 1;
% else
%     r = 0;
% end

a =   CurrentResult(1) + CurrentResult(2) + CurrentResult(3);
b =   Result(1) + Result(2)  + Result(3);
if a <= b
    r = 1;
else
    r = 0;
end

end