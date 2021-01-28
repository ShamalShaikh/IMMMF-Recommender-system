%function [Result] = EvaluationAll(Yprd, Y1, k, cutoff)
function [Result] = EvaluationAll(Yprd, Y1)
% Result(1,1) = ZOE
% Result(2,1) = MAE
% Result(3,1) = RMSE
% Result(4,1) = Precision
% Result(5,1) = Recall

%Result = zeros(5+k,1);
Result = zeros(3,1);


Result(1,1) = zoe(Yprd,Y1);
Result(2,1) = mae(Yprd,Y1);
Result(3,1) = RMSE(Yprd,Y1);
%Result(4,1) = precAtK(Yprd,Y1, k, cutoff);
%Result(5,1) = recallAtK(Yprd,Y1, k, cutoff);

%ndcgAtk     = ndcg_k(Yprd, Y1, k);
%Result(6:(6+k-1),1) = ndcgAtk';
%Result(6,1) = AUC(Yprd,Y1, cutoff);
%Result(4,1) = RRMSE(Yprd,Y1+Y2);
end
