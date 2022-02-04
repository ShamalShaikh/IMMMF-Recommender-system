%function [Result] = EvaluationAll(Yprd, Y1, k, cutoff)
function [Result] = EvaluationAllUpdated(Yprd, Y1, k, cutoff)

Result = {};

Result.MAE  = mae(Yprd,Y1);
Result.RMSE = RMSE(Yprd,Y1);
Result.AUC  = AUC(Yprd,Y1, cutoff);

Result.PRECISION =  precAtK(Yprd,Y1, k, cutoff);
Result.RECALL    =  recallAtK(Yprd,Y1, k, cutoff);
Result.F1        =  (2.*Result.PRECISION.*Result.RECALL)./(Result.PRECISION + Result.RECALL +eps);

Result.NDCG =  ndcgAtk(Yprd,Y1, k);

end
