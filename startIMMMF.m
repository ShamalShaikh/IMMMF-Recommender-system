clear
addpath(genpath('.'));

i = (40:-1:1)./16;
regvals = power(10,i);
%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nRun    = 5;
non0Per = 100;  % ??
tstPer  = 30;
k       = 100;
l       = 5; %Rating level
maxiter = 50;
tol     = 1e-3;
regstart = 25;
lambdaMMMF = regvals(regstart);
alpha   = 0.5;  % ??
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ttlEvaluationMetrices = 3;

ResultTrnMMMF  = zeros(ttlEvaluationMetrices,nRun);
ResultTstMMMF  = zeros(ttlEvaluationMetrices,nRun);


filename = strcat( 'resultFinal.txt');
fs = fopen(filename,'a');

fn = strcat('results/expt_3222_12.txt');
f1 = fopen(fn, 'w');

%% Data Generation
Y = load('movielens.txt');
% Y = [3 0 0 0 1; 1 2 0 1 3; 0 1 0 3 0; 4 3 5 0 4];
%% data pre-processing
Y(sum(Y~=0,2)==0,:) = []; %code to delete user who has not given any rating
Y = sparse(Y);
[n,m] = size(Y);
v = randn(n*k+m*k+n*(l+1),1); %U, V and Theta

par               = {};
[Ytrn,Ytst] = divideData(Y,tstPer);
% [Ytrn,Ytst,Remain] = customDivide(Y,tstPer,l);
% tlen = length(find(sparse(Remain)));
% fprintf(f1,'Size of Ytrn : %d\n', length(find(Ytrn)));
% fprintf(f1,'Size of Ytst : %d\n', length(find(Ytst)));
% fprintf(f1,'Size of Remain : %d\n', length(find(Remain)));
%wei = zeros(n,m);
%cost = zeros(n,m);
YPredPrev = zeros(n,m);
count_rat(Ytst,l);
for runNo = 1:nRun
        if runNo > 1
            maxiter = 50;
        end
%     add = samp(full(Remain), tlen/l);
%     count_rat(add, l, f1);
%     Remain = Remain - sparse(add);
%     fprintf(f1,'Size of add : %d\n', length(find(sparse(add))));
%     fprintf(f1,'Size of Remain : %d\n', length(find(Remain)));
%     Ytrn = Ytrn + add;
%     fprintf(f1,'Size of Y : %d\n', length(find(Ytrn)));
%     L = full(max(max(Ytrn(:),Ytst(:))));
    L = 5;
    minRating =full(min(min(Ytrn(Ytrn>0)), min(Ytst(Ytst>0))));
    
    ratios = zeros(L,1);
    tot = length(find(Ytrn));
    for rat = 1:L
        ratios(rat,1) = length(find(Ytrn==rat))/tot;
    end
    par.ratio = ratios;
    par.new_add = 100;
    fprintf(1,'Iteration %d start\n', runNo);
    count_rat(Ytrn,l);
    
    %fprintf(fs,'\nrows left: %d\t column left:  %d\n',n,m);
    %lambdaMMMF = regvals(20 + runNo);
    %% Maximum Margin Matrix Factorization
    %
    
    par.lineSearchFun = @cgLineSearch;
    par.c2            = 1e-2;
    par.objGrad       = @m3fshc;
    par.softmax       = @m3fSoftmax;
%     par.lambda        = regvals(regstart);
    par.lambda        = lambdaMMMF;
    par.l             = L;
    par.tol           = tol;
    par.maxiter       = maxiter;
    par.p             = k;
    par.Y             = Ytrn;
    par.eta           = 1e-2;
    
    [v, numiter, ogcalls, J] = conjgrad(v,par);
    %[v, numiter, J] = graddesc(v,par, f1);
    
    U                 = reshape(v(1:n*k),n,k);
    V                 = reshape(v(n*k+1:n*k+m*k),m,k);
    theta             = reshape(v(n*k+m*k+1:n*k+m*k+n*(l+1)),n,l+1);
    X                 = U*V';
    YPred             = m3fSoftmax(X,theta);

    ResultTrnMMMF(:,runNo) = EvaluationAll(YPred, Ytrn);
    ResultTstMMMF(:,runNo) = EvaluationAll(YPred, Ytst);
    fprintf(f1,'\nRun No : %d\n', runNo);
    mae_r = zeros(L,1);
    rmse_r = zeros(L,1);
    for rat = 1:L
        mae_r(rat,1) = mae_each(YPred, Ytst, rat);
        rmse_r(rat,1) = rmse_each(YPred, Ytst, rat);
        fprintf(f1, '%d -> MAE = %.4f\t\tRMSE = %.4f\n', rat, mae_r(rat,1), rmse_r(rat,1));
    end
    fprintf(f1,'MMMF-CG Training Error:     ZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f\n',...
        ResultTrnMMMF(1,runNo),ResultTrnMMMF(2,runNo),ResultTrnMMMF(3,runNo));
    fprintf(f1,'MMMF-CG Testing Error:\t\tZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f\n',...
        ResultTstMMMF(1,runNo),ResultTstMMMF(2,runNo), ResultTstMMMF(3,runNo));
    
    wrong(Ytrn, Ytst, YPred, par.l, f1);
    %% Adding new samples to Y for next iteration
    
    Ytrn = newSamples(runNo, Ytrn, Ytst, YPred, YPredPrev, X, theta, par, alpha, f1);
    %{
    length(find(Ytrn))
    if length(find(Ytrn)) == n*m
        break;
    end
    
    fprintf(f1,'\nSize of Y = %d------------------------\n\n', length(find(Ytrn)));
    %}
    fprintf(1,'Iteration %d done\n', runNo);
    count_rat(Ytrn,l);
    YPredPrev = YPred;
    
end
ResultTrnMMMFAvg = mean(ResultTrnMMMF,2);
ResultTstMMMFAvg = mean(ResultTstMMMF,2);

subplot(3,1,1);
plot(1:nRun,ResultTstMMMF(1,:));
subplot(3,1,2);
plot(1:nRun,ResultTstMMMF(2,:));
subplot(3,1,3);
plot(1:nRun,ResultTstMMMF(3,:));

fprintf(fs,'\n\nMMMF-CG Training Error:     ZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f',...
    ResultTrnMMMFAvg(1,1),ResultTrnMMMFAvg(2,1),ResultTrnMMMFAvg(3,1));
fprintf(fs,'\nMMMF-CG Testing Error:\t\tZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f',...
    ResultTstMMMFAvg(1,1),ResultTstMMMFAvg(2,1),ResultTstMMMFAvg(3,1));

fclose(fs);
fclose(f1);

function [Rt] = samp(conf, tot)
    [n,m] = size(conf);
    R = sparse(conf);

    all = find(R); 
    non0size = length(all);

    req_size = min(ceil(tot), non0size);

    idx_perm = randperm(non0size);
    idx_req = all(idx_perm(1:req_size));

    Rt=zeros(n,m);
    Rt(idx_req) =  R(idx_req);
end

% function [val] = minimum(a,b)
%     if a<b
%         val = a;
%     else
%         val = b;
%     end
% end