clear
addpath(genpath('.'));

i = (40:-1:1)./16;
regvals = power(10,i);
%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nRun    = 3;
non0Per = 100;
tstPer  = 30;
k       = 100;
l       = 5; %Rating level
maxiter = 65;
tol     = 1e-3;
lambdaMMMF = regvals(22);
alpha   = 0.5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ttlEvaluationMetrices = 3;

ResultTrnMMMF  = zeros(ttlEvaluationMetrices,nRun);
ResultTstMMMF  = zeros(ttlEvaluationMetrices,nRun);


filename = strcat( 'resultFinal.txt');
fs = fopen(filename,'a');

fn = strcat('results/temp2.txt');
f1 = fopen(fn, 'w');

%% Data Generation
Y = load('movielens.txt');

%% data pre-processing
Y(sum(Y~=0,2)==0,:) = []; %code to delete user who has not given any rating
Y = sparse(Y);
[n,m] = size(Y);
v = randn(n*k+m*k+n*(l+1),1); %U, V and Theta

par               = {};

[Ytrn,Ytst] = divideData(Y,tstPer);

for runNo = 1:nRun
    %[Ytrn,Ytst] = divideData(Y,tstPer);
    
    %fprintf(fs,'\nrows left: %d\t column left:  %d\n',n,m);
    
    L = full(max(max(Ytrn(:),Ytst(:))));
    minRating =full(min(min(Ytrn(Ytrn>0)), min(Ytst(Ytst>0))));
    %% Maximum Margin Matrix Factorization
    %
    
    % par.lineSearchFun = @cgLineSearch;
    par.c2            = 1e-2;
    par.objGrad       = @m3fshc;
    par.softmax       = @m3fSoftmax;
    par.lambda        = lambdaMMMF;
    par.l             = L;
    par.tol           = tol;
    par.maxiter       = maxiter;
    par.p             = k;
    par.Y             = Ytrn;
    par.eta           = 1e-2;
    
    [v, numiter, J] = graddesc(v,par, f1);
    
    U                 = reshape(v(1:n*k),n,k);
    V                 = reshape(v(n*k+1:n*k+m*k),m,k);
    theta             = reshape(v(n*k+m*k+1:n*k+m*k+n*(l+1)),n,l+1);
    X                 = U*V';
    YPred             = m3fSoftmax(X,theta);

    ResultTrnMMMF(:,runNo) = EvaluationAll(YPred, Ytrn);
    ResultTstMMMF(:,runNo) = EvaluationAll(YPred, Ytst);
    fprintf(f1,'\nRun No : %d\n', runNo);
    fprintf(f1,'MMMF-CG Training Error:     ZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f\n',...
        ResultTrnMMMF(1,runNo),ResultTrnMMMF(2,runNo),ResultTrnMMMF(3,runNo));
    fprintf(f1,'MMMF-CG Testing Error:\t\tZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f\n',...
        ResultTstMMMF(1,runNo),ResultTstMMMF(2,runNo),ResultTstMMMF(3,runNo));
    %% Adding new samples to Y for next iteration
    Ytrn = newSamples(Ytrn, YPred, X, theta, par.l, alpha, f1);
    length(find(Ytrn))
    if length(find(Ytrn)) == n*m
        break;
    end
    
    fprintf(f1,'\nSize of Y = %d------------------------\n\n', length(find(Ytrn)));
    fprintf(1,'Run %d done\n', runNo);
end

ResultTrnMMMFAvg = mean(ResultTrnMMMF,2);
ResultTstMMMFAvg = mean(ResultTstMMMF,2);


fprintf(fs,'\n\nMMMF-CG Training Error:     ZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f',...
    ResultTrnMMMFAvg(1,1),ResultTrnMMMFAvg(2,1),ResultTrnMMMFAvg(3,1));
fprintf(fs,'\nMMMF-CG Testing Error:\t\tZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f',...
    ResultTstMMMFAvg(1,1),ResultTstMMMFAvg(2,1),ResultTstMMMFAvg(3,1));

fclose(fs);
fclose(f1);