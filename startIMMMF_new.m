%%
% Created by : Shamal Shaikh
% Year : 2021 - 2022
% Modified version of startIMMMF
%%

clear
addpath(genpath('.'));

i = (40:-1:1)./16;
%regvals = power(10,i);
%regvals = linspace(3.6,4,40);
regvals = linspace(17.7828,13.3352,40);
%regvals = linspace(1,0,40);
%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nRun    = 1;
nIter  = 5;
non0Per = 100;
tstPer  = 30;
k       = 100;
l       = 5; %Rating level
maxiter = 110;
tol     = 1e-4;
%regstart = 22;
%lambdaMMMF = regvals(regstart);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ttlEvaluationMetrices = 3;

ResultTrnIMMMF  = zeros(ttlEvaluationMetrices,nIter);
ResultTstIMMMF  = zeros(ttlEvaluationMetrices,nIter);

filename = strcat( 'final/final-mmmf-f.txt');
fs = fopen(filename,'a');

%% Data Generation
Y = load('movielens.txt');
%Y = load('movielens_1M_matrix.txt');

%% data pre-processing
% Y(sum(Y~=0,2)==0,:) = []; %code to delete user who has not given any rating
Y = sparse(Y);
[n,m] = size(Y);
%v = randn(n*k+m*k+n*(l+1),1); % initialise U, V and Theta randomly

par               = {};
interTrn  = zeros(ttlEvaluationMetrices, nRun, nIter);
interTst  = zeros(ttlEvaluationMetrices, nRun, nIter);

newadd = 100;
%[Ytrn,Ytst] = divideData(Y,tstPer);
for regstart = 1:1
    fn = strcat('final-f-',num2str(regstart),'-',num2str(regvals(regstart)));
    fh = strcat('results/', fn, '.txt');
    f1 = fopen(fh, 'w');
    for runNo = 1:nRun
        [Ytrn,Ytst] = divideData(Y,tstPer);
        YPredPrev = zeros(n,m);
        
        fprintf(f1,'Run No : %d\t Reg: %d\t Val: %.4f\n', runNo, regstart, regvals(regstart));
        for iterNo = 1:nIter
            v = randn(n*k+m*k+n*(l+1),1); % initialise U, V and Theta randomly
            L = full(max(max(Ytrn(:),Ytst(:))));
            minRating =full(min(min(Ytrn(Ytrn>0)), min(Ytst(Ytst>0))));
    
            ratios = zeros(L,1);
            tot = length(find(Ytrn));
            for rat = 1:L
                ratios(rat,1) = length(find(Ytrn==rat))/tot;
            end
    
            %[Ytrn,Ytst] = divideData(Y,tstPer);
    
            %fprintf(fs,'\nrows left: %d\t column left:  %d\n',n,m);
            %lambdaMMMF = regvals(20 + runNo);
            %% Maximum Margin Matrix Factorization
            %
            par.ratio         = ratios;
            par.new_add       = newadd;
            par.top           = 25;
    
            par.lineSearchFun = @cgLineSearch;
            par.c2            = 1e-2;
            par.objGrad       = @m3fshc;
            %par.objGrad       = @m3fshc_nnorm;
            par.softmax       = @m3fSoftmax;
            par.lambda        = regvals(regstart);
            %par.lambda        = 0.2422;
            par.l             = L;
            par.tol           = tol;
            par.maxiter       = maxiter;
            par.p             = k;
            par.Y             = Ytrn;
            par.eta           = 1e-2;
            %[v, numiter, ogcalls, J] = conjgrad(v,par,f1);
            
            [v, numiter, J] = graddesc(v,par, f1);
    
            U                 = reshape(v(1:n*k),n,k);
            V                 = reshape(v(n*k+1:n*k+m*k),m,k);
            theta             = reshape(v(n*k+m*k+1:n*k+m*k+n*(l+1)),n,l+1);
            X                 = U*V';
            YPred             = m3fSoftmax(X,theta);
    
            interTrn(:,runNo,iterNo) = EvaluationAll(YPred, Ytrn);
            interTst(:,runNo,iterNo) = EvaluationAll(YPred, Ytst);
            fprintf(f1,'\nIteration No : %d\n', iterNo);
            mae_r = zeros(L,1);
            rmse_r = zeros(L,1);
            for rat = 1:L
                mae_r(rat,1) = mae_each(YPred, Ytst, rat);
                rmse_r(rat,1) = rmse_each(YPred, Ytst, rat);
                fprintf(f1, '%d -> MAE = %.4f\t\tRMSE = %.4f\n', rat, mae_r(rat,1), rmse_r(rat,1));
            end
            fprintf(f1,'MMMF-CG Training Error:     ZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f\n',...
                interTrn(1,runNo,iterNo),interTrn(2,runNo,iterNo),interTrn(3,runNo,iterNo));
            fprintf(f1,'MMMF-CG Testing Error:\t\tZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f\n',...
                interTst(1,runNo,iterNo),interTst(2,runNo,iterNo), interTst(3,runNo,iterNo));
            wrong(Ytrn, Ytst, YPred, par.l, f1);
            
            %% Adding new samples to Y for next iteration
            
            Ytrn = newSamples(iterNo, Ytrn, Ytst, YPred, YPredPrev, X, theta, par, f1);
            if length(find(Ytrn)) == n*m
                break;
            end
    
            fprintf(f1,'\nSize of Y = %d------------------------\n', length(find(Ytrn)));
            %}
            fprintf(1,'\nReg: %d\tRun: %d\tIteration %d done\n', regstart, runNo, iterNo);
            fprintf(1,'MMMF-CG Testing Error:\t\tZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f\n',...
                interTst(1,runNo,iterNo),interTst(2,runNo,iterNo), interTst(3,runNo,iterNo));
            YPredPrev = YPred;
            %figure, geometricalMMMF(Ytrn,U,V, theta);
            %geometricalMMMF(Ytst,U,V, theta);
        end

    end

    ResultTrnIMMMFAvg = mean(interTrn,2);
ResultTstIMMMFAvg = mean(interTst,2);

make_plot(nIter, ResultTstIMMMFAvg,fn);

fprintf(fs, '\nReg : %d\tVal: %.4f\n', regstart, regvals(regstart));
fprintf(fs,'Training Error:\n');
for iterNo = 1:nIter
    fprintf(fs,'IMMMF-CG Training Error:     ZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f\n',...
        ResultTrnIMMMFAvg(1,1,iterNo),ResultTrnIMMMFAvg(2,1,iterNo),ResultTrnIMMMFAvg(3,1,iterNo));
end
fprintf(fs,'\nTesting Error:\n');
for iterNo = 1:nIter
    fprintf(fs,'IMMMF-CG Testing Error:\t\tZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f\n',...
        ResultTstIMMMFAvg(1,1,iterNo),ResultTstIMMMFAvg(2,1,iterNo),ResultTstIMMMFAvg(3,1,iterNo));
end
fclose(f1);
%fh = strcat('vals/',fn);
%save(fh);
end
    

fclose(fs);
