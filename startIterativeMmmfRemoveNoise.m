clear
addpath(genpath('.'));

i = (40:-1:1)./16;
regvals = power(10,i);
%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nRun    = 1;
nRows   = 20;
nColumns= 20;
non0Per = 60;
tstPer  = 30;
k       = 100;
L       = 5;
maxiter = 50;
tol     = 1e-3;
% lambdaIterMMMF = [regvals(40), regvals(35), regvals(30),regvals(28),regvals(25), regvals(20), regvals(15),regvals(10),regvals(5),regvals(1)];
% lambdaIterMMMF = [16.5, 16.5, 16.5, 16.5, 16.5, 16.5, 16.5, 16.5, 16.5, 16.5];
lambdaIterMMMF = repmat(16.5,50,1);
% lambdaIterMMMF = linspace();
lambdaMMMF = regvals(21);
randPerFill = 1;
rank = 1000;
IterAdd = 1000;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

maxMMMFIter       = 50;
% marginPerFromBound = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]; %at 50 left and right boundary will be same
marginPerFromBound = repmat(10,50,1);
% expMarginPerForCenter = [49.8, 49.8, 49.8, 48.9, 49.8, 49.8, 49.8, 49.8, 48.9, 49.8]; %at 50 left and right boundary will be same
expMarginPerForCenter = repmat(49.8,50,1);
YperIter = cell(1,maxMMMFIter);

ttlEvaluationMetrices = 3;

ResultRefineTrnMMMF = zeros(ttlEvaluationMetrices,nRun,maxMMMFIter);
ResultRefineTstMMMF  = zeros(ttlEvaluationMetrices,nRun,maxMMMFIter);


ResultIterTrnMMMF = zeros(ttlEvaluationMetrices,nRun,maxMMMFIter);
ResultIterTstMMMF  = zeros(ttlEvaluationMetrices,nRun,maxMMMFIter);


filename = strcat( 'Result/resultFinal_equal.txt');
fs = fopen(filename,'a');

filename = strcat( 'results/testingNewAdd_13.txt');
f1 = fopen(filename,'w');

fprintf(f1,'Removing 1000 worst points based on confi ratios for first 40 iterations, train-test split based on distribution\n');

for runNo = 1:nRun
    %% Data Generation
    Y = load('movielens.txt');
    %Y = load('movielens_1M_matrix.txt');
    %Y = generateData(nRows,nColumns,non0Per);
    %fprintf(fs,'\n\nrows:      %d\t column:       %d\t\t non0:  %d',size(Y,1),size(Y,2),non0Per);
    %% data pre-processing
    Y(sum(Y~=0,2)==0,:) = []; %code to delete user who has not given any rating
    Y = sparse(Y);
    [Ytrn,Ytst] = equal_divideData(Y,tstPer);
%     [Ytrn,Ytst] = divideDataPMMMF(Y,tstPer);
    %[Ytrn, Ytst, ~, ~] = allBut1Division(Y, []);
    [n,m] = size(Ytrn);
    %fprintf(fs,'\nrows left: %d\t column left:  %d\n',n,m);
    L = full(max(max(Ytrn(:),Ytst(:))));
    minRating =full(min(min(Ytrn(Ytrn>0)), min(Ytst(Ytst>0))));
    %% Maximum Margin Matrix Factorization
    %
    par = {};
    par.lineSearchFun = @cgLineSearch;  par.c2 = 1e-2;
    par.objGrad = @m3fshc;
    par.l = L;                          par.tol = tol;
    par.maxiter = maxiter;              par.p = k;
    
    v0 = randn(n*k+m*k+n*(L-1),1);
    for iter = 1:maxMMMFIter
        fprintf(1,"Iteration %d\n",iter);
        YperIter{1,iter} = Ytrn;
        par.lambda = lambdaIterMMMF(iter);
        par.v0     = v0;
        
        % conjgrad
        [yMMMF, U , V ,mmmfTheta]   = mmmfWeak(Ytrn, par);
        ResultRefineTrnMMMF(:,runNo,iter) = EvaluationAll(yMMMF, Ytrn);
        ResultRefineTstMMMF(:,runNo,iter) = EvaluationAll(yMMMF, Ytst);

        fprintf(f1,'\n\nMMMF-CG Training Error:     ZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f',...
            ResultRefineTrnMMMF(1,runNo,iter),ResultRefineTrnMMMF(2,runNo,iter),ResultRefineTrnMMMF(3,runNo,iter));
        fprintf(f1,'\nMMMF-CG Testing Error:\t\tZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f',...
            ResultRefineTstMMMF(1,runNo,iter),ResultRefineTstMMMF(2,runNo,iter),ResultRefineTstMMMF(3,runNo,iter));
        

        confusion_mat(Ytrn,yMMMF,L,f1, 'Training set vs Predicted Ratings');
        confusion_mat(Ytst,yMMMF,L,f1, 'Testing set vs Predicted Ratings');
%         confusion_mat(yMMMF,Ytst, L, fs);
        if iter <= 40

%         Count the ratio = (no. of (r-r) rating / total no. of r in train set)
        ratios = CalculateConfi(Ytrn,yMMMF,L);

%         Remove points which are at max distance from the rating region.
%         Yiter = removeDistNoise(U*V',Ytrn,mmmfTheta, rank,ratios);
        
%         confusion_mat(Yiter,Ytrn,L,f1, 'Ratings before removing noise vs Train set');

%       Removes Noise - values in the 20 range from the margin
%         [Yiter] = refineData(U*V', mmmfTheta, marginPerFromBound(iter));

%         Ytrn = Yiter .* (Ytrn~=0);
%         Ytrn = Yiter.*((YperIter{1,iter}==Ytrn) & (YperIter{1,iter}~=0)); 
        
%         confusion_mat(Ytrn,yMMMF,L,f1, 'Training set after removing noise vs Predicted Ratings');


%         confusion_mat(Ytrn,Yiter,L,f1, 'Train set vs Ratings before removing noise');

%         Ytrn = removeNoise(Yiter,Ytrn,L);

%         confusion_mat(Ytrn,Yiter,L,f1, 'Train set vs Ratings after removing noise');
        
        
        [Yimpute] = refineData(U*V',mmmfTheta, expMarginPerForCenter(iter));               
        
        confusion_mat(Ytrn,Yimpute,L,f1, 'Train set vs Yimpute (top ranked data)');
        confusion_mat(Ytst,Yimpute,L,f1, 'Test set vs Yimpute (top ranked data)');

%         Add new data points based on the ratios 
        Ytrn = AddNewData(Ytrn, Yimpute, ratios,L,IterAdd);

        confusion_mat(Ytrn,Yimpute,L,f1, 'Train set vs Yimpute (top ranked data)');

        end
%         Ytrn( Ytrn==0)     = Yimpute(Ytrn==0);

%         confusion_mat(Ytrn,Ytst, L, f1, 'New Training set vs Testing set');

        %[idx_rand] = select0Idx(Ytrn, Yiter, randPerFill);        
        %Ytrn(idx_rand) = Yiter(idx_rand);
        
%         v0 = [U(:); V(:); mmmfTheta(:)];
        
%          for rating=1:L
%             original = full(sum(sum(YperIter{1,iter}==rating)));
%             retained = full(sum(sum((YperIter{1,iter}==Ytrn).*YperIter{1,iter}==rating)));
%             perRetained = (retained/original)*100;
%             fprintf('Rating = %d \t original = %d \t retained = %d \t perRetained = %.2f \n',rating,original,retained,perRetained)
%         end
    end
    
%     v0 = randn(n*k+m*k+n*(L-1),1);
%     for iter = 1:maxMMMFIter
%         par.v0     = v0;       
%         par.lambda = lambdaMMMF;        
%         [yMMMF, U , V ,mmmfTheta]   = mmmfWeak(YperIter{1,iter}, par);
%         ResultIterTrnMMMF(:,runNo,iter) = EvaluationAll(yMMMF, Ytrn);
%         ResultIterTstMMMF(:,runNo,iter) = EvaluationAll(yMMMF, Ytst);
%         
%         %v0 = [U(:); V(:); mmmfTheta(:)];
%     end
    
    %}
end


ResultTrnMMMFAvg = mean(ResultRefineTrnMMMF,2);
% ResultTstMMMFAvg = mean(ResultIterTstMMMF,2);
ResultTstMMMFAvg = mean(ResultRefineTstMMMF,2);

fprintf(fs,'\n\nMMMF-CG Training Error:     ZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f',...
    ResultTrnMMMFAvg(1,1),ResultTrnMMMFAvg(2,1),ResultTrnMMMFAvg(3,1));
fprintf(fs,'\nMMMF-CG Testing Error:\t\tZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f',...
    ResultTstMMMFAvg(1,1),ResultTstMMMFAvg(2,1),ResultTstMMMFAvg(3,1));
