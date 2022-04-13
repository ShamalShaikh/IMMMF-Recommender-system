load fiveK51.mat;
%  load 1M_var.mat;

% i = (50:-1:1)./16;
i = linspace(1,5,50);
regvals = linspace(5960,12500,50);

filename = strcat( 'TuneLambda/result.txt');
fs = fopen(filename,'a');

filename = strcat( 'TuneLambda/ConfusionResults.txt');
f1 = fopen(filename,'a');

filename = strcat( 'TuneLambda/train.txt');
f2 = fopen(filename,'a');

filename = strcat( 'TuneLambda/test.txt');
f3 = fopen(filename,'a');

ResultRefineTrnMMMF = zeros(ttlEvaluationMetrices,nRun,maxMMMFIter);
ResultRefineTstMMMF  = zeros(ttlEvaluationMetrices,nRun,maxMMMFIter);

% par.tol = 1e-3;
% par.c2 = 1e-2;
% par.v0     = v0;
% par.maxiter = 200;
% par.p = 100;
for r = 1:50
par.lambda = regvals(r);
% conjgrad
    [yMMMF, U , V ,mmmfTheta]   = mmmfWeak(Ytrn, par);
    ResultRefineTrnMMMF(:,runNo,r) = EvaluationAll(yMMMF, Ytrn);
    ResultRefineTstMMMF(:,runNo,r) = EvaluationAll(yMMMF, Ytst);
    
    fprintf(fs,'\n\nMMMF-CG Training Error:     ZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f',...
        ResultRefineTrnMMMF(1,runNo,r),ResultRefineTrnMMMF(2,runNo,r),ResultRefineTrnMMMF(3,runNo,r));
    fprintf(fs,'\nMMMF-CG Testing Error:\t\tZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f',...
        ResultRefineTstMMMF(1,runNo,r),ResultRefineTstMMMF(2,runNo,r),ResultRefineTstMMMF(3,runNo,r));
    
    % Printing ZOE for each iteration
    print_zoe_each(yMMMF,Ytrn,L,f2,ResultRefineTrnMMMF,r,par.lambda);
    print_zoe_each(yMMMF,Ytst,L,f3,ResultRefineTstMMMF,r,par.lambda);

    confusion_mat(Ytrn,yMMMF,L,f1, 'Training set vs Predicted Ratings');
    confusion_mat(Ytst,yMMMF,L,f1, 'Testing set vs Predicted Ratings');
end