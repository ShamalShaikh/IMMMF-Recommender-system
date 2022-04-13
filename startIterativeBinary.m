% clear
addpath(genpath('.'));

i = (40:-1:1)./16;
regvals = power(10,i);
%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nRun    = 1;
tstPer  = 30;
k       = 100;
L       = 5;
maxiter = 50;
tol     = 1e-3;
lambdaMMMF = regvals(21);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

maxMMMFIter       = 10;
YperIter = cell(1,maxMMMFIter);

ttlEvaluationMetrices = 3;

ResultRefineTrnMMMF = zeros(ttlEvaluationMetrices,nRun,maxMMMFIter);
ResultRefineTstMMMF  = zeros(ttlEvaluationMetrices,nRun,maxMMMFIter);


ResultIterTrnMMMF = zeros(ttlEvaluationMetrices,nRun,maxMMMFIter);
ResultIterTstMMMF  = zeros(ttlEvaluationMetrices,nRun,maxMMMFIter);


filename = strcat( 'Biresults/resultFinal_equal.txt');
fs = fopen(filename,'a');

filename = strcat( 'Biresults/result_3.txt');
f1 = fopen(filename,'a');

filename = strcat( 'Biresults/ZOE_each/train3.txt');
f2 = fopen(filename,'a');

filename = strcat( 'Biresults/ZOE_each/test3.txt');
f3 = fopen(filename,'a');

fprintf(f1,'Binary HMF - Training for 1 vs {2,3,4and5}\n');

for runNo = 1:nRun
    %% Data Generation
    Y = load('movielens.txt');
%     Y = load('movielens_1M_matrix.txt');
    %% data pre-processing
    Y(sum(Y~=0,2)==0,:) = []; %code to delete user who has not given any rating
    Y = sparse(Y);
    [Ytrn,Ytst] = equal_divideData(Y,tstPer);
    [n,m] = size(Ytrn);
    L = full(max(max(Ytrn(:),Ytst(:))));
    minRating =full(min(min(Ytrn(Ytrn>0)), min(Ytst(Ytst>0))));
    %% Maximum Margin Matrix Factorization
    %
    par = {};
    par.lineSearchFun = @cgLineSearch;  par.c2 = 1e-2;
    par.objGrad = @m3fshcBinary;
    par.l = L;                          par.tol = tol;
    par.maxiter = maxiter;              par.p = k;
    par.ratingLevel = L;
    par.increment = 1;
    par.minRating = 1;
    par.IterAdd = IterAdd;
    
    %% ZOE PRINTING SECTION
    fprintf(f2,'\n\n Training Error: \n It# \t\t 1 \t\t\t2 \t\t\t 3 \t\t\t 4 \t\t\t5 \t\t\tZOE \t\tMAE\t\tRMSE');
    fprintf(f3,'\n\n Testing Error: \n It# \t\t 1 \t\t\t2 \t\t\t 3 \t\t\t 4 \t\t\t 5 \t\t\tZOE \t\tMAE \t\t RMSE');
    
    %%
    v0 = randn(n*k+m*k,1);
    lvec = linspace(0.1,3,10);
    lambda = [zeros(10,1), lvec', repmat([10,7.5],10,1)];
    for iter = 1:maxMMMFIter
        fprintf(1,"Iteration %d\n",iter);
        YperIter{1,iter} = Ytrn;
%         par.lambda = lambdaIterMMMF(iter);
        par.lambdaHMF = lambda(iter,:);
%         par.lambdaHMF = [lvec',6.5,10,7.5];
        
        par.v0     = v0;
        
        % conjgrad
        [RFinal,ULevelWise,VLevelWise,hmf_Lvl_trn_err,hmf_Lvl_tst_err,hmf_fnl_trn_err,hmf_fnl_tst_err] = BiLevelWeak(Ytrn, Ytst, par);

        fprintf(f1,'\n\nMMMF-CG Training Error:     ZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f',...
            hmf_fnl_trn_err(1),hmf_fnl_trn_err(2),hmf_fnl_trn_err(3));
        fprintf(f1,'\nMMMF-CG Testing Error:\t\tZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f',...
            hmf_fnl_tst_err(1),hmf_fnl_tst_err(2),hmf_fnl_tst_err(3));
        
        % Printing ZOE for each iteration
        print_zoe_each(RFinal,Ytrn,L,f2,hmf_fnl_trn_err(1:3),iter, par.lambdaHMF);
        print_zoe_each(RFinal,Ytst,L,f3,hmf_fnl_tst_err(1:3),iter, par.lambdaHMF);

    end

end


fprintf(fs,'\n\nMMMF-CG Training Error:     ZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f',...
    hmf_fnl_trn_err(1),hmf_fnl_trn_err(4),hmf_fnl_trn_err(5));
fprintf(fs,'\nMMMF-CG Testing Error:\t\tZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f',...
    hmf_fnl_tst_err(1),hmf_fnl_tst_err(4),hmf_fnl_tst_err(5));
