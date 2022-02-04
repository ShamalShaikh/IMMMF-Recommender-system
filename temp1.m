
v = v0;
regstart = 22;
fn = strcat('results/dummy.txt');
f1 = fopen(fn, 'w');
fprintf('reg : %d', regstart);
fprintf(f1,'Reg %d : %.4f\n',regstart, regvals(regstart));
L = full(max(max(Ytrn(:),Ytst(:))));
minRating =full(min(min(Ytrn(Ytrn>0)), min(Ytst(Ytst>0))));


%% Maximum Margin Matrix Factorization
%

par.lineSearchFun = @cgLineSearch;
par.c2            = 1e-2;
par.objGrad       = @m3fshc;
par.softmax       = @m3fSoftmax;
par.lambda        = regvals(regstart);
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

fprintf(f1,'MMMF-CG Training Error:     ZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f\n',...
    ResultTrnMMMF(1,runNo),ResultTrnMMMF(2,runNo),ResultTrnMMMF(3,runNo));
fprintf(f1,'MMMF-CG Testing Error:\t\tZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f\n',...
    ResultTstMMMF(1,runNo),ResultTstMMMF(2,runNo), ResultTstMMMF(3,runNo));

fclose(f1);