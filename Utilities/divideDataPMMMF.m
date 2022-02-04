function [Ytrn,Ytst] = divideDataPMMMF(Y,testPer)
[n,m] = size(Y);
%
%code to delete user who has not given rating to every class
L = max(Y(:));

allnz = find(Y);
non0size = length(allnz);
test_size = ceil(non0size * (testPer/100));
idx_perm = randperm(non0size);
idx_test = allnz(idx_perm(1:test_size));
idx_train = allnz(idx_perm(test_size+1:non0size));

Ytrn=zeros(n,m);
Ytrn(idx_train) =  Y(idx_train);
Ytst = zeros(n,m);
Ytst(idx_test) = Y(idx_test);

%

userWiseRatingTrn = false(n,L);
userWiseRatingTst = false(n,L);
for i=1:L
    userWiseRatingTrn(:,i) = sum((Ytrn == i),2);
    userWiseRatingTst(:,i) = sum((Ytst == i),2);
end
wrongSelection = ( userWiseRatingTrn == 0) &  ( userWiseRatingTst ~= 0);

for i=1:L
    for j=1:n
        if wrongSelection(j,i) == 1
            Ytrn(j,Ytst(j,:) == i) = i;
            Ytst(j,Ytst(j,:) == i) = 0;
        end
    end
end
%}


Ytrn = sparse(Ytrn);
Ytst = sparse(Ytst);

clear R;

end