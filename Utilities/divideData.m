function [Rtrain,Rtest] = divideData(R, testPer)

[n,m] = size(R);
all = find(R); 
non0size = length(all);
%test= 20;
test_size = ceil(non0size * (testPer/100));

idx_perm = randperm(non0size);
idx_test = all(idx_perm(1:test_size));
idx_train = all(idx_perm(test_size+1:non0size));

Rtrain=zeros(n,m);
Rtrain(idx_train) =  R(idx_train);
Rtest = zeros(n,m);
Rtest(idx_test) = R(idx_test);

Rtrain = sparse(Rtrain);
Rtest = sparse(Rtest);

clear R;

end