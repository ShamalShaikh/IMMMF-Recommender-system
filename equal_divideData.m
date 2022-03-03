function [Rtrain,Rtest] = equal_divideData(R, testPer)
R = full(R);
[n,m] = size(R);

count = histc(R(:), [1 2 3 4 5]);
test_size = ceil(count .* (testPer/100));
Rtrain=zeros(n,m);
Rtest = zeros(n,m);
for i = 1:5
    all = find(R==i);
    idx_perm = randperm(count(i));
    idx_test = all(idx_perm(1:test_size(i)));
    idx_train = all(idx_perm(test_size(i)+1:count(i)));
    Rtrain(idx_train) =  R(idx_train);
    Rtest(idx_test) = R(idx_test);
end

Rtrain = sparse(Rtrain);
Rtest = sparse(Rtest);

clear R;

end