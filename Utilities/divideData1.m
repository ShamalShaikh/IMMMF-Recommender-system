function [Rtrain,Rtest] = divideData1(R, testPer)
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

%clear R;
idxUserWith0Rating = sum(Rtrain~=0,2)==0;
Rtrain(idxUserWith0Rating,:) = Rtest(idxUserWith0Rating,:);
Rtest(idxUserWith0Rating,:)  = 0;

idxItemWith0Rating = sum(Rtrain~=0,1)==0;
Rtrain(:,idxItemWith0Rating) = Rtest(:,idxItemWith0Rating);
Rtest(:,idxItemWith0Rating)  = 0;

Rtrain = sparse(Rtrain);
Rtest  = sparse(Rtest); 
end