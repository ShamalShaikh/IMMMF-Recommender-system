function [Rtrain,Rtest] = divideData2(R, noRating)
[n,m] = size(R);
Rtrain=zeros(n,m);
Rtest = zeros(n,m);

for userNo = 1:n
   all = find(R(userNo,:)~=0); 
   non0size = length(all);
   idx_perm = randperm(non0size);
   idx_train = all(idx_perm(1:noRating));
   idx_test = all(idx_perm(noRating+1:end));
   Rtrain(userNo,idx_train) =  R(userNo,idx_train);
   Rtest(userNo,idx_test) = R(userNo,idx_test);  
end

clear R;

Rtrain = sparse(Rtrain);
Rtest = sparse(Rtest);
end