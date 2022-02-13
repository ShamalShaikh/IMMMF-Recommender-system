function [Rtemp,Rtest, Remain] = customDivide(R, testPer, l)

[n,m] = size(R);
F_Remain = [];
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

%Rtrain = sparse(Rtrain);
Rtest = sparse(Rtest);

mi = intmax;
for k = 1:l
    num = length(find(sparse(Rtrain==k)));
    mi = min(mi,num);
end

Rtemp = zeros(n,m);
for k=1:l
    Rtemp = Rtemp + samp((Rtrain==k), mi);
end
Remain = (Rtrain~=0) - Rtemp;
Rtemp = Rtemp .* Rtrain;
Remain = Remain .* Rtrain;
tlen = length(find(sparse(Remain)));

Rtemp = sparse(Rtemp);
Remain = sparse(Remain);
clear R;

function [Rt] = samp(conf, tot)
    [n,m] = size(conf);
    R = sparse(conf);

    all = find(R); 
    non0size = length(all);

    req_size = min(ceil(tot), non0size);

    idx_perm = randperm(non0size);
    idx_req = all(idx_perm(1:req_size));

    Rt=zeros(n,m);
    Rt(idx_req) =  R(idx_req);

