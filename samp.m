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
end