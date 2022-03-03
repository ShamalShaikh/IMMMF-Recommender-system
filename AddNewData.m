function Ytrn = AddNewData(Ytrn, Yimpute, ratios,L,IterAdd)
    req_size = round(((1 - ratios)./sum(1 - ratios)) .* IterAdd);
    for r = 1:L
        all = find(Yimpute==r & Ytrn==0); 
        rsize = length(all);
        idx_perm = randperm(rsize);
        idx_req = all(idx_perm(1:req_size(r)));
        Ytrn(idx_req) = Yimpute(idx_req);  
    end
end