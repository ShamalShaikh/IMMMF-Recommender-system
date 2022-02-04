function [idx_rand] = select0Idx1(Ytrn, randPer)

all = find(Ytrn==0); 
ttl0size = length(all);
%test= 20;
rand_size = ceil(ttl0size * (randPer/100));

idx_perm = randperm(ttl0size);
idx_rand = all(idx_perm(1:rand_size));

end