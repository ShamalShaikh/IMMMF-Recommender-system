function [Ynew] = newSamples(Y, YPred, XY, theta, l, alpha)
%NEWSAMPLES Summary of this function goes here
%   Detailed explanation goes here
[n,m] = size(Y);
Ynew = zeros(n,m);
S = (Y==0);     % boolean matrix of new predictions
N = YPred .* S; % contains all new predictions
conf = confi(N, XY, theta, l);
BCF = (conf > alpha);
R = sparse(BCF);


all = find(R); 
non0size = length(all);

req_size = ceil(non0size * 0.1);

idx_perm = randperm(non0size);
idx_req = all(idx_perm(1:req_size));

Rt=zeros(n,m);
Rt(idx_req) =  R(idx_req);

Ynew = Y + (Rt.*YPred) ;
%Ynew = sparse(Ynew);
end

