function [Ynew] = newSamples(iter, Y, Ytst, YPred, YPredPrev, XY, theta, par, alpha, f1)
%NEWSAMPLES Summary of this function goes here
%   Detailed explanation goes here
[n,m] = size(Y);

% S1 = (Y==0);
S = (Y==0);
% S2 = (Ytst ~= 0);
% S = S1-S2;
N = YPred .* S; % contains all new predictions excluding Ytst
% Yt = YPred .* (Ytst~=0); % contains all the new predictions in Ytst
% YPrev = YPredPrev .* (Yt>0);

[Y, conf, noise] = confi(Y, YPred, N, XY, theta, par, 0.45, 0.40, f1); 

%{
wei = wei + (Y ~= Yt);
if iter > 1
    cor = (Yt == YPrev & Yt>0 & Yt~= Y);
    
    cor = sampling(cor,0.1);
    all = find(sparse(cor));
    conf = conf | cor;
    fprintf(1, "Corrected: %d",length(all));
    
end
%}
% last two params confidence and noise distance from margins


a=length(find(sparse(conf)));
b=length(find(sparse(noise)));
fprintf(f1,'Conf : %d \n', a);
fprintf(f1,'Noise : %d \n', b);
Ynew = Y .* (noise ~= 1) + (conf.*YPred) ;
Ynew = sparse(Ynew);

function [Rt] = sampling(conf, per)
    [n,m] = size(conf);
    R = sparse(conf);

    all = find(R); 
    non0size = length(all);

    req_size = ceil(non0size * per);

    idx_perm = randperm(non0size);
    idx_req = all(idx_perm(1:req_size));

    Rt=zeros(n,m);
    Rt(idx_req) =  R(idx_req);



