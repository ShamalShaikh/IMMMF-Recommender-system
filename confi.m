function [Y, conf, noise] = confi(Y, YPred, N, XY, theta, par, dist_mar_percent, noise_mar_percent, f1)
    %CONFI Summary of this function goes here
    %   Detailed explanation goes here
    % Y contains initial matrix with unobserved entries as zeros
    % YPred contains all predicted + initial ratings filled in
    % N contains only new predictions
    l = par.l; % max rating
    [n,m] = size(Y);
    %TBY = zeros(n,m); % temp bool Y
    conf = zeros(n,m); % storing points which cleared criteria for conf
    noise = zeros(n,m); % for marking those noise points to be removed
    
    T = zeros(n,m); % temp matrix
    Yt = YPred .* (Y~=0); % New Predictions for values in train set
    
    M = zeros(n,l); % for distance between margins,
    Noise_M = zeros(n,l); % for noise, gen 10% from margins
    
    for k=1:l
        tmp = abs(theta(:,k+1) - theta(:,k));
        if k==1
            M(:,k) = 0.2 .* tmp;
%               M(:,k) = 0.5 .* tmp;
        elseif k==2
            M(:,k) = 0.3 .* tmp;
%               M(:,k) = 0.45 .* tmp;
        elseif k==3
            M(:,k) = 0.4 .* tmp;
%               M(:,k) = 0.45 .* tmp;
        elseif k==4
            M(:,k) = 0.499 .* tmp;
%               M(:,k) = 0.45 .* tmp;
        else
            M(:,k) = 0.45 .* tmp; % Predicted values have to clear this distance from margins
        end
        Noise_M(:,k) = noise_mar_percent .* tmp; % gen 10% distance from margins
    end
    
    for k=1:l
        ltmp = theta(:,k) * ones(1,m); %left margin
        rtmp = theta(:,k+1) * ones(1,m); %right margin
        
        BY = (Y==k & Yt==k);
        BN = (N==k);
        
        M_temp = (M(:,k) * ones(1,m)) .* BN; % temp matrix for comparing distance from margins
        N_temp = (Noise_M(:,k) * ones(1,m)) .* BY; % temp matrix for comparing distance from margins for noise
        
        ltmp = abs((ltmp - XY) ); % distance from left margin
        rtmp = abs((rtmp - XY) ); % distance from right margin
        
        %P = zeros(n,m);
        if k==1
            P = (rtmp >= M_temp) .* BN ;
            N_P = (rtmp < N_temp) .* BY;
        elseif k==5
            P = (ltmp >= M_temp) .* BN ;
            N_P = (ltmp < N_temp) .* BY;
        else
            P = ((ltmp >= M_temp) & (rtmp >= M_temp)) .* BN ; % bool for points clearing margins distance
            N_P = ((ltmp < N_temp) | (rtmp < N_temp)) .* BY; % bool for init points to check noise or not
        end
        
        %{
        if k==1
            Rt = sampling(P,0.01);
        elseif k==2
            Rt = sampling(P,0.01);
        elseif k==3 | k==4
            Rt = sampling(P,0.002);
        else
            Rt = sampling(P,0.001);
        end
        %}
        Rt = samp(P, par.ratio(k,1), par.new_add);
        N_P = sampling(N_P, 0.01); % Taking 10% of noise points
        
        conf = conf + Rt;
        noise = noise + N_P;
        
        a=length(find(sparse(P)));
        b=length(find(sparse(BN)));
        rt=length(find(sparse(Rt)));
        fprintf(f1,'Rating %d : %d / %d -- %.4f -- %d added\n', k, a,b, (a/b)*100, rt);
        fprintf(f1,'Noise %d : %d \n', k, length(find(sparse(N_P))));
        
        clear ltmp;clear rtmp;clear M_temp;clear BY;
    end
    
function [Rt] = sampling(conf, per)
    [n,m] = size(conf);
    R = sparse(conf);

    all = find(R); 
    non0size = length(all);

    req_size = min(ceil(non0size * per),non0size);

    idx_perm = randperm(non0size);
    idx_req = all(idx_perm(1:req_size));

    Rt=zeros(n,m);
    Rt(idx_req) =  R(idx_req);

function [Rt] = samp(conf, ratio, tot)
    [n,m] = size(conf);
    R = sparse(conf);

    all = find(R); 
    non0size = length(all);

    req_size = min(ceil(tot * (1/ratio)),non0size);

    idx_perm = randperm(non0size);
    idx_req = all(idx_perm(1:req_size));

    Rt=zeros(n,m);
    Rt(idx_req) =  R(idx_req);
