function [conf, noise] = confi(Y, YPred, N, XY, theta, l, dist_mar_percent, noise_mar_percent, f1)
    %CONFI Summary of this function goes here
    %   Detailed explanation goes here
    % Y contains initial matrix with unobserved entries as zeros
    % YPred contains all predicted + initial ratings filled in
    % N contains only new predictions
    [n,m] = size(Y);
    %TBY = zeros(n,m); % temp bool Y
    conf = zeros(n,m); % storing points which cleared criteria for conf
    noise = zeros(n,m); % for marking those noise points to be removed
    
    T = zeros(n,m); % temp matrix
    
    M = zeros(n,l); % for distance between margins,
    Noise_M = zeros(n,l); % for noise, gen 10% from margins
    
    for k=1:l
        tmp = abs(theta(:,k+1) - theta(:,k));
        if k==1 | k==l
            M(:,k) = 0.10 .* tmp;
        else
            M(:,k) = 0.48 .* tmp; % Predicted values have to clear this distance from margins
        end
        Noise_M(:,k) = noise_mar_percent .* tmp; % gen 10% distance from margins
    end
    
    for k=1:l
        ltmp = theta(:,k) * ones(1,m); %left margin
        rtmp = theta(:,k+1) * ones(1,m); %right margin
        BY = zeros(n,m); % bool Y
        BY = (N==k);
        
        M_temp = (M(:,k) * ones(1,m)) .* BY; % temp matrix for comparing distance from margins
        N_temp = Noise_M(:,k) * ones(1,m); % temp matrix for comparing distance from margins for noise
        
        ltmp = abs((ltmp - XY) .* BY); % distance from left margin
        rtmp = abs((rtmp - XY) .* BY); % distance from right margin
        
        P = zeros(n,m);
        P = ((ltmp >= M_temp) & (rtmp >= M_temp)) .* BY ; % bool for points clearing margins distance
        N_P = ((ltmp < N_temp) | (rtmp < N_temp)) .* (Y>0); % bool for init points to check noise or not
        
        conf = conf + P;
        noise = noise + N_P;
        
        a=length(find(sparse(P)));
        b=length(find(sparse(BY)));
        fprintf(f1,'Rating %d : %d / %d -- %.4f\n', k, a,b, (a/b)*100);
        
        if k==1 | k==5
            PL = (ltmp < M_temp) .* BY;
            PR = (rtmp < M_temp) .* BY;
            fprintf(f1, 'Left : %d\t', length(find(sparse(PL))));
            fprintf(f1, 'Right : %d\n', length(find(sparse(PR))));
        end
        
        %length(find(sparse(conf)))
        % Minmax normalization
        %ma = full(max(max(tmp)));
        %mi = full(min(min(tmp(tmp>0))));
        
        %tmp = ((tmp-mi)./(ma-mi)) .* (tmp>0);
        %conf = conf + tmp;
        clear ltmp;clear rtmp;clear M_temp;clear BY;
    end
    
end

