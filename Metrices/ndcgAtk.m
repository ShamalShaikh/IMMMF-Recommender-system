function res=ndcgAtk(Ypre, Yt, k)
%Yt  : Test Set of size n *m
%Ypre: Prediction set of size n *m
%cutoff : threshold for relevant item
%k: precision@k
%%

% userWithRating = sum(Yt~=0,2)>0;
% Yt             = Yt(userWithRating,:);
% Ypre           = Ypre(userWithRating,:);
% [N,~]          = size(Yt);

res = zeros(1, k);
cnt = 0;
[N,~]          = size(Yt);


for user=1:N

    ratedIdx            = Yt(user,:)~=0;
    userRating          = Yt(user,ratedIdx);
    userPrediction      = Ypre(user,ratedIdx);
    nz                  = length(userRating);
    
    ranks               = zeros(1, nz);
    ideal_ranks         = zeros(1, nz);
    
    
    [~, I]              = sort(userPrediction, 2, 'descend');    
    [~, ideal_I]        = sort(userRating, 2, 'descend');
    

    ranks(I)   = 1:nz; %sequence in which items will be predicted
    ideal_ranks(ideal_I) = 1:nz; %sequence in which items was rated
    
    [~,oriOrder]       =  sort(ideal_ranks, 2, 'ascend');
    
    nominator           = userRating./log(ranks + 1);
    denominator         = userRating ./ log(ideal_ranks + 1);
    
    nominator           = nominator(oriOrder);
    denominator         = denominator(oriOrder);
    
    
    
    if k > nz
        % pad more 0
        nominator = padarray(nominator, [0, k - nz], 0, 'post');
        denominator = padarray(denominator, [0, k - nz], 0, 'post');
    elseif k < nz
        % truncate the tail
        nominator = nominator(1:k);
        denominator = denominator(1:k);
    end
    
    if size(find(cumsum(denominator)==0), 2) ~= 0
        tmp = zeros(1,k);
    else
        tmp = cumsum(nominator)./ cumsum(denominator);
        cnt = cnt + 1;
    end
    %tmp = full(tmp);
    %tmp = padarray(tmp, [0, length(k_vals) - size(tmp, 2)], 0, 'post');
	res = res + tmp;
    
end
res = res / cnt;
