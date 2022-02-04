function auc = AUC(Ypre,Yt, cutoff)
%Yt  : Test Set of size n * m
%Ypre: Prediction set of size n *m
%cutoff : threshold for relevant item

userWithRating = sum(Yt~=0,2)>0;
Yt             = Yt(userWithRating,:);
Ypre           = Ypre(userWithRating,:);
[N,~]          = size(Yt);

auc           = 0;

uWithPair     = 0;
for user = 1:N
    
    
    userRating     = Yt(user,:); 
    userPrediction = Ypre(user,:);
    
    ratedIdx       = userRating>0;
    binUserRating  = 2*(userRating(ratedIdx) >= cutoff)-1;    
    predForKnown   = userPrediction(ratedIdx);
    
    posIdx         = find(binUserRating ==1)';
    negIdx         = find(binUserRating ==-1);
    
    if ~isempty(posIdx) &  ~isempty(negIdx) 
    negIdxGrid     = repmat(negIdx,length(posIdx),1);
    ttlNeg         = numel(negIdxGrid);    
    negIdxGrid     = reshape(negIdxGrid,ttlNeg,1);    
    posIdxGrid     = repmat(posIdx,length(negIdx),1);    
    pairs          = [negIdxGrid,posIdxGrid];    %neg , pos
    pairsPred      = [pairs,predForKnown(negIdxGrid)',predForKnown(posIdxGrid)']; %negIdx , posIdx, predictionNeg, predictionPos
    
    pairsPredWithScore  =  [pairsPred,0.5.*(pairsPred(:,3)== pairsPred(:,4)), 1.*(pairsPred(:,3)<pairsPred(:,4))];
    
    auc = auc + (sum(sum(pairsPredWithScore(:,5:6))))./size(pairsPredWithScore,1);
    uWithPair = uWithPair +1;
    end
end
%auc = auc./N;
auc = auc./uWithPair;
end