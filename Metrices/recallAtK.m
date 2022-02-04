function recall = recallAtK(Ypre,Yt, k, cutoff)

%Yt  : Test Set of size n *m
%Ypre: Prediction set of size n *m
%cutoff : threshold for relevant item
%k: recall@k


userWithRating = sum(Yt~=0,2)>0;
Yt             = Yt(userWithRating,:);
Ypre           = Ypre(userWithRating,:);
[N,~]          = size(Yt);

recall         = zeros(1, k);
for user = 1:N
    userRating     = Yt(user,:); 
    userPrediction = Ypre(user,:);
   
    ratedRelevantIdx = find(userRating >= cutoff);    
    
    ratedIdx         = find(userRating~=0);
    prediction       = userPrediction(ratedIdx);        
    [~,sortesPIdxTmp]= sort(prediction,'descend');  
    sortesPIdx       = ratedIdx(sortesPIdxTmp);
    
%     tempk            = k;    
%     if length(sortesPIdx)< k
%         tempk      = length(sortesPIdx);
%     end    
%     recall = recall + length(intersect(sortesPIdx(1:tempk),ratedRelevantIdx))./max(length(ratedRelevantIdx),eps);
    
    nz = length(sortesPIdx);
    for kNo = 1:k    
    recall(kNo) = recall(kNo) + length(intersect(sortesPIdx(1:min(kNo,nz)),ratedRelevantIdx))./max(length(ratedRelevantIdx),eps);
    end   
end
recall = recall./N;
end