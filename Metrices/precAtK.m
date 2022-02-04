function prec = precAtK(Ypre,Yt, k, cutoff)
%Yt  : Test Set of size n *m
%Ypre: Prediction set of size n *m
%cutoff : threshold for relevant item
%k: precision@k

userWithRating = sum(Yt~=0,2)>0;
Yt             = Yt(userWithRating,:);
Ypre           = Ypre(userWithRating,:);
[N,~]          = size(Yt);

%prec           = 0;
prec           = zeros(1, k);


for user = 1:N
    userRating       = Yt(user,:);
    ratedRelevantIdx = find(userRating >= cutoff);  
    
    ratedIdx         = find(userRating~=0);
    
    userPrediction   = Ypre(user,:);   
    prediction       = userPrediction(ratedIdx);    
    
    [~,sortesPIdxTmp]= sort(prediction,'descend');  
    sortesPIdx       = ratedIdx(sortesPIdxTmp);
    
%     tempk            = k;     
%     if length(sortesPIdx)< k
%         tempk        = length(sortesPIdx);
%     end   
%     prec = prec + length(intersect(sortesPIdx(1:tempk),ratedRelevantIdx))./max(tempk,eps);


   nz = length(sortesPIdx);
   for kNo = 1:k    
    prec(kNo) = prec(kNo) + length(intersect(sortesPIdx(1:min(kNo,nz)),ratedRelevantIdx))./max(min(kNo,nz),eps);
   end

end
prec = prec./N;
end