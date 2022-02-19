function [Ynew] = removeDistNoise(X, Ytrn, theta, rank)
        
    X = full(X);
    [n,m] = size(X);
    L = size(theta,2) + 1;
    Ynew = zeros(n,m);
    
    averageThetaD = zeros(n,1);
    for l=1:L-2
         averageThetaD = averageThetaD + abs(theta(:,l+1)-theta(:,l));
    end
    averageThetaD = averageThetaD./(L-2);

%     A = Ytrn .* (Ytrn==1);
    B = X - (theta(:,1)*ones(1,m) .* (Ytrn==1));
    dist = B .* (B~=X);

    Nmax = rank; % get Nmax biggest entries
    [Av, Ind] = sort(dist(:),1,'descend');
    max_values = Av(1:Nmax);
    fprintf(1,"\n");fprintf(1, "%g ", max_values);fprintf(1,"\n");
    [ind_row, ind_col] = ind2sub(size(dist),Ind(1:Nmax)); % fetch indices
    dist(ind_row,ind_col) = 0;
%     dist(dist > 2*averageThetaD) = 0;
    Ynew(dist~=0) = 1;

    for r = 1:L-2
        B = X - (theta(:,r) * ones(1,m) .* (Ytrn==r+1));
        A = X - (theta(:,r+1) * ones(1,m) .* (Ytrn==r+1));
        dist = B .* (B~=X);
        Nmin = rank; % get Nmin small entries
        [Av, Ind] = sort(dist(:),1,'ascend');
        min_values = Av(1:Nmin);
        fprintf(1,"\n");fprintf(1, "%g ", min_values);fprintf(1,"\n");
        [ind_row, ind_col] = ind2sub(size(dist),Ind(1:Nmin)); % fetch indices
        dist(ind_row,ind_col) = 0;
%         dist(dist < -1*averageThetaD) = 0;
        
        dist1 = A .* (A~=X);
        Nmax = 100; % get Nmax biggest entries
        [Av, Ind] = sort(dist1(:),1,'descend');
        max_values = Av(1:Nmax);
        fprintf(1,"\n");fprintf(1, "%g ", max_values);fprintf(1,"\n");
        [ind_row, ind_col] = ind2sub(size(dist1),Ind(1:Nmax)); % fetch indices
        dist1(ind_row,ind_col) = 0;
%         dist1(dist1 > 2*averageThetaD) = 0;
        Ynew(dist~=0 & dist1~=0) = r+1;
    end 
    B = X - (theta(:,4) * ones(1,m) .* (Ytrn==5));
    dist = B .* (B~=X);
    Nmin = 10; % get Nmax biggest entries
    [Av, Ind] = sort(dist(:),1,'ascend');
    min_values = Av(1:Nmin);
    fprintf(1,"\n");fprintf(1, "%g ", min_values);fprintf(1,"\n");
    [ind_row, ind_col] = ind2sub(size(dist),Ind(1:Nmin)); % fetch indices
    dist(ind_row,ind_col) = 0;
%     dist(dist < -2*averageThetaD) = 0;
    Ynew(dist~=0) = L;
    
end