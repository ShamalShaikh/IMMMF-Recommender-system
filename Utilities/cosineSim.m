function [A] = cosineSim(Y)

[N,~] = size(Y);
nominator = Y*Y';

Ysq = Y.^2;
denominator = zeros(N,N);

for i=1:N
    for j=i:N
        common = Y(i,:)~=0 & Y(j,:)~=0;
        squareSum1 = sqrt(sum(Ysq(i,common)));
        squareSum2 = sqrt(sum(Ysq(j,common)));
        denominator(i,j) = squareSum1*squareSum2;
        denominator(j,i) = denominator(i,j);        
    end
end
denominator(denominator==0) = eps;
A = nominator ./ denominator;
if isnan(A)
    A = 0;
end

end