function [err] = wrong(Y, Ytst, YPred, l, f1)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
Yt = YPred .* (Ytst~=0);
fprintf(f1, "Wrong pred in testing set i.e with abs>0 : %d\n", length(find(sparse(abs(Yt-Ytst) > 0))));
for rat = 1:l
    val = abs(Yt-Ytst);
    fprintf(f1, 'Rating %d wrong predictions : %d\n', rat, length(find(sparse(val.*(Ytst==rat) > 0))));
    fprintf(f1, "1 : %d\t\t", length(find(sparse((val.*(Ytst==rat)) == 1))));
    fprintf(f1, "2 : %d\t\t", length(find(sparse((val.*(Ytst==rat)) == 2))));
    fprintf(f1, "3 : %d\t\t", length(find(sparse((val.*(Ytst==rat)) == 3))));
    fprintf(f1, "4 : %d\n", length(find(sparse((val.*(Ytst==rat)) == 4))));
end
end

