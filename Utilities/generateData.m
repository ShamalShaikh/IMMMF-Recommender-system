function [Y1] = generateData(row,column,non0Per,rating_per)
%row = 100;
%disp('row');
%disp(row);

%column = 100;
max_index = row*column;
non_zero = ceil((max_index*non0Per)/100);
Y=zeros(row,column);

%
temp=randi([1 5],row,column);
idx=randperm(max_index);
idx=idx(1:non_zero);
Y(idx)=temp(idx);

%}
%{
%rating_per = [1 1 1 1 0];
%rating_per = [1 2 5 6 4];
%rating_per = [7 5 2 2 3];
%rating_per = [375 1138 7869 20560 20875];
%rating_per = [6	10	26	35	23];


total = sum(rating_per(:));
ratio = ceil(non_zero / total);

for i=1:5
    idx_zero = find( Y == 0 );
    rand_idx = randperm(size(idx_zero,1));
    Y(idx_zero(rand_idx(1:rating_per(i)*ratio)))= i;
    clear rand_idx idx_zero
end
%}
n = row;
m = column;
d = min(m,n)-1;
V = rand(m, d);
%U = Y * inv(V*V') * V;
U = Y * pinv(V');
Y1 = round(U*V');
Y1 = Y1 .* (Y1>0);
Y1(Y1>5) = 5;
Y0 = Y;
i = 1;
U1 = U;
V1 = V;
I = Y > 0;
Y1 = Y1 .* I;

while(sum(sum(Y0-Y1)) ~= 0)
%while(sum(sum(I.* abs(Y-Y1))) ~= 0)
    Y0 = Y1;
    U0 = U1;
    V0 = V1;
    if(mod(i,2) == 0)
        %U1 = U0;
        V1 = (pinv(U1) * Y0)';
        %V1 = ((U1'* (inv(U1*U1'))) * Y0)';
    else
        %V1 = V0;
        U1 = Y0 * pinv(V');
    end
    Y1 = round(U1 * V1');
    Y1 = Y1 .* (Y1>0);
    Y1(Y1>5) = 5;
    i = i+1;    
    Y1 = Y1 .* I;
    %sum(sum(abs(Y0 - Y1)))
end
%i, Y1, Y
%Y
%Y1 = Y1.* I;
end

