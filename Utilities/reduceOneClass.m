function [Rtrain]=reduceOneClass(Rtrain,i)
ind3=find(Rtrain==i);
reduce =70;
no3=size(ind3,1);
ind1_perm= randperm(no3);
Rtrain(ind3(ind1_perm(1:ceil(no3*reduce/100))))=0;
end