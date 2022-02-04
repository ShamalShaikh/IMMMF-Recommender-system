% Mean absolute error
function [c] = mae(a,b)
  %c = full(sum(sum(abs(a-b).*(b>0)))./sum(sum(b>0)));
  %c = full(sum(sum(abs(a.*(b>0)-b)))./sum(sum(b>0))); %original
  c = full(sum(sum(abs(a.*(b~=0)-b)))./sum(sum(b~=0)));
  