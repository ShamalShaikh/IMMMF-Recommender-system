% Zero-one error
function [ret] = zoe(x,y)
  %c = full(sum(sum((a~=b).*(b>0)))./sum(sum(b>0)));
  ret = full(sum(sum((x.*(y~=0))~=y))./nnz(y));
