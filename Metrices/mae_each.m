% Mean absolute error
function [c] = mae_each(a,b,r)
  %c = full(sum(sum(abs(a-b).*(b>0)))./sum(sum(b>0)));
  %c = full(sum(sum(abs(a.*(b>0)-b)))./sum(sum(b>0))); %original
  BR = (b==r);
  c = full(sum(sum(abs(a.*BR- b.*BR))))./sum(sum(BR)); % why > 1?
  