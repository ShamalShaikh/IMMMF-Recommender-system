% Rescale columns of U and V so that the i'th column of U has the same L2
% length as the i'th column of V.
%
% function [newU,newV] = normCols(U,V)
% U - matrix [m,p]
% V - matrix [n,p]

function [U,V] = normCols(U,V)
  [m,p] = size(U);
  [n,p1] = size(V);
  if p~=p1
    error('Dimensions of U and V don''t match');
  end
  Ucl = sqrt(sum(U.^2,1));
  Vcl = sqrt(sum(V.^2,1));
  Uprod = sqrt(Vcl./Ucl);
  Vprod = sqrt(Ucl./Vcl);
  U = U.*repmat(Uprod,m,1);
  V = V.*repmat(Vprod,n,1);
