% Rounds number to a given number of significant digits.
% 
% [y] = pround(x,d)
% x - input number
% d - number of digits
% y - rounded number
% 
% Written by Jason Rennie, December 2003
% Last modified: Tue Jul 25 01:01:30 2006

function [y] = pround(x,d)
  if nargin ~= 2
    error('Wrong number of arguments (%d)',nargin)
  end
  d = round(d);
  if d < 1
    error('Number of digits must be integer d=%.4e\n',d);
  end
  if x==0 | isnan(x) | isinf(x)
    y = x;
  else
    p = floor(log10(abs(x)))+1;
    factor = 10.^(d-p);
    y = round(x.*factor)./factor;
  end
