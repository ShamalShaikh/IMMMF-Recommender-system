function [ err ] = rre( UVT, Y )

UVT = round( UVT .* (Y>0));
Ynorm = sum(sum(Y .* Y));
YUVTnorm = sum(sum( (Y- UVT) .* (Y- UVT) ));
err = YUVTnorm ./ Ynorm;
end