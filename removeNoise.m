function Ytrn = removeNoise(Yiter, Ytrn, L)

    Yn = Yiter .* (Ytrn ~= 0);
    for r = 1:L
        count = full((Yn == r) .* Ytrn);
        c = histc(count(:), [1 2 3 4 5]);
        conf = c(r)/sum(c);
        [s,r1] = sort(c,1,"descend");

%         & Ytrn~=r1(2)
        if conf >= 0.9
                Ytrn(Ytrn~=0 & Ytrn~=r & Yiter==r  ) = r;
        end
    end
%     confRat = (conf>=0.9);
%     Ytrn = zeros(size(Ytrn));
%     for r = 1:L
%         if confRat(1,r)==1
% 
% % Ytrn = Ytrn + Yiter .* (Yiter == r);
%         end
%     end
end