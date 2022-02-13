function Ytrn = removeNoise(Yiter, Ytrn, L)

    Yn = Yiter .* (Ytrn ~= 0);
    conf = zeros(1,L);
    for r = 1:L
        count = full((Yn == r) .* Ytrn);
        c = histc(count(:), [1 2 3 4 5]);
        conf(1,r) = c(r)/sum(c);
    end
    confRat = (conf>=0.9);
    [n,m] = size(Yiter);
    Yt = 
    for r = 1:L
        if confRat(1,r)==1
            Yt(n,m,r) = Yiter .* (Yiter == r);
            
        end
    end
end