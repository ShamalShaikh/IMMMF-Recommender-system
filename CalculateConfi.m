function ratios = CalculateConfi(Y, Ytst, L)
    ratios = zeros(1,L);
    YtstPred = Y .* (Ytst ~= 0);
    
    for r = 1:L
        count = full((YtstPred == r) .* Ytst);
        c = histc(count(:), [1 2 3 4 5]);
        ratios(1,r) = c(r)/sum(c);
    end
end