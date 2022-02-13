function ratings = noisy(Yiter,Ytrn,L)
    YtstPred = Y .* (Ytst ~= 0);
    
    fprintf(f, "\n %s \n", str);

    fprintf(f, "Confusion Matrix: %d\n", length(find(sparse(abs(YtstPred-Ytst) > 0))));
    
    fprintf(f, "\t 1 \t    2 \t    3 \t    4 \t    5 \n");

    for r = 1:L
        count = full((YtstPred == r) .* Ytst);
        fprintf(f, "%d \t%d \t%d \t%d \t%d \t%d\n", r, histc(count(:), [1 2 3 4 5]) );
    end
end