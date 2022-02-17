function confusion_mat(Y, Ytst, L, f, str)
        
    YtstPred = Y .* (Ytst ~= 0);
    
    fprintf(f, "\n %s \n", str);

    fprintf(f, "Confusion Matrix: %d\n", length(find(sparse(abs(YtstPred-Ytst) > 0))));
    
    fprintf(f, "\t 1 \t    2 \t    3 \t    4 \t    5 \n");
    
    for r = 1:L
        count = full((YtstPred == r) .* Ytst);
        c = histc(count(:), [1 2 3 4 5]);
        fprintf(f, "%d \t%d \t%d \t%d \t%d \t%d \t\t %d\n", r, c(1:5), c(r)/sum(c) );
    end

end