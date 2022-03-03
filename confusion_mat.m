function confusion_mat(Y, Ytst, L, f, str)
        
    YtstPred = Y .* (Ytst ~= 0);
    
    fprintf(f, "\n %s \n", str);

    fprintf(f, "Confusion Matrix: %d\n", length(find(sparse(abs(YtstPred-Ytst) > 0))));
    
    fprintf(f, "  \t  \t 1\t \t \t 2 \t \t \t 3 \t \t \t 4 \t \t \t 5 \t \t \t ratio\n");
    
    for r = 1:L
        count = full((YtstPred == r) .* Ytst);
        c = histc(count(:), [1 2 3 4 5]);
        fprintf(f, "%d \t %6d \t %6d \t %6d \t %6d \t %6d \t \t %1.4f\n", r, c(1:5), c(r)/sum(c) );
    end

end