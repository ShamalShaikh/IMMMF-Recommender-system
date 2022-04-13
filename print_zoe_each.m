function print_zoe_each(Ypred,Ytrn, L, f2, error, i,l)
    rat_zoe = zeros(1,L);
    for r = 1:L
        Yr = Ypred .* (Ypred==r);
        Ytr = Ytrn .* (Ytrn==r);
        rat_zoe(1,r) = full(sum(sum((Yr.*(Ytr~=0)~=Ytr)))./nnz(Ytr));
    end 
    fprintf(f2,'\n %3d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f', i, rat_zoe, error(:));
end