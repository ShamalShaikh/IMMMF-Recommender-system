function count_rat(add,l)
    for i = 1:l
        A = full(add==i);
        fprintf('Count of %d : %d\n', i,sum(sum(A)));
    end
end