function idx = spectralClu(A,K)
% Ng, A., Jordan, M., and Weiss, Y. (2002). On spectral clustering: analysis and an algorithm. In T. Dietterich,
% S. Becker, and Z. Ghahramani (Eds.), Advances in Neural Information Processing Systems 14 
% (pp. 849 – 856). MIT Press.

% A: Affinity Matrix, Higher value -> more similar
% K: Number of CLuster

n = size(A,1);
D = sum(A,2); D(D==0) = eps;
D = spdiags(1./sqrt(D),0,n,n);
L = D * A * D;
[V,evals] = evecs(L,K);
for i=1:size(V,1) 
        V(i,:)=V(i,:)/(norm(V(i,:))+eps);  
end
idx = kmeans(V,K,'MaxIter',100,'OnlinePhase','off');
end