function [X,Variance] = sample(n, p, R, alpha, Theta, maxit)
% Trancated PGM Gibbs Sampler (formerly "GibbsTPGM")
% 
% function to simulate from the WLLGM model
% alpha a px1 vector
% Theta a pxp symmetric matrix (only off diags matter)
% n = sample size
% p = variable size
% R = truncating number
% maxit = iterations for Gibbs sampler
temp_t = tic;
fprintf('  Sampleing T-PGM ... ')
X = poissrnd(1,n,p);
% csvwrite('x.csv',X);
iter = 1; setp = 1:p;
Variance=zeros(n,p,maxit);
logFactCache = log(factorial((0:R)));
while iter<maxit
    for s=1:p
        %num2 = exp( repmat((alpha(s)*(0:R) - log(factorial((0:R)))),n,1)...
        %    + kron((0:R),X(:,setp~=s)*Theta(setp~=s,s)) );
        num = exp(bsxfun(@plus, alpha(s)*(0:R) - logFactCache, (X(:,setp~=s)*Theta(setp~=s,s))*(0:R) ));
        %assert(all(num(:)==num2(:)),'');
        Pmat = num./(repmat(sum(num,2),1,R+1));
        
        %simple = mnrnd(1,Pmat,n);
        cumPmat = cumsum(Pmat,2);
        randVec = rand(size(Pmat,1),1); 
        simple = bsxfun(@lt, randVec, cumPmat) & bsxfun(@gt, randVec, [zeros(size(randVec)), cumPmat(:,1:(end-1))]);
        [ii,~] = ind2sub([R+1 n],find(simple'));
        X(:,s) = ii-1;
    end
    Variance(:,:,iter)=X;
    iter = iter + 1;
end
fprintf(' %2.1f (s).\n', toc(temp_t))
