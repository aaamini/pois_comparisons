function [biased, unbiased] = fast_mmd(X, Y, sigvec, nBasis)
% Copyright 2020 Arash A. Amini
% MIT License 

% The function use the Random Kitchen Sink idea (i.e., random Fourir
% features) to compute the MMD based on a Gaussian kernel.
% nBasis: Number of Fourier basis vectors

% rng('default');

k0 = 1; % k0 = K(0,0)
d = size(X,2);
if size(Y,2) ~= d
    error('# of columns of X and Y should match.')
end
n = size(X,1);
m = size(Y,1);
N = nBasis;

i = sqrt(-1);
Z = randn(N,d);
ZXt = Z*(X.'); % N x n
ZYt = Z*(Y.'); % N x m
phi = @(sigma) mean( exp(i*ZXt/sigma) / sqrt(N), 2) ;
psi = @(sigma) mean( exp(i*ZYt/sigma) / sqrt(N), 2) ;


[biased, unbiased] = deal(zeros(size(sigvec)));
for k = 1:numel(sigvec)
    sigma = sigvec(k);
    u = phi(sigma); % N x 1
    v = psi(sigma); % N x 1
    temp = l2norm_squared(u - v) ;
    biased(k) = sqrt(temp); % norm(phi(sigma) - psi(sigma));
    unbiased(k) = temp + l2norm_squared(u)/(n-1) + l2norm_squared(v)/(m-1) - k0*(n+m-2)/(n-1)/(m-1);
    unbiased(k) = sqrt(max(unbiased(k),0));
end

end


function out = l2norm_squared(x) 
    % Assumes x is a column vector with complex entries and computes its
    % l_2 norm squared.
    out = sum(x'*x);
end