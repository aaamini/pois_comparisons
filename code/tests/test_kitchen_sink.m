% load('test_data.mat')
% [d1, d2] = MMDFourierFeature(Xt, XtSample, 10.^(-2:0.2:2), 2^6)
% figure(1), hist(d2)

%%
sigma = 2;

N = 2^13;
d = 5;
n = 300;
m = 400;
x = randn(n,d);
y = randn(m,d);

%%
% i = sqrt(-1);
% Z = randn(N,d);
% phi = @(X) exp(i*Z*(X.')/sigma) / sqrt(N) ;
% 
% %%
% Kapp = real(phi(x)'*phi(y));
% 
% K = zeros(n,m);
% for p = 1:n
%     for q = 1:m
%      K(p,q) = exp(-norm(x(p,:)-y(q,:))^2/(2*sigma^2));
%     end
% end
% Kapp
% K
% norm(K - Kapp)/norm(K)
% abs(sum(Kapp(:))-sum(K(:)))/abs(sum(K(:)))

%%
[d1, d2] = MMDFourierFeature(x, y, sigma, 2^13);
[e1, e2] = fast_mmd(x, y, sigma, 2^13);

[d1,d2]
[e1,e2]

%%
load('test_data.mat')
nBasis = 2^10;
sigvec = [1 2];
tic, [d1, d2] = MMDFourierFeature(Xt, XtSample, sigvec, nBasis); toc
tic, [e1, e2] = fast_mmd(Xt, XtSample, sigvec, nBasis); toc

[d1,d2]
[e1(:), e2(:)]
%%
mmd =  pair_complement_mmd(Xt, XtSample, sigma, 2^8);
figure(1), hist(mmd)

%%
load('amazon_data.mat')
% full(mean(Xt,1))
% X = Xt(1:100, 1:5);
% Y = Xt(500:600, 1:5);
[d1, d2] = MMDFourierFeature(X, Y, 2, 64);
[e1, e2] = fast_mmd(X, Y, 2, 64);

[d1,d2, e1(:), e2(:)]

%%
datasetLabel = 'amazon';
% datasetLabel = 'PROall';
% datasetLabel = 'toxic_all_new';
% nDim = [];
nDim = 50;
[Xt, labels, nDim] = experim.load_data(datasetLabel, nDim);

sigmaVec = 10.^(-2:0.2:2)';
nBasisPair = 2^6;% Number of basis for MMD approximation
evalFunc = @(X,Y) pair_complement_mmd(X, Y, sigmaVec, nBasisPair);
max_nPairs = 200; % limit the max number of pairs for a "rough" estimate
evalFuncRough = @(X,Y) pair_complement_mmd(X, Y, sigmaVec, nBasisPair, max_nPairs); % rough version, used for tuning
lamVec = 10.^linspace(-4,-1.3, 12);
% lamVec = 10.^linspace(-3,-1, 12);
testPect = 0.3;
nreps = 2;
nSamples = 1000;
[XtSample, fLambda, timing, meanEvalTune]  = ... 
        pois.tune_fit(lamVec, Xt, nSamples, testPect, nreps, evalFuncRough);

    %%
sig = 2;
[~, d2] = MMDFourierFeature(Xt, XtSample, sig, 2^9);
[~, e2] = fast_mmd(Xt, XtSample, sig, 2^9);

[d2, e2]
%%
X = Xt(1:100, 1:50);
Y = Xt(101:300, 1:50);
tic, out = pair_complement_mmd(X, Y, 10.^(-2:0.2:2)', 64, 500); toc
figure(1), hist(out)

