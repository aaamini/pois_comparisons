
% load('amazon_data.mat')
% load('../data/PRO_breast_all_6months.mat')
load('../data/toxic_all_new.mat')
%%
load('../data/toxic_all_new.mat')
nSamples = 1000;
nCV = 3;
sigmaVec = 10.^(-2:0.2:2)';= 10.^(-2:0.2:2)';
nBasisPair = 2^6;% Number of basis for MMD approximation
evalFunc = @(X,Y) pair_complement_mmd(X, Y, sigmaVec, nBasisPair);
lamVec = 10.^linspace(-4,-1.3, 12);
% lamVec = 10.^linspace(-3,-1, 12);
testPect = 0.3;
nreps = 2;
% evalFunc = @(x,y) 0;
% [XtSampleTune, timing] = pois.tune_fit([0.001 0.01 0.1], Xt(:,1:5), nSamples, nCV, evalFunc);
% [fLambda, timing, meanEvalTune] = pois.tune_fit(lamVec, Xt(:,1:20), nSamples, testPect, nreps, evalFunc);
[fXtSample, fLambda, timing, meanEvalTune] = pois.tune_fit(lamVec, Xt, nSamples, testPect, nreps, evalFunc);

fLambda
timing.tune.total
%%
figure(1), semilogx(lamVec, meanEvalTune)

find(lamVec == fLambda)
meanEvalTune(lamVec == fLambda)

%%
figure(1), subplot(121), imagesc(fXtSample), colorbar, subplot(122), spy(fXtSample)
%%
[out.XtSampleS, trainTime, sampleTime] = pois.fit_and_sample(Xt(1:200,1:5), lamVec, 100);

%%
[model, trainTime] = IndNegBin.fit(Xt)
[Xs, sampleTime] = IndNegBin.sample(nSamples, model)

%%
[model, trainTime] = IndMult.fit(Xt);
[Xs, sampleTime] = IndMult.sample(nSamples, model);

%%
[model, trainTime] = CoupulaMult.fit(Xt);
[XtSample, sampleTime] = CoupulaMult.sample(nSamples, model);

%%
[model, trainTime] = CoupulaPoi.fit(Xt, 'Gaussian');
[XtSample, sampleTime] = CoupulaPoi.sample(nSamples, model);
