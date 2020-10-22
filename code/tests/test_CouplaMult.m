%%
[~,XX] = max(mnrnd(1,[0.4 0.2 0.2 0.1 0.1],50), [], 2);
% XX = mnrnd(1,[0.4 0.2 0.2 0.1 0.1],50) * (1:5)';
tabulate(XX)

%%
XX = Xt(:,4)
% [p, levels] = IndMultModel.estimate1d(Xt(:,1))
[p, levels] = IndMultModel.estimate1d(XX)
x = linspace(-0.1,6,100);
% nlevels = length(p);
% cump = [0 cumsum(p(:)')];

figure(1), clf
% plot(x, cump( 1+min(max(0,floor(x)),nlevels) ))
plot(x, CoupulaMult.cdf(x,p))

[XX CoupulaMult.invcdf(CoupulaMult.cdf(XX,p),p)]

%%
y = linspace(0,1,100)
%[pknots,YY] = meshgrid(cump,y);
% [y(:) sum(YY >= pknots,2)-1]
[y(:) CoupulaMult.invcdf(y,p)]
figure(2), plot(y, CoupulaMult.invcdf(y,p))
%%
figure(2), clf, hist(CoupulaMult.cdf(XX,IndMultModel.estimate1d(XX)))

%%

% n = 1000;
% XX = zeros(n,3);
% for s = 1:3
%     [~,XX(:,s)] = max(mnrnd(1,[0.4 0.2 0.2 0.1 0.1],n), [], 2);
% end
XX = Xt;

[rhohat, mnP, Levels, trainTime] = CoupulaMult.estimateCopula(XX);
nSamples = 1000;
[XSample, sampleTime] = CoupulaMult.sample(rhohat, mnP, Levels, nSamples)

idx = 3
tabulate(XSample(:,idx))
tabulate(XX(:,idx))
%%
% lam = 4.5;
% XX = poissrnd(lam,500,1);
% figure(1), clf, plot(x, poisscdf(x,lam))
% figure(2), clf, hist((poisscdf(XX,lam) + poisscdf(XX-1,lam))/2), hold on
% figure(3), clf, hist(poisscdf(XX,lam))
