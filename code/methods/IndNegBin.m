classdef IndNegBin
 %% Impelements independent negative binomial model

    methods(Static)
        function [Xs, sampleTime] = sample(nSamples, model)
            ts = tic;
            p = length(model.dispersion);
            Xs = NaN(nSamples,p);
            for s = 1:p
                if (model.dispersion(s) > 1)
                    Xs(:,s) = nbinrnd(model.negbinR(s), model.negbinP(s), nSamples, 1);
                else
                    Xs(:,s) = poissrnd(model.poissMean(s), nSamples,1);
                end
            end
            sampleTime = toc(ts);
        end
               
        function [model, trainTime] = fit(X)
            ts = tic;
            p = size(X, 2);
            model.dispersion = NaN(p,1);
            model.poissMean = NaN(p,1);
            model.negbinR = NaN(p,1);
            model.negbinP = NaN(p,1);
            for s = 1:p
                xs = full(X(:,s));
                model.dispersion(s) = var(xs)/mean(xs);
                if (model.dispersion(s) > 1)
                    params = nbinfit(xs);
                    model.negbinR(s) = params(1);
                    model.negbinP(s) = params(2);
                else
                    model.poissMean(s) = mean(xs);
                end
            end
            trainTime = toc(ts);  
        end
    end
end




