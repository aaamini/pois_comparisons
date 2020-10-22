classdef CoupulaPoi
% Implements Copula Poisson Model.

    methods(Static)       
        function [model, trainTime] = fit(Xt, copulaType)
            ts = tic;
            poissMean = full(mean(Xt)); % MLE estimate
            % DT transform (take mean of DT)
            U = (poisscdf(Xt, repmat(poissMean, size(Xt,1),1)) + ...
                poisscdf(Xt - 1, repmat(poissMean, size(Xt,1),1)))/2;
            U(U==1) = 1-eps; % Need to make strictly < 1 because of rounding error

            % Fit copula
            if(strcmp(copulaType,'t'))
                [rhohat, nuhat] = copulafit(copulaType, U);
            else
                rhohat = copulafit(copulaType, U);
                nuhat = NaN;
            end
            trainTime = toc(ts);
            model.poissMean = poissMean;
            model.rhohat = rhohat;
            model.nuhat = nuhat;
            model.copulaType = copulaType;
        end
        function [XSample, sampleTime] = sample(nSamples, model)
            % Get samples by getting uniform samples and then using inverse of kernel density estimate
            ts = tic;
            if(strcmp(model.copulaType, 't'))
                USample = copularnd(model.copulaType, model.rhohat, model.nuhat, nSamples);
            else
                USample = copularnd(model.copulaType, model.rhohat, nSamples);
            end
            XSample = poissinv(USample, repmat(model.poissMean, nSamples, 1));
            sampleTime = toc(ts);
        end
    end
end





