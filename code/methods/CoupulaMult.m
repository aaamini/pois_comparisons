classdef CoupulaMult
% Implements Copula Multinomial Model.

    methods(Static)
        function vals = cdf(x,p)
            % p should be a probability vector
            % evaluates the CDF of a catergorical(p) random variable at
            % points x
            nlevels = length(p);
            cump = [0 cumsum(p(:)')];
            vals = cump( 1+min(max(0,floor(x)),nlevels) );
        end
        function x = invcdf(y,p) 
            % p should be a probability vector
            % evaluates the inverse CDF (quantile function) of a catergorical(p) random variable at
            % points y \in [0,1]
            cump = [0 cumsum(p(:)')];
            [pknots,YY] = meshgrid(cump,y);
            x = sum(YY >= pknots,2)-1;
        end
        function [model, trainTime] = fit(Xt)
            ts = tic;
            U = NaN(size(Xt));
            % mnP = IndMultModel.estimate(Xt);  
            p = size(Xt,2);
            [mnP, Levels] = deal(cell(p,1));
            for s = 1:p
                xs = full(Xt(:,s));
                [mnP{s}, Levels{s}, mappedx] = IndMult.estimate1d(xs);
                %DT Transform
                U(:,s) = ( CoupulaMult.cdf(mappedx, mnP{s}) + CoupulaMult.cdf(mappedx-1, mnP{s}) )/2;
            end
            U(U==1) = 1-eps; % Need to make strictly < 1 because of rounding error
            % Fit copula
            model.rhohat = copulafit('Gaussian', U);
            model.mnP = mnP;
            model.Levels = Levels;
            model.Levels = Levels;
            trainTime = toc(ts);
        end
        function [XSample, sampleTime] = sample(nSamples, model)
            ts = tic;
            USample = copularnd('Gaussian', model.rhohat, nSamples);
            XSample = zeros(size(USample));
            p = size(model.rhohat,2);
            for s = 1:p
                temp = CoupulaMult.invcdf(USample(:,s), model.mnP{s});
                XSample(:,s) = model.Levels{s}(temp + 1);  % this could be wrong if there is no zero level (?)
            end          
            sampleTime = toc(ts);
        end
    end
end


% %% Tests
% [p, levels] = IndMultModel.estimate1d(Xt(:,10))
% tabulate(IndMultModel.sample1d(500, p, levels))
% 
% %%
% [P, Levels] = IndMultModel.estimate(Xt)
% IndMultModel.sample(50, P, Levels)



