classdef IndMult
    methods(Static)
        function xs = sample1d(nSamples, p, levels)
            % p should be a row probability vector
            xs = mnrnd(1,repmat(p,nSamples,1)) * levels(:);
        end
        function [mappedx, levels] = transform_to_cts_range(x)
            levels = sort(unique(x));
            nlevels = length(levels);
            dict = zeros(1,max(levels)+ 1); % assuming minimum level is >=0
            dict(levels+1) = 1:nlevels; % dictionary for mapping values to 1:nlevels
            mappedx = dict(x + 1);    
        end
        function [p, levels, mappedx] = estimate1d(x)
%             levels = sort(unique(x));
%             nlevels = length(levels);
%             dict = zeros(1,max(levels)+ 1); % assuming minimum level is >=0
%             dict(levels+1) = 1:nlevels; % dictionary for mapping values to 1:nlevels
% 
%             mappedx = dict(x + 1);
            [mappedx, levels] = IndMult.transform_to_cts_range(x);
            nlevels = length(levels);
            IdMat = eye(nlevels);
            p =  mean(IdMat(mappedx,:));            
        end
        function [Xs, sampleTime] = sample(nSamples, model)
            ts = tic;
            d = size(model.P,1);
            Xs = zeros(nSamples, d);
            for j = 1:d 
                Xs(:,j) = IndMult.sample1d(nSamples, model.P{j}, model.Levels{j});
            end
            sampleTime = toc(ts);
        end
        function [model, trainTime] = fit(X)
            ts = tic;
            d = size(X,2);
            Levels = cell(d,1);
            P = cell(d,1);
            for j = 1:d
                [P{j}, Levels{j}] = IndMult.estimate1d(X(:,j));
            end
            model.Levels = Levels;
            model.P = P;
            trainTime = toc(ts);
        end
       
    end
end


% %% Tests
% [p, levels] = IndMult.estimate1d(Xt(:,10))
% tabulate(IndMult.sample1d(500, p, levels))
% 
% %%
% [P, Levels] = IndMult.estimate(Xt)
% IndMult.sample(50, P, Levels)



