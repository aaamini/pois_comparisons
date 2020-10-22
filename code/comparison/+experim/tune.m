function [fModel, fXtSample, fParam, timing, meanPairTune] = ...
    tune(trainFunc, paramVec, sampleFunc, XtTrain, nSamples, nCV, evalFunc, tuneParamVecOverride)
% Tuning function for hyperparameters of model
    
    if(~isempty(tuneParamVecOverride))
        paramVec = tuneParamVecOverride;
    end
    %% Tune split
    timing = [];
    ts = tic;
    rndSeed = 1;
    [XtTrainTune, XtTestTune] = mrfs.utils.traintestsplit( XtTrain, 1/nCV, rndSeed );

    %% Loop through parameter values
    fprintf('  << Starting hyperparameter tuning >>\n');
    if(length(paramVec) == 1)
        fParam = paramVec;
        bestTuneModel = [];
        meanPairTune = NaN;
        timing.tune = [];
        fprintf('    Only one parameter so not tuning\n');
    else
        meanPairTune = NaN(size(paramVec));
        medianPairTune = NaN(size(paramVec));
        nSamplesTune = max(100, nSamples/10);
        model = [];
        timing.tune.model = zeros(size(paramVec));
        timing.tune.sample = zeros(size(paramVec));
        for pi = 1:length(paramVec)
            %% Train
            tsModel = tic;
            param = paramVec(pi);
            model = trainFunc(XtTrainTune, param, model);
            timing.tune.model(pi) = toc(tsModel);

            %% Sample
            tsSample = tic;
            XtSampleTune = sampleFunc( model, nSamplesTune );
            timing.tune.sample(pi) = toc(tsSample);

            %% Compute MMD
            %[~, mmdTune] = MMDFourierFeature(XtTestTune, XtSampleTune, sigmaVec, nBasis);
            % pairValues = evalFunc(XtTestTune, XtSampleTune, sigmaVec, nBasisPair);
            pairValues = evalFunc(XtTestTune, XtSampleTune);

            % upperTri = triu(true(size(pairValues)),0);
            % meanPairTune(pi) = mean(pairValues(upperTri));
            meanPairTune(pi) = mean(vertcat(pairValues{:})); % mean([pairValue{:}]);
            % medianPairTune(pi) = median(pairValues(upperTri));
            fprintf('    Finished tune paramIdx = %d, param = %g, meanPair = %g in %g s\n', ...
                pi, param, meanPairTune(pi), timing.tune.model(pi) + timing.tune.sample(pi));

            [~,minI] = min(meanPairTune);
            if(minI == pi)
                bestTuneModel = model;
            end

            if(meanPairTune(pi) >= 2*meanPairTune(1))
                fprintf('    Stopping tuning early since meanPairTune(cur) > 2*meanPairTune(1)\n\n');
                break;
            end
        end

        %% Display results
        fprintf('    %10s, %10s, %10s\n','paramVec', 'meanPair', 'medianPair');
        fprintf('    %10.5g, %10.5g, %10.5g\n', [paramVec;meanPairTune;medianPairTune]);
        [~,minI] = min(meanPairTune);
        fParam = paramVec(minI);
        timing.tune.total = toc(ts);
        fprintf('\n    Selected parameter = %d with mean pair value of = %g\n',fParam,meanPairTune(minI));
    end
    fprintf('  << Finished hyperparameter tuning >>\n\n');
    
    %% Use final param value to train 
    ts = tic;
    fModel = trainFunc( XtTrain, fParam, bestTuneModel );
    timing.train = toc(ts);
    
    %% Sample from final model
    ts = tic;
    fXtSample = sampleFunc( fModel, nSamples );
    timing.sample = toc(ts);
end
