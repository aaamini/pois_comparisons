function [fXtSample, fLambda, timing, meanEvalTune] = ...
    tune_fit(lamVec, XtTrain, nSamples, testPerc, nreps, evalFunc)
% wrapper for R code fitting the POIS model with lambda-tuning

timing = [];
ts = tic;
rndSeed = 1;

%% Loop through parameter values
fprintf('  << Starting hyperparameter tuning >>\n');
if(length(lamVec) == 1)
    fLambda = lamVec;
    meanEvalTune = NaN;
    timing.tune = [];
    fprintf('    Only one parameter so not tuning\n');
else
    % [XtTrainTune, XtTestTune] = mrfs.utils.traintestsplit( XtTrain, 1/nCV, rndSeed );
    meanEvalTune = zeros(size(lamVec));
    nSamplesTune = max(100, nSamples/10);
    for r = 1:nreps
        fprintf('<--- Starting rep %d/%d --->\n', r, nreps);
        [XtTrainTune, XtTestTune] = ... 
            mrfs.utils.traintestsplit( XtTrain, testPerc, rndSeed );
    
        [XtSampleTune, timing.tune.model, timing.tune.sample] = ... 
            pois.fit_and_sample( XtTrainTune, lamVec, nSamplesTune );
        % save(input_fpath, 'XtTrainTune','lamVec','nSamplesTune')
        % if ~system(sprintf('cd %s && Rscript tune_fit.R', curr_path))
            % fprintf('   R script ran successfully.\n')
        % end
        % S = load(output_fpath);
        % XtSampleTune = S.XtSampleTune;
        % timing.tune.model = S.tsModel;
        % timing.tune.sample  = S.tsSample; 

        % delete(input_fpath)
        % delete(output_fpath)

        for li = 1:length(lamVec)
            pairValues = evalFunc(XtTestTune, XtSampleTune{li});          
            meanEvalTune(li) = meanEvalTune(li) + mean(vertcat(pairValues{:})); % mean([pairValues{:}]);
        end
        fprintf('<--- End of rep %d/%d ---\n', r, nreps);
    end
    meanEvalTune = meanEvalTune / nreps;
    
    %fprintf('    %10s, %10s, %10s\n','paramVec', 'meanPair', 'medianPair');
    %fprintf('    %10.5g, %10.5g, %10.5g\n', [paramVec;meanPairTune;medianPairTune]);
    [~, minI] = min(meanEvalTune);
    fLambda = lamVec(minI);
    timing.tune.total = toc(ts);
    fprintf('\n    Selected parameter = %d with mean eval value of = %g\n', ... 
        fLambda, meanEvalTune(minI));
end
fprintf('  << Finished hyperparameter tuning >>\n\n');


%% Use final param value to train and sample
% save(input_fpath, 'XtTrainTune','lamVec','nSamplesTune')  
% if ~system(sprintf('cd %s && Rscript tune_fit.R', curr_path))
    % fprintf('   R script ran successfully.\n')
% end
% S = load(output_fpath);
[XtSampleS, timing.train, timing.sample] = ... 
    pois.fit_and_sample(XtTrain, lamVec, nSamples); % Use all lamVec since glmnet uses warm starts
fXtSample = XtSampleS{minI};

% fXtSample = S.XtSampleTune{minI};
% timing.train = S.tsModel;
% timing.sample  = S.tsSample; 
% delete(input_fpath)
% delete(output_fpath)
