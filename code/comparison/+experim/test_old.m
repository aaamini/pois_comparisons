function cvArray = test(methodName, XtAll, nSamples, nCV, tuneParamVecOverride, nWorkers)
% Run the training and test on a single method

    % if(nargin < 3); metric = 'mmd'; end
    if(nargin < 4); nSamples = 1000; end
    if(nargin < 5); nCV = 3; end
    if(nargin < 6); tuneParamVecOverride = []; end

    %% Setup
    rng default; % For reproducibility
    [n,p] = size(XtAll); % Extract dimensions from dataset
    N_LAM = 10;
    
    initParallel(nWorkers);
    
    % Split into CV train and test sets
    cvArray = experim.create_cv_array(nCV,1);

    rndSeed = 1; % For reproducibility
    [trainIdxArray, testIdxArray] = mrfs.utils.cvsplit( n, nCV, rndSeed );

    sigmaVec = 10.^(-2:0.2:2)';
    nBasisPair = 2^6;% Number of basis for MMD approximation
    evalFunc = @(X,Y) pair_complement_mmd(X, Y, sigmaVec, nBasisPair);
    max_nPairs = 200; % limit the max number of pairs for a "rough" estimate
    evalFuncRough = @(X,Y) pair_complement_mmd(X, Y, sigmaVec, nBasisPair, max_nPairs); % rough version, used for tuning
    
    modMethodName = methodName;

    %% Run cross validation splits (most models have internal parallel programming)
    % parfor cvi = 1:nCV
    for cvi = 1:nCV
        %% Setup train/test split for this CV split
        XtTrain = XtAll(trainIdxArray{cvi},:);
        XtTest = XtAll(testIdxArray{cvi},:);
        
        lamMaxXX = full((1/size(XtTrain,1))*max(max(abs(triu(XtTrain'*XtTrain,1)))));
        [nTrain, pTrain] = size(XtTrain);
        
        lamScaleVec = logspace(0,-4, N_LAM);
    
        %% Fit model and get samples
        tuneTime = []; % Default unless reset later
        model = struct();  % save information about the fitted model, can be left unimplemented
        fprintf('\n\n<< Starting CV = %d for model %s >>\n', cvi, methodName );
        switch modMethodName

            case 'po-is-glmnet'
                pois_coord = false;
                pois_method = 'glmnet'; % other options 'bayesglm' and 'firth'
                [XtSample, trainTime, sampleTime] = ...
                    pois.wrapper(XtTrain, nSamples, pois_method, pois_coord);

                % fileparts(mfilename('fullpath'))
                % bisect_flag = false;
                % pois_method = "glment";
                % save('../../../data.mat','XtTrain','nSamples', 'bisect_flag','pois_method')
                % % system('Rscript ../../run_potts_ising.R')
                % system('Rscript ../../run_potts_ising.R');
                % S = load('../../../output.mat');
                % XtSample = S.XtSample;
                % trainTime = S.trainTime; 
                % sampleTime = S.sampleTime;
            case 'po-is-tune'
                lamVec = 10.^linspace(-4,-1.3, 12);
                % lamVec = 10.^linspace(-3,-1, 12);
                testPect = 0.3;
                nreps = 2;
                [XtSample, fLambda, timing, meanEvalTune]  = ... 
                    pois.tune_fit(lamVec, XtTrain, nSamples, testPect, nreps, evalFuncRough);
                
                model.lambda = fLambda;
                model.meanEvalTune = meanEvalTune;
                trainTime = timing.train; 
                tuneTime = timing.tune;
                sampleTime = timing.sample; 
            
            % case 'pois-g-glmnet'
            %     [XtSample, trainTime, sampleTime] = pois_wrapper(XtTrain, nSamples, 'glmnet', false);
            % case 'pois-1.5-glmnet'
            %     [XtSample, trainTime, sampleTime] = pois_wrapper(XtTrain, nSamples, 'glmnet', true);
            % case 'pois-g-bayes'
            %     [XtSample, trainTime, sampleTime] = pois_wrapper(XtTrain, nSamples, 'bayesglm', false);
            % case 'pois-1.5-bayes'
            %     [XtSample, trainTime, sampleTime] = pois_wrapper(XtTrain, nSamples, 'bayesglm', true);
            % case 'pois-g-firth'
            %     [XtSample, trainTime, sampleTime] = pois_wrapper(XtTrain, nSamples, 'firth', false);
            % case 'pois-1.5-firth'
            %     [XtSample, trainTime, sampleTime] = pois_wrapper(XtTrain, nSamples, 'firth', true);
            
           
            case 'ind-mult'
                % [P, Levels, trainTime] = IndMult.fit(XtTrain);
                % [XtSample, sampleTime] = IndMult.sample(nSamples, P, Levels);
                % model.P = P;
                % model.Levels = Levels;
                [model, trainTime] = IndMult.fit(XtTrain);
                [XtSample, sampleTime] = IndMult.sample(nSamples, model);
                
                
            case 'ind-negbin'
                [model, trainTime] = IndNegBin.fit(XtTrain);
                [XtSample, sampleTime] = IndNegBin.sample(nSamples, model);

            case 'copula-mult'
                [model, trainTime] = CoupulaMult.fit(XtTrain);
                [XtSample, sampleTime] = CoupulaMult.sample(nSamples, model);
                % model.rohat = rhohat;
                % model.mnP = mnP;
                % model.Levels = Levels;
                
            case 'copula-poi'
                copulaType = 'Gaussian';
                [model, trainTime] = CoupulaPoi.fit(Xt, copulaType);
                [XtSample, sampleTime] = CoupulaPoi.sample(nSamples, model);
            
            case 'bootstrap'
                ts = tic;
                XtSample = XtTrain(randsample(size(XtTrain,1), nSamples, true),:);
                sampleTime = toc(ts);
                trainTime = 0;

            case 'mixture-tune'
                %% Setup functions
                paramVec = 10:10:100;
                trainFunc = @( Xt, param, model) MixPoi.fit( XtTrain, param );
                sampleFunc = @( model, nSamples )  MixPoi.sample( model, nSamples );
                
                %% Call tuning function
                [model, XtSample, fParam, timing, ~] = ...
                    experim.tune(trainFunc, paramVec, sampleFunc, XtTrain, nSamples, nCV, evalFuncRough, tuneParamVecOverride);
                
                %% Save parameters
                % model.poissMean = model.poissMean;
                % model.pVec = model.pVec;
                model.k = fParam;
                trainTime = timing.train; 
                tuneTime = timing.tune;
                sampleTime = timing.sample;        
            
            case 'tpgm-tune'
                %% Setup functions
                paramVec = lamScaleVec * lamMaxXX;
                R = full(round(quantile(XtTrain(XtTrain>0),0.99)));
                nThreads = 20;
                trainFunc = @( Xt, lam, model) tpgm.fit( Xt, 'tpgm', lam, nThreads, R );
                
                nGibbs = 5000;
                sampleFunc = @( model, nSamples ) ...
                    tpgm.sample(nSamples, length(model.thetaNode), R, model.thetaNode, model.thetaEdge, nGibbs);
                
                %% Call tuning function
                [model, XtSample, fParam, timing, ~] = ...
                    experim.tune(trainFunc, paramVec, sampleFunc, XtTrain, nSamples, nCV, evalFuncRough, tuneParamVecOverride);
                
                %% Save parameters
                % model.thetaNode = model.thetaNode;
                % model.thetaEdge = model.thetaEdge;
                model.lam = fParam;
                trainTime = timing.train; tuneTime = timing.tune;
                sampleTime = timing.sample;
                    

            otherwise
                error('Model %s not implemented (did you forget to add the suffix ''-tune''\n to the methods that require tuning?\n', methodName);
        end
        
        %% Compute test statistics
        evalMetricValues = evalFunc(XtTest, XtSample);
        
        % fprintf('Computing m-way mmd ... '); tic
        % cvArray(cvi).mwaymmd = mwaymmd(XtTest, XtSample);
        % fprintf('done in %2.2f\n', toc)

        % Save some values
        cvArray(cvi).methodName = methodName;
        cvArray(cvi).model = model;
        cvArray(cvi).tuneTime = tuneTime;
        cvArray(cvi).trainTime = trainTime;
        cvArray(cvi).sampleTime = sampleTime;
        cvArray(cvi).evalMetricValues = evalMetricValues;
        cvArray(cvi).evalParams = struct('sigmaVec',sigmaVec,'nBasisPair',nBasisPair);
        
        
        % Print out some results
        % fiveNum = num2cell(quantile(evalMetricValues,[0,0.25,0.5,0.75,1]));
        meanPair = mean(evalMetricValues);
        fprintf('  Model = %s\n', methodName);
        % fprintf('  Metric = %s\n', metric);
        fprintf('  Mean Pair Value = %.4g\n', meanPair);
        % fprintf('  Min Q1 Med Q2 Max of Pair Values = \n      [%.4g, %.4g, %.4g, %.4g, %.4g]\n', fiveNum{:});
        fprintf('  Total hyperparameter tuning time = %g s\n',trainTime);
        fprintf('  Train time = %g s\n',trainTime);
        fprintf('  Sample time = %g s\n', sampleTime);
        fprintf('<< Finished CV = %d >>\n', cvi);
    end
end