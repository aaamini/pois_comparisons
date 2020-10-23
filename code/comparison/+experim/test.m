function cvArray = test(methodIdx, XtAll, exset)
% Run the training and test on a single method

    % if(nargin < 3); metric = 'mmd'; end
    if(nargin < 4); nSamples = 1000; end
    if(nargin < 5); nCV = 3; end
    if(nargin < 6); tuneParamVecOverride = []; end

    
    %% Setup
    rng default; % For reproducibility
    [n,p] = size(XtAll); % Extract dimensions from dataset
    N_LAM = 10;
    nSamples = exset.nSamples;
    nCV = exset.nCV;
    mmdAggType = exset.mmdAggType;
    nWorkers = exset.nWorkers;
    tuneParamVecOverride = [];
    methodName = exset.batch{methodIdx};

    fprintf('Running method: %s ... \n', methodName);
    
    % Split into CV train and test sets
    cvArray = experim.create_cv_array(nCV,1);

    rndSeed = 1; % For reproducibility
    [trainIdxArray, testIdxArray] = mrfs.utils.cvsplit( n, nCV, rndSeed );

    sigmaVec = exset.sigmaVec; % 10.^(-2:0.2:2)';
    nBasisPair = 2^6;% Number of basis for MMD approximation
    evalFunc = @(X,Y) pair_complement_mmd(X, Y, sigmaVec, nBasisPair, Inf, mmdAggType);
    %evalFunc = @(X,Y) pair_complement_mmd(X, Y, sigmaVec, nBasisPair, 400, mmdAggType);
    max_nPairs = 200; % limit the max number of pairs for a "rough" estimate
    evalFuncRough = @(X,Y) pair_complement_mmd(X, Y, sigmaVec, nBasisPair, max_nPairs, mmdAggType); % rough version, used for tuning
    
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
        model = struct();  % save information about the fitted model, can be left unimplemented
        timing = struct('train', 0, 'sample', 0, ... 
            'tune', struct('model', 0, 'sample', 0, 'total', 0));
        % timing.tune = []; % Default unless reset later
        fprintf('\n\n<< Starting CV = %d for model %s >>\n', cvi, methodName );
        switch modMethodName

            case 'po-is-tune'
                lamVec = 10.^linspace(-4,-1.3, 15);
                testPect = 0.3;
                nreps = 3;
                [XtSample, fLambda, timing, meanEvalTune]  = ... 
                    pois.tune_fit(lamVec, XtTrain, nSamples, testPect, nreps, evalFuncRough);
                
                model.lambda = fLambda;
                model.meanEvalTune = meanEvalTune;
                
           
            case 'ind-mult'
                [model, timing.train] = IndMult.fit(XtTrain);
                [XtSample, timing.sample] = IndMult.sample(nSamples, model);
                
                
            case 'ind-negbin'
                [model, timing.train] = IndNegBin.fit(XtTrain);
                [XtSample, timing.sample] = IndNegBin.sample(nSamples, model);

            case 'copula-mult'
                [model, timing.train] = CoupulaMult.fit(XtTrain);
                [XtSample, timing.sample] = CoupulaMult.sample(nSamples, model);
                
                
            case 'copula-poi'
                copulaType = 'Gaussian';
                [model, timing.train] = CoupulaPoi.fit(XtTrain, copulaType);
                [XtSample, timing.sample] = CoupulaPoi.sample(nSamples, model);
            
            case 'bootstrap'
                ts = tic;
                XtSample = XtTrain(randsample(size(XtTrain,1), nSamples, true),:);
                timing.sample = toc(ts);
                timing.train = 0;

            case 'mixture-tune'
                %% Setup functions
                % paramVec = 10:10:100;
                paramVec = round(logspace(log10(5), log10(30), 5)); % number of mixture components
                % paramVec = 5:5:30;
                % paramVec = 10:10:50;
                trainFunc = @( Xt, param, model) MixPoi.fit( XtTrain, param );
                sampleFunc = @( model, nSamples )  MixPoi.sample( model, nSamples );
                
                %% Call tuning function
                [model, XtSample, fParam, timing, ~] = ...
                    experim.tune(trainFunc, paramVec, sampleFunc, XtTrain, nSamples, nCV, evalFuncRough, tuneParamVecOverride);
                
                model.k = fParam;
            
            
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
                
                model.lam = fParam;
                

            case 'poisson-sqr-tune'
                %% Setup functions
                paramVec = lamScaleVec * sqrt(lamMaxXX);
                %iif = @(varargin) varargin{2*find([varargin{1:2:end}], 1, 'first')}();
                %trainFunc = @(Xt, param, grm) iif( ...
                %    nargin < 3 || isempty(grm),  @() mrfs.grm.fitgrm( Xt, 2, struct('lam', param) ), ...
                %    true,  @() mrfs.grm.fitgrm( Xt, 2, struct('lam', param, 'grm', grm ) )...
                %    );
                trainFunc = @(Xt, param, grm) mrfs.grm.fitgrm( Xt, 2, struct('lam', param ) );
                
                nGibbs = 5000; nInner = 2; nBatches = 20;
                sampleFunc = @( model, nSamples ) psqr.sample( model.getPsiExt(), nSamples, nGibbs, nInner, nBatches );
                
                %% Call tuning function
                % tuneParamVecOverride = [1];
                [temp_model, XtSample, fParam, timing, ~] = ...
                    experim.tune(trainFunc, paramVec, sampleFunc, XtTrain, nSamples, nCV, evalFuncRough, tuneParamVecOverride);
                % temp_model is a class in this case, can not dynamically assign fields to it
            
                %% Save parameters
                model.PsiExt = temp_model.getPsiExt();
                model.lam = fParam;
                model.lamMax = sqrt(lamMaxXX);
                % timing.train = timing.train; 
                % timing.tune = timing.tune;
                % timing.sample = timing.sample;
                    

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
        cvArray(cvi).timing = timing;
        % cvArray(cvi).timing.tune = timing.tune;
        % cvArray(cvi).timing.train = timing.train;
        % cvArray(cvi).timing.sample = timing.sample;
        cvArray(cvi).evalMetricValues = evalMetricValues;
        cvArray(cvi).evalParams = struct('sigmaVec', sigmaVec, 'nBasisPair', nBasisPair);
        
        
        % Print out some results
        % fiveNum = num2cell(quantile(evalMetricValues,[0,0.25,0.5,0.75,1]));
        meanPair = mean(vertcat(evalMetricValues{:}));
        fprintf('  Model = %s\n', methodName);
        % fprintf('  Metric = %s\n', metric);
        fprintf('  Mean Pair Value = %.4g\n', meanPair);
        % fprintf('  Min Q1 Med Q2 Max of Pair Values = \n      [%.4g, %.4g, %.4g, %.4g, %.4g]\n', fiveNum{:});
        % fprintf('  Total hyperparameter tuning time = %g s\n',timing.train);
        fprintf('  Total hyperparameter tuning time = %g s\n', timing.tune.total);
        fprintf('  Train time = %g s\n', timing.train);
        fprintf('  Sample time = %g s\n', timing.sample);
        fprintf('<< Finished CV = %d >>\n', cvi);
    end
end