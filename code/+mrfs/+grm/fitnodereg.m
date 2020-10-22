function betaSet = fitnodereg( betaSet, x, ZtSet, optsOverride )
if(nargin < 1)
    mrfs.grm.NIPS2016();
    return;
end
tStart = tic;
k = length(betaSet);

%% Default options
opts = [];
opts.maxOuterIter = 30;
opts.maxInnerIter = 2000;
opts.outerThresh = 1e-5;
opts.stepMaxIter = 20;
opts.stepParam = 0.25;
opts.stepRuleParam = 1e-6;
opts.lam = [0, 1e-3*ones(1,k-1)];
opts.baseDist = mrfs.grm.univariate.Poisson();
opts.nQuery = max(20, length(betaSet) + 1);
opts.newtonMethod = true;
opts.eigMinHessian = 1e-4;
if(nargin >= 4 && isstruct(optsOverride))
    for f = fieldnames(optsOverride)'
        opts.(f{1}) = optsOverride.(f{1});
    end
end
% Expand lambda and assume independent term has 0 lambda
if(k > 1 && isscalar(opts.lam))
    opts.lam = [0, opts.lam*ones(1,k-1)];
end
% Index set needed for Newton optimization
indexSet = betaSet;
for j = 1:k; indexSet{j}(:) = j; end

%% Initialize with mean
betaZero = true;
for j = 2:k
    if nnz(betaSet{j}) > 0
        betaZero = false;
        break;
    end
end
if(betaZero)
    if(mean(x) == 0)
        betaSet{1} = -100;
        warning('  mean(x) == 0, i.e. no signal, just returning beta = -100 instead of log(0) = -Inf');
        return;
    else
        betaSet{1} = log(mean(x));
    end
end

%% Handle trivial case
onlyNode = true;
for j = 2:k
    if nnz(ZtSet{j}) > 0
        onlyNode = false;
        break;
    end
end
if onlyNode; betaSet{1} = log(mean(x)); return; end

%% Initial setup
% Get X matrix of sufficient statistics for node conditional
TXcell = cell(1,k);
for j = 1:length(betaSet)
    TXcell{j} = x.^(1/j);
end
TX = cell2mat(TXcell);

% Compute value, gradient and Hessian of A
Eta = computeEta( betaSet, ZtSet );
[Au,Al] = computeA(Eta, opts.nQuery, opts.baseDist );
f = computef(TX, Eta, Au, betaSet, opts.lam);
for iter = 1:opts.maxOuterIter
    
    %% Compute gradient (only optimization set has non-zero ZtSet)
    [gradSet, Mugrad, Mlgrad] = computeGradSet( Eta, betaSet, TX, ZtSet, Au, Al, opts.nQuery, opts.baseDist );
        
    %% Newton direction
    if(opts.newtonMethod)
        % Get free set and gradient vector on free set
        freeSet = getFreeSet( betaSet, gradSet, opts.lam );
        betaFree = getSubset( freeSet, betaSet );
        gradFree = getSubset( freeSet, gradSet );
        indexFree = getSubset( freeSet, indexSet ); 
        hessianFree = computeHessianFree( freeSet, Eta, ZtSet, Au, Al, Mugrad, Mlgrad, opts.nQuery, opts.baseDist, opts.eigMinHessian );
        %fprintf('gradFree\n');
        %disp(gradFree');
        %fprintf('hessianFree\n');
        %disp(hessianFree);
        %fprintf('eigenvalues of hessian:\n');
        %disp(eig(hessianFree)');
        %fprintf('  Minimum eigenvalue of hessian = %g\n', min(eig(hessianFree)));

        % Loop to solve Newton direction
        dFree = zeros(size(gradFree));
        dTimesHessian = zeros(size(dFree));
        for newtonIter = 1:opts.maxInnerIter
            % Loop through each coordinate in free
            for fi = 1:length(dFree)
                % Solve closed form
                a = hessianFree(fi,fi);
                b = gradFree(fi) + dTimesHessian(fi);
                c = dFree(fi) + betaFree(fi);
                z = c - b/a;
                mu = -c + sign(z)*max(abs(z) - opts.lam(indexFree(fi))/a, 0);
                % Update
                if(mu ~= 0)
                    dFree(fi) = dFree(fi) + mu;
                    dTimesHessian = dTimesHessian + mu*hessianFree(:,fi); % Maintain dTimesHessian
                end
                %assert(~any(isnan(dFree(:))),'Some NaN in testBetaSet');
            end
        end
        % Put dFree back into dirSet
        dirSet = betaSet; % Just make a copy for correct structure of dirSet
        dirSet = setSubset( dirSet, freeSet, dFree );
    else
        % Simple gradient direction
        dirSet = scaleBetaSet( -1, gradSet ); % Negative gradient direction
    end
    
    %% Check step size
    foundStep = false;
    for stepI = 1:opts.stepMaxIter
        step = opts.stepParam^(stepI-1);

        % Compute objective
        if(opts.newtonMethod)
            testBetaSet = addBetaSet( step, betaSet, dirSet );
        else
            testBetaSet = prox( addBetaSet( step, betaSet, dirSet ), opts.lam );
        end
        %assert(~any(isnan(testBetaSet{1}(:))),'Some NaN in testBetaSet');
        testEta = computeEta( testBetaSet, ZtSet );
        [testAu, testAl] = computeA( testEta, opts.nQuery, opts.baseDist );
        testf = computef( TX, testEta, testAu, testBetaSet, opts.lam );
        
        % Step condition
        diffTest = addBetaSet( -1, testBetaSet, betaSet);
        actualDir = scaleBetaSet( 1/step , diffTest ); % Actual direction Bnew = B + step*actualDir
        if(testf < f && testf < f + step*opts.stepRuleParam*innerProductSet( actualDir, gradSet ) )
            foundStep = true;
            break;
        end
    end
    
    %% Check for convergence
    if(foundStep)
        relDiff = (f-testf)/abs(testf);
    else
        relDiff = 0;
    end
    if(relDiff < 0); warning('Reldiff is negative but should be positive'); end
    if(relDiff < opts.outerThresh)
        break;
    end
    
    if(relDiff > 0)
        betaSet = testBetaSet;
        Eta = testEta;
        Au = testAu;
        Al = testAl;
        f = testf;
    end
    
end

nnzBeta = NaN(length(betaSet),1);
for j = 1:length(betaSet)
    nnzBeta(j) = nnz(betaSet{j});
end

lastIterFormat = sprintf('%%%dd/%%%dd', floor(log10(opts.maxOuterIter))+1, floor(log10(opts.maxOuterIter))+1);
fprintf(['%slastIter = ',lastIterFormat,', time = %.2e s, nnz = %s, AMaxRelDiff = %.2e, f = %.2e, prevf = %.2e, relDiff = %.2e, parWorkerId = %d \n'], ...
    opts.outPrefix, iter, opts.maxOuterIter, toc(tStart), mat2str(nnzBeta), max((Au-Al)./abs(Al)), f, testf, relDiff, get(getCurrentTask,'ID'));

end

function freeSet = getFreeSet( betaSet, gradSet, lam )
    freeSet = cell(size(gradSet));
    for j = 1:length(gradSet)
        freeSet{j} = find(betaSet{j} ~= 0 | abs(gradSet{j}) > lam(j));
    end
end

function vecSet = setSubset( vecSet, subSet, subVec )
    curI = 1;
    for j = 1:length(vecSet)
        vecSet{j}(:) = 0; % First ensure everything is 0
        subLength = length(subSet{j});
        vecSet{j}(subSet{j}) = subVec(curI:(curI+subLength-1));
        curI = curI + subLength;
    end
end

function subVec = getSubset( subSet, vecSet )
    freeCell = cell(size(subSet));
    for j = 1:length(vecSet)
        freeCell{j} = vecSet{j}(subSet{j});
    end
    subVec = full(cell2mat(freeCell(:)));
end

function newBetaSet = addBetaSet( scale, betaSet, dSet )
    newBetaSet = cell(size(betaSet));
    for j = 1:length(betaSet)
        newBetaSet{j} = betaSet{j} + scale*dSet{j};
    end
end

function newBetaSet = scaleBetaSet( scale, betaSet )
    newBetaSet = cell(size(betaSet));
    for j = 1:length(betaSet)
        newBetaSet{j} = scale*betaSet{j};
    end
end

function y = innerProductSet( setOne, setTwo )
    y = 0;
    for j = 1:length(setOne)
        y = y + setOne{j}'*setTwo{j};
    end
end

function betaSet = prox( betaSet, lam )
    % Proximal projection (soft threshold)
    for j = 1:length(betaSet) % Only 2 and more
        betaSet{j} = sign(betaSet{j}).*max(abs(betaSet{j})-lam(j),0);
    end
end

function Eta = computeEta(betaSet, ZtSet)
    k = length(betaSet);
    n = size(ZtSet{1},1);
    Eta = zeros(n,k);
    for j = 1:length(betaSet)
        Eta(:,j) = ZtSet{j} * betaSet{j};
    end
    assert(~any(isnan(Eta(:))),'Some NaN in Eta');
    assert(~any(isinf(Eta(:))),'Some Inf in Eta');
end

function f = computef( TX, Eta, A, betaSet, lam )
    % Regularization term
    r = 0;
    for j = 1:length(betaSet) % Skip j=1 term
        r = r + lam(j)*sum(abs(betaSet{j}));
    end
    
    % Normal term
    n = size(TX,1);
    f = (1/n)*sum( -sum(Eta.*TX,2) + A) + r;
end

function [Au,Al] = computeA( Eta, nQuery, baseDist )
    [Au,Al] = baseDist.approxA( Eta, nQuery );
end

function [gradSet, Mugrad, Mlgrad] = computeGradSet( Eta, betaSet, TX, ZtSet, Au, Al, nQuery, baseDist )
    n = size(TX,1);
    [gradAu, gradAl, Mugrad, Mlgrad] = baseDist.approxGradA( Eta, nQuery, Au, Al ); % N x K matrix of gradients
    gradSet = cell(size(betaSet));
    for j = 1:length(betaSet)
        gradSet{j} = (1/n)*((-TX(:,j)+ gradAu(:,j) )'*ZtSet{j})';
    end
    %assert(gradSet{1}(1) < 100, 'gradSet is really large');
end

function hessianFree = computeHessianFree( freeSet, Eta, ZtSet, Au, Al, Mugrad, Mlgrad, nQuery, baseDist, eigMin )
    k = length(freeSet);
    n = size(Eta,1);
    % Compute hessian of A
    [hessianAu, hessianAl] = baseDist.approxHessianA( Eta, nQuery, Au, Al, Mugrad, Mlgrad );
    
    hessianFreeCell = cell(k,k);
    % Compute each block of the hessian
    for j1 = 1:k
        for j2 = 1:j1
            hessianFreeCell{j1,j2} = full(ZtSet{j1}(:,freeSet{j1})' * bsxfun(@times, ZtSet{j2}(:,freeSet{j2}), (1/n)*hessianAu(:,j1,j2)));
            hessianFreeCell{j2,j1} = hessianFreeCell{j1,j2}';
        end
    end
    hessianFree = cell2mat(hessianFreeCell);
    
    % Correct for infinities (also NaN just so it doesn't error)
    maxHessian = 1e10;
    hessianFree(hessianFree > maxHessian) = maxHessian;
    hessianFree(isnan(hessianFree)) = eigMin;
    
    % Ensure positive definiteness and well-conditioned hessian
    hessianFree = real((hessianFree + hessianFree')./2);
    [V, D] = eig(hessianFree);
    diagD = diag(D);
    diagD(diagD < eigMin) = eigMin;
    hessianFree = V*diag(diagD)*V';
    hessianFree = real((hessianFree + hessianFree')./2);
end
