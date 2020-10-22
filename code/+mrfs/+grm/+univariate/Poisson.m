classdef Poisson
    methods (Static)
        
        function XtSample = sampleSQR_SMC( PsiExt, nSamples, nAnneal, nGibbs, nInner )
            
            %% Loop through different distributions
            % Sample from Gibbs
            betaVec = linspace(0,1,nAnneal);
            for bi = 1:length(betaVec)
                beta = betaVec(bi);
                % Allows for one last gibbs iteration at the end where logW = 0 or weights are uniform
                if(bi + 1 > length(betaVec))
                    betaNext = betaVec(bi);
                    nGibbs = nAnneal*nGibbs;
                else
                    betaNext = betaVec(bi+1);
                end
                modPsiExt = getPsiExt(PsiExt,beta);
                modPsiExtNext = getPsiExt(PsiExt,betaNext);
                
                if(bi == 1)
                    XtSample = poissrnd(repmat(exp(PsiExt{1,1}'), nSamples, 1));
                    %logW = zeros(nSamples,1);
                else
                    XtSample = mrfs.grm.univariate.Poisson.sampleSQR_Gibbs( ...
                        modPsiExt, nSamples, nGibbs, nInner, XtSample );
                    % Determine weights
                    %logW(i) = logW(i) + logpmrfprop(x,thetaNode,thetaEdge*betaNext) - logpmrfprop(x,thetaNode,thetaEdge*beta)
                    logW = logSqrProp(XtSample, modPsiExtNext) - logSqrProp(XtSample, modPsiExt);
                    w = exp(logW-max(logW));
                    w = w./sum(w);
                    
                    % Resample based on weights
                    if(var(w) > 0)
                        weightedSample = XtSample;
                        resampleI = mnrnd(nSamples, w);
                        curI = 0;
                        for ii = 1:length(resampleI)
                            idx = (curI+1):(curI+resampleI(ii));

                            XtSample(idx,:) = repmat(weightedSample(ii,:), resampleI(ii), 1);
                            curI = curI+resampleI(ii);
                        end
                    end
                    %mrfs.grm.NIPS2016.visXt( XtSample, 15 );
                    %fprintf('Vis\n');
                end
            end
            
            function logProp = logSqrProp(Xt, PsiExt)
                sqrtXt = sqrt(Xt);
                logProp = Xt*PsiExt{1,1} + Xt*PsiExt{1,2} + sum( (sqrtXt*PsiExt{2,2}).*sqrtXt, 2 );
            end
            
            function modPsiExt = getPsiExt(PsiExt, beta)
                modPsiExt = PsiExt;
                k = size(PsiExt,2);
                for j = 1:k
                    if(j == 1); continue; end % Don't change PsiExt
                    for ell = 1:j
                        modPsiExt{ell,j} = beta*PsiExt{ell,j};
                    end
                end
            end
        end
        
        function XtSample = sampleSQR_Gibbs( PsiExt, nSamples, nGibbs, nInner, XtSample )
            if(nargin < 5); XtSample = poissrnd(repmat(exp(PsiExt{1,1}'), nSamples, 1)); end
            k = size(PsiExt,1);
            assert(k == 2, 'This only works when k=2 (i.e. SQR model)');
            p = length(PsiExt{1,1});

            theta = PsiExt{1,2};
            Phi = PsiExt{2,2};
            PhiDiag = PsiExt{1,1};
            
            propLam = @(xs) max(0.3,xs);
            if(nnz(Phi) == 0 && nnz(theta) == 0)
                XtSample = poissrnd(repmat(exp(PsiExt{1,1}'), nSamples, 1));
                return; % No more sampling needed since independent model
            end
            
            sqrtXtSample = sqrt(XtSample);
            nAccept = 0;
            nTotal = 0;
            for gi = 1:nGibbs
                for s = 1:p
                    for mi = 1:nInner
                        % Generate proposal
                        xs = XtSample(:,s);
                        xsProp = poissrnd(propLam(xs));
                        
                        eta1 = PhiDiag(s);
                        eta2 = theta(s) + 2*sqrtXtSample*Phi(:,s);
                        
                        logPxProp = eta1.*xsProp + eta2.*sqrt(xsProp) - gammaln(xsProp+1);
                        logPx = eta1.*xs + eta2.*sqrt(xs) - gammaln(xs+1);
                        logQxGxProp = log(propLam(xsProp)).*xs - gammaln(xs+1) - propLam(xsProp);
                        logQxPropGx = log(propLam(xs)).*xsProp - gammaln(xsProp+1) - propLam(xs);
                        
                        Aprob = min(1,exp(logPxProp - logPx + logQxGxProp - logQxPropGx));
                        accept = Aprob == 1 | rand(size(Aprob)) < Aprob;
                        XtSample(accept,s) = xsProp(accept);
                        sqrtXtSample(accept,s) = sqrt(xsProp(accept));
                        
                        nAccept = nAccept + sum(accept);
                        nTotal = nTotal + length(accept);
                    end
                end
                %mrfs.grm.NIPS2016.visXt( XtSample, 15 );
                %fprintf('Vis\n');
            end
            %fprintf('Average acceptance rate = %d/%d = %g\n', nAccept, nTotal, nAccept/nTotal);
        end
        
        function XtSample = sampleGRM_Gibbs( PsiExt, nSamples, nGibbs, nInner, XtSample )
            if(nargin < 5); XtSample = poissrnd(repmat(exp(PsiExt{1,1}'), nSamples, 1)); end
            k = size(PsiExt,1);
            p = length(PsiExt{1,1});

            % Fast sampling if independent Poissons
            dependent = false;
            for j = 2:k
                for ell = 1:j
                    if(nnz(PsiExt{ell,j}) > 0)
                        dependent = true;
                    end
                end
            end
            if(~dependent)
                XtSample = poissrnd(repmat(exp(PsiExt{1,1}'), nSamples, 1));
                return; % No more sampling needed since independent model
            end
            
            % Gibbs sampling
            propLam = @(xs) max(0.3,xs);
            nAccept = 0;
            nTotal = 0;
            XtSampleCache = {XtSample, sqrt(XtSample), XtSample.^(1/3)};
            for gi = 1:nGibbs
                for s = 1:p
                    for mi = 1:nInner
                        % Generate proposal
                        xs = XtSample(:,s);
                        xsProp = poissrnd(propLam(xs));
                        
                        etaMat = mrfs.grm.univariate.Poisson.getEtaMat(PsiExt, s, XtSample, XtSampleCache);
                        logPxProp = -gammaln(xsProp+1);
                        logPx = -gammaln(xs+1);
                        for j = 1:k
                            logPxProp = logPxProp + etaMat(:,j).*xsProp.^(1/j);
                            logPx = logPx + etaMat(:,j).*xs.^(1/j);
                        end
                                                
                        logQxGxProp = log(propLam(xsProp)).*xs - gammaln(xs+1) - propLam(xsProp);
                        logQxPropGx = log(propLam(xs)).*xsProp - gammaln(xsProp+1) - propLam(xs);
                        
                        Aprob = min(1,exp(logPxProp - logPx + logQxGxProp - logQxPropGx));
                        accept = Aprob == 1 | rand(size(Aprob)) < Aprob;
                        
                        XtSample(accept,s) = xsProp(accept);
                        XtSampleCache{1}(accept,s) = xsProp(accept);
                        XtSampleCache{2}(accept,s) = sqrt(xsProp(accept));
                        XtSampleCache{3}(accept,s) = xsProp(accept).^(1/3);
                        
                        nAccept = nAccept + sum(accept);
                        nTotal = nTotal + length(accept);
                    end
                end
                %mrfs.grm.NIPS2016.visXt( XtSample, 15 );
                %fprintf('Vis\n');
                
                if(mod(gi,100) == 0); fprintf('.'); end
            end
            fprintf('\n');
            fprintf('Average acceptance rate = %d/%d = %g\n', nAccept, nTotal, nAccept/nTotal);
        end
        
        %% Sample from Poisson using parameters specified by grm
        function XtSample = sample( PsiExt, nSamples, nGibbs, nInner )
            if(nargin < 3); nGibbs = 10; end
            if(nargin < 4); nInner = 10; end
            p = length(PsiExt{1,1});
            XSample = NaN(p,nSamples);
            parfor i = 1:nSamples
                XSample(:,i) = mrfs.grm.univariate.Poisson.gibbs(PsiExt, nGibbs, nInner);
            end
            XtSample = XSample';
        end
        function x = gibbs(PsiExt, nGibbs, nInner)
            x = poissrnd(exp(PsiExt{1,1}));
            p = length(PsiExt{1,1});
            for gi = 1:nGibbs
                for s = 1:p
                    etaVec = mrfs.grm.univariate.Poisson.getEtaMat(PsiExt, s, x);
                    x(s) = mrfs.grm.univariate.Poisson.unimcmc(x(s), etaVec, nInner);
                end
            end
        end
        function xs = unimcmc(xs, etaVec, nInner)
            propLam = @(xs) max(0.3,xs);
            k = length(etaVec);
            for ii = 1:nInner
                % Generate proposal
                xsProp = poissrnd(propLam(xs));
                
                % Compute needed probabilities
                logPxProp = etaVec*xsProp.^(1./(1:k)') - gammaln(xsProp+1);
                logPx = etaVec*xs.^(1./(1:k)') - gammaln(xs+1);
                logQxGxProp = log(propLam(xsProp))*xs - gammaln(xs+1) - propLam(xsProp);
                logQxPropGx = log(propLam(xs))*xsProp - gammaln(xsProp+1) - propLam(xs);
                
                % Check acceptance
                Aprob = min(1,exp(logPxProp - logPx + logQxGxProp - logQxPropGx));
                if(Aprob == 1 || rand(1) < Aprob)
                    xs = xsProp;
                end
            end
        end
        function etaMat = getEtaMat(PsiExt, s, Xt, XtCache)
            if(nargin < 4); XtCache = {Xt, sqrt(Xt), Xt.^(1/3)}; end
            k = size(PsiExt,1);
            etaMat = zeros(size(Xt,1),k);
            for j = 1:k
                for ell = 1:j
                    PsiEllJ = PsiExt{ell, j};
                    if(ell == 1)
                        V = PsiEllJ(s);
                    elseif(ell == 2)
                        if(nnz(PsiEllJ(:,s)) == 0)
                            V = 0;
                        else
                            V = XtCache{2}*PsiEllJ(:,s);
                        end
                    else
                        iCell = [repmat({':'},1,ell-1), s];
                        subPsi = squeeze(PsiEllJ(iCell{:}));
                        if(ell == 3)
                            if(nnz(subPsi) == 0)
                                V = 0;
                            else
                                cubeRootXt = XtCache{3};
                                V = sum((cubeRootXt*subPsi).*cubeRootXt,2);
                            end
                        else
                            error('ell=4 not implemented yet');
                            %T = mrfs.grm.GRMParameters.outerProduct( X.^(1/ell), ell-1 );
                            %v = subPsi(:)'*T(:);
                        end
                    end
                    etaMat(:,j) = etaMat(:,j) + ell*V;
                end
            end
        end
        function testsmc(visualize)
            if(nargin < 1); visualize = true; end
            %% Setup parameters
            PsiExt = { [-1;-1],[0;0];
                       [],[0,2;2,0] };
            
            %% Sample
            nSamples = 1000; nAnneal = 100; nGibbs = 10; nInner = 2;
            %ts = tic;
            %XtSample = mrfs.grm.univariate.Poisson.sample( PsiExt, nSamples, nGibbs, nInner );
            %toc(ts);
            ts = tic;
            XtSample = mrfs.grm.univariate.Poisson.sampleSQR_SMC( PsiExt, nSamples, nAnneal, nGibbs, nInner );
            toc(ts);

            %% Visualize
            if(visualize)
                maxX = 15;
                %mrfs.grm.NIPS2016.visXt3( XtSample, 30, 2, 3, 1:3 );
                subplot(1,2,1);
                mrfs.grm.NIPS2016.visXt( XtSample, maxX );
                subplot(1,2,2);
                mrfs.grm.NIPS2016.visDist( PsiExt, maxX );
            end

        end
        
        function [PsiExt,PsiExtGRM, PsiExtSQR] = testgrmgibbs(visualize)
            if(nargin < 1); visualize = true; end
            
            %% Setup parameters
            p = 3; k = 3;
            PsiExt = { zeros(p,1), zeros(p,1), zeros(p,1);
                       [], zeros(p,p), zeros(p,p);
                       [], [], zeros(p,p,p);
                       };
            PsiExt{1,1}(:) = -3;
            PsiExt{1,2}(:) = 0;
            pairParam = 0;
            PsiExt{2,2}(getInd(1:2,p,2)) = pairParam;
            PsiExt{2,2}(getInd(2:3,p,2)) = pairParam;
            PsiExt{2,2}(getInd([1,3],p,2)) = pairParam;
            PsiExt{3,3}(getInd(1:3,p,3)) = 2;
            
            PsiExtSQR = PsiExt(1:2,1:2);
            
            %% Simulate samples
            nSamples = 1000; nGibbs = 100; nInner = 2;
            ts = tic;
            XtSampleGRMSim = mrfs.grm.univariate.Poisson.sampleGRM_Gibbs( PsiExt, nSamples, nGibbs, nInner );
            toc(ts)
            
            %% Fit GRM and get samples
            sqrLam = 0.01;
            temp = logspace(0,-1,5);
            grmLam = sqrLam;
            ts = tic;
            grm = mrfs.grm.fitgrm( XtSampleGRMSim, 3, struct('lam', [0, sqrLam, grmLam]) );
            PsiExtGRM = grm.getPsiExt();
            XtSampleGRM = mrfs.grm.univariate.Poisson.sampleGRM_Gibbs( PsiExtGRM, nSamples, nGibbs, nInner );
            toc(ts)
            
            
            %% Fit SQR and get samples
            ts = tic;
            sqr = mrfs.grm.fitgrm( XtSampleGRMSim, 2, struct('lam', [0, sqrLam]) );
            PsiExtSQR = sqr.getPsiExt();
            XtSampleSQR = mrfs.grm.univariate.Poisson.sampleSQR_Gibbs( PsiExtSQR, nSamples, nGibbs, nInner );
            toc(ts)
            
            fprintf('Simulated spearman correlation:\n');
            disp(corr(XtSampleGRMSim,'type','spearman'));
            fprintf('SQR spearman correlation:\n');
            disp(corr(XtSampleSQR,'type','spearman'));
            fprintf('GRM spearman correlation:\n');
            disp(corr(XtSampleGRM,'type','spearman'));
            
            sigmaVec = 10.^(-2:0.2:2)';
            nBasis = 2^10;
            [~, sqrMmd] = MMDFourierFeature(XtSampleGRMSim, XtSampleSQR, sigmaVec, nBasis);
            [~, grmMmd] = MMDFourierFeature(XtSampleGRMSim, XtSampleGRM, sigmaVec, nBasis);
            
            if(p == 3)
                visFunc = @visScatter;
                %visFunc = @(XtSample) poissonplotmatrix(struct(), XtSample);
            else
                visFunc = @(XtSample) poissonplotmatrix(struct(), XtSample);
            end
            if(visualize)
                clf;
                subplot(2,2,1);
                visFunc(XtSampleGRMSim);
                title('GRM Data');
                
                subplot(2,2,2);
                visFunc(XtSampleSQR);
                title('Poisson SQR');
                
                subplot(2,2,3);
                visFunc(XtSampleGRM);
                title('Poisson GRM');
                
                subplot(2,2,4);
                semilogx(sigmaVec, [sqrMmd, grmMmd]);
                legend({'SQR','GRM'});
            end
            
            function ind = getInd( exampleInd, p, ell )
                if(ell == 1); ind = exampleInd; return; end % Trivial case
                P = perms(exampleInd);
                P = mat2cell(P, size(P,1), ones(ell,1));
                ind = sub2ind(p*ones(ell,1), P{:} );
            end
        end
        
        function testgibbs(visualize)
            if(nargin < 1); visualize = true; end
            %% Setup parameters
            PsiExt = { [-1;-1],[0;0];
                       [],[0,2;2,0] };
            
            %% Sample
            nSamples = 1000; nGibbs = 2000; nInner = 2;
            %ts = tic;
            %XtSample = mrfs.grm.univariate.Poisson.sample( PsiExt, nSamples, nGibbs, nInner );
            %toc(ts);
            ts = tic;
            XtSample = mrfs.grm.univariate.Poisson.sampleSQR_Gibbs( PsiExt, nSamples, nGibbs, nInner );
            toc(ts);

            %% Visualize
            if(visualize)
                maxX = 15;
                %mrfs.grm.NIPS2016.visXt3( XtSample, 30, 2, 3, 1:3 );
                subplot(1,2,1);
                mrfs.grm.NIPS2016.visXt( XtSample, maxX );
                subplot(1,2,2);
                mrfs.grm.NIPS2016.visDist( PsiExt, maxX );
            end

        end
        function testuni()
            etaVec = [3,-5];
            k = length(etaVec);
            nInner = 200;
            nSamples = 1000;
            X = zeros(nSamples,1);
            for i = 1:nSamples
                X(i) = mrfs.grm.univariate.Poisson.unimcmc(poissrnd(exp(etaVec(1))), etaVec, nInner);
            end
            
            % Plot samples and true density
            dom = (0:40)';
            % Empirical density
            counts = hist(X,dom);
            % Actual density
            X = bsxfun(@power, dom, (1./(1:k)));
            logProp = X*etaVec' - gammaln(dom+1);
            prob = exp(logProp - mrfs.grm.univariate.Poisson.approxA( etaVec, 10));
            % Plot
            bar(dom, [prob, counts'./sum(counts)] );
            legend('Prob','Emp. Prob');
            xlim([-1,max(dom)]);
        end
        
        %% Upper and lower bounds on A
        function [Au, Al] = approxA( Eta, nQuery )
            [Au, Al] = mrfs.grm.univariate.Poisson.approxM(0, Eta, nQuery);
            %fprintf('  Au vs. Al max relDiff = %g\n', max((Au-Al)./abs(Al)));
        end
        
        %% Upper and lower bounds on gradA
        function [gradAu, gradAl, Mu, Ml] = approxGradA( Eta, nQuery, Au, Al )
            [n, k] = size(Eta);
            % Compute needed M functions
            Mu = zeros(n,k); Ml = zeros(n,k);
            for j = 1:k
                [ Mu(:,j), Ml(:,j) ] = mrfs.grm.univariate.Poisson.approxM(1/j, Eta, nQuery );
            end
            %fprintf('  Mugrad vs. Mlgrad max relDiff = %g\n', max((Mu(:)-Ml(:))./abs(Ml(:))));
            
            % Compute grad as expectations
            gradAu = exp(bsxfun(@minus, Mu, Al));
            gradAl = exp(bsxfun(@minus, Ml, Au));
        end
        
        %% Upper and lower bounds on hessianA
        function [hessianAu, hessianAl] = approxHessianA( Eta, nQuery, Au, Al, Mugrad, Mlgrad )
            [n, k] = size(Eta);
            
            % Get needed ratios
            ND = mrfs.grm.univariate.Poisson.getRatios(k);
            nRatios = size(ND,2);
            
            % Compute extra needed functions
            Mu = zeros(n,nRatios); Ml = zeros(n,nRatios);
            for ii = 1:nRatios
                if ND(1,ii) == 1
                    % Copy from previous calculations
                    Mu(:,ii) = Mugrad(:,ND(2,ii));
                    Ml(:,ii) = Mlgrad(:,ND(2,ii));
                else
                    % Compute new expectations
                    [ Mu(:,ii), Ml(:,ii) ] = mrfs.grm.univariate.Poisson.approxM(ND(1,ii)/ND(2,ii), Eta, nQuery );
                end
            end
            %fprintf('  Muhess vs. Mlhess max relDiff = %g\n', max((Mu(:)-Ml(:))./abs(Ml(:))));
            
            % Compute Hessians as expectations
            hessianAu = zeros([n,k,k]);
            hessianAl = zeros([n,k,k]);
            for j1 = 1:k
                for j2 = 1:j1
                    % Get indexes of Mu and Ml needed for Hessian computation
                    ND1 = [1;j1]; ND2 = [1;j2];
                    NDBoth = mrfs.grm.univariate.Poisson.simplifyFrac( ...
                            mrfs.grm.univariate.Poisson.addFrac(ND1, ND2 ) )';
                    i1 = ND(1,:) == ND1(1) & ND(2,:) == ND1(2);
                    i2 = ND(1,:) == ND2(1) & ND(2,:) == ND2(2);
                    iBoth = ND(1,:) == NDBoth(1) & ND(2,:) == NDBoth(2);
                    
                    % Actually compute hessian using indices
                    % Debug code: nRatios = [Mu(:,iBoth)-Al;Ml(:,i1)-Au;Ml(:,i2)-Au]; disp(nRatios(971,:));
                    %hessianAu(:,j2,j1) = exp(Mu(:,iBoth)-Al) - exp(Ml(:,i1)-Au) .* exp(Ml(:,i2)-Au);
                    %hessianAl(:,j2,j1) = exp(Ml(:,iBoth)-Au) - exp(Mu(:,i1)-Al) .* exp(Mu(:,i2)-Al);
                    hessianAu(:,j2,j1) = expDiff( Mu(:,iBoth)-Al, Ml(:,i1) + Ml(:,i2) - 2*Au );
                    hessianAl(:,j2,j1) = expDiff( Ml(:,iBoth)-Au, Mu(:,i1) + Mu(:,i2) - 2*Al );
                    hessianAu(:,j1,j2) = hessianAu(:,j2,j1);
                    hessianAl(:,j1,j2) = hessianAl(:,j2,j1);
                end
            end
            
            function y = expDiff(a,b)
                y = exp(a).*(1-exp(b-a));
            end
        end
        
        %% Get ratios we need
        function ND = getRatios(k)
            NDTcell = cell(k,k);
            for j1 = 1:k
                for j2 = 1:j1
                    ND1 = [1; j1];
                    ND2 = [1; j2];
                    NDTcell{j1,j2} = mrfs.grm.univariate.Poisson.simplifyFrac( ...
                            mrfs.grm.univariate.Poisson.addFrac(ND1, ND2 ) )';
                end
            end
            NDT = [ones(k,1),(1:k)'; cell2mat(NDTcell(:)) ];
            ND = unique(NDT,'rows')';
            %fprintf('Num duplicate fractions = %d\n',size(NDT,1)-size(ND,2));
        end
        
        function ND = simplifyFrac(ND)
            ND = round(ND./gcd(ND(1),ND(2)));
        end

        function ND = addFrac(ND1,ND2)
            newD = lcm(ND1(2),ND2(2));
            newN = ND1(1)*round(newD/ND1(2)) + ND2(1)*round(newD/ND2(2));
            ND = [newN; newD];
        end
        
        %% Main important function
        function [Mu, Ml] = approxM(a, Eta, nQuery, isVis)
            %if(size(Eta,1) == 1); Eta = [Eta, zeros(size(Eta))]; end % Weird trivial case... don't remember why (might be able to remove)
            [n,k] = size(Eta);
            minDescarte = k - 1; % Minimum number of real zeros based on descartes rule of signs
            if(nargin < 3 || isempty(nQuery)); nQuery = minDescarte + 2; end
            if(nargin < 4 || isempty(isVis)); isVis = false; end
            % Including endpoints (supposing k-1 positive roots of ddg)
            assert(nQuery >= minDescarte + 2, 'nQuery too small, needs to be at least (k-1) + 2');
            
            % Get J set (skip 1 and only non-zero values)
            JNz = 2:k;
            nonZeros = sum(Eta(:,JNz) ~= 0,1) > 0;
            JNz = JNz(nonZeros);
            if(isempty(JNz))
                JNz = 2; % At least one index
            end
            
            % Loop through each eta
            nRegions = nQuery - 1;
            R = mrfs.grm.univariate.RegionMat( repmat(Eta(:,1),1,nRegions) );

            %% Setup qMat
            % Find roots of g'' (based on roots of expanded polynomial)
            rootsMatT = mrfs.grm.univariate.findRoots( ddgCoeff(Eta), JNz );
            qMat = [rootsMatT',ones(n,1),inf(n,1)]; % Add endpoints
            qMat = uniqMat(qMat);
            concavityMat = testRegions( qMat, Eta );
            
            % Filter to only columns that have nonNaN values
            nonNaNCols = any(~isnan(qMat),1);
            qMat = qMat(:,nonNaNCols);
            concavityMat = concavityMat(:,nonNaNCols);
            
            %% Add regions incrementally
            %disp(qMat(1,:));
            for ri = 1:nRegions
                R = addRegion( R, ri, Eta, qMat, concavityMat);
                if(isVis); visualize(R, a, Eta ); end
                %fprintf('<<< ri = %d, n = %d >>>\n',ri,n);
                %disp([R.minX(1,:);R.maxX(1,:)]);
            end
            
            %% Get sum of regions
            Mu = mrfs.utils.logsumexp([log(zeros(n,1).^a), R.logU], 2); % zero is for x = 0, Pr(x) propto exp(0)
            Ml = mrfs.utils.logsumexp([log(zeros(n,1).^a), R.logL], 2);
            
            
            function Xuniq = uniqMat(X)
                Xsort = sort(X,2);
                diffX = [diff(Xsort,1,2), ones(size(Xsort,1),1)];
                dupSel = diffX == 0;
                Xsort(dupSel) = NaN; % Make all duplicates NaN
                Xuniq = sort(Xsort,2); % Sort again to make unique
                
                % Naive check
                %{
                Xuniq2 = NaN(size(X));
                for ii = 1:size(X,1)
                    temp = unique(X(ii,:));
                    Xuniq2(ii,1:length(temp)) = temp;
                end
                
                assert(all( Xuniq(:)==Xuniq2(:) | (isnan(Xuniq(:)) & isnan(Xuniq2(:))) ),'Uniq does not work');
                %}
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%% Nested Functions %%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            function y = fg(x,Eta)
                % Expand eta if the same
                if(size(Eta,1) == 1 && length(x) > 1)
                    Eta = repmat(Eta,length(x),1);
                end
                y = sum((bsxfun(@power, x, 1./(1:k))).*Eta,2) + log(x.^a) - gammaln(x+1);
            end
            
            function y = g(x,Eta)
                % Expand eta if the same
                if(size(Eta,1) == 1 && length(x) > 1)
                    Eta = repmat(Eta,length(x),1);
                end
                y = sum((bsxfun(@power, x, 1./JNz)).*Eta(:,JNz),2) + log(x.^a);
            end
            function y = dg(x,Eta)
                % Expand eta if the same
                if(size(Eta,1) == 1 && length(x) > 1)
                    Eta = repmat(Eta,length(x),1);
                end
                %y = (1./x).*( (bsxfun(@power, x, 1./JNz)) * (eta(JNz)./JNz') + a);
                y = (1./x).*sum( (bsxfun(@power, x, [1./JNz,0])) .* dgCoeff(Eta), 2);
            end
            function y = ddg(x,Eta)
                if(isempty(x)); y = []; return; end
                % Expand eta if the same
                if(size(Eta,1) == 1 && length(x) > 1)
                    Eta = repmat(Eta,length(x),1);
                end
                %y = (1./x.^2).*( (bsxfun(@power, x, 1./JNz)) * ( eta(JNz)./JNz.*(-(JNz-1)./JNz) ) - a);
                %y = (1./x.^2).*( (bsxfun(@power, x, 1./JNz)) * ( -eta(JNz)./JNz'.^2.*(JNz'-1) ) - a );
                y = (1./x.^2).*sum( (bsxfun(@power, x, [1./JNz,0])) .* ddgCoeff(Eta),2 );
            end
            
            function coeff = dgCoeff(Eta)
                coeff = [bsxfun(@times, Eta(:,JNz), 1./JNz), a*ones(size(Eta,1),1)];
            end
            function coeff = ddgCoeff(Eta)
                coeff = -[bsxfun(@times, Eta(:,JNz), 1./JNz.^2.*(JNz-1)), a*ones(size(Eta,1),1)];
                assert(sum(sum(isnan(coeff))) == 0 && sum(sum(isinf(coeff))) == 0 , 'Inf or NaN in coeff');
            end
            
            function concavityMat = testRegions( qMat, Eta )
                concavityMat = NaN(size(qMat));
                for qi = 1:(size(qMat,2)-1)
                    q1 = qMat(:,qi);
                    q2 = qMat(:,qi+1);
                    
                    idx = find(~isnan(q1) & ~isnan(q2));
                    mqVec = NaN(size(idx));
                    % Handle +/- inf
                    q1inf = q1(idx) == -Inf;
                    q2inf = q2(idx) == Inf;
                    other = ~q1inf & ~q2inf;
                    mqVec(q1inf) = q2(idx(q1inf))-1;
                    mqVec(q2inf) = q1(idx(q2inf))+1;
                    mqVec(other) = mean([q1(idx(other)), q2(idx(other))],2);
                    
                    % Test derivative or concavity at these points
                    concavityMat(idx,qi) = sign(ddg(mqVec,Eta(idx,:)));
                end
            end
            
            function R = addRegion( R, ri, Eta, qMat, concavityMat )
                %% Set start and end indices
                % Case 1: Have not added all critical regions yet
                if(size(qMat,2) >= ri+1)
                    selNew = ~isnan(qMat(:,ri+1));
                    R = newRegion( R, selNew, ri, Eta(selNew,:), qMat(selNew,ri), qMat(selNew,ri+1), concavityMat(selNew,ri) );
                else
                    selNew = false(size(qMat,1),1);
                end
                % Case 2: All critical regions added so split a previous region
                R = splitRegion( R, ~selNew, ri, Eta );

                %% Update regions
                % First check if any can be computed directly
                cutoff = 30;
                updateIdxExact = find(ceil(R.maxX)-1-ceil(R.minX)+1 <= cutoff & ~isinf(R.maxX) & ~isinf(R.minX) & ~isnan(R.minX) & isnan(R.logL)); % Min set but logL not set (i.e. needs update)
                if(size(R,1) == 1 && size(R,2) >= 2) % Weird case because R has row vectors but we assume column vectors when R(updateIdx) is given
                    RT = exactUpdate( R', Eta, updateIdxExact, cutoff );
                    R = RT';
                else
                    R = exactUpdate( R, Eta, updateIdxExact, cutoff );
                end
                
                % Update other regions
                updateIdx = find(~isnan(R.minX) & isnan(R.logL)); % Min set but logL not set (i.e. needs update)
                if(size(R,1) == 1 && size(R,2) >= 2) % Weird case because R has row vectors but we assume column vectors when R(updateIdx) is given
                    RT = updateRegions( R', updateIdx );
                    R = RT';
                else
                    R = updateRegions( R, updateIdx );
                end
                
                %fprintf('%d, %d, %g\n', length(updateIdxExact), length(updateIdx), length(updateIdxExact)/length(updateIdx));
                
                function R = exactUpdate( R, Eta, idx, cutoff )
                    if(isempty(idx)); return; end
                    % Create X matrix and find values for each of these x values
                    X = NaN(length(idx),cutoff);
                    fgX = NaN(size(X));
                    for i = 1:cutoff
                        curX = ceil(R.minX(idx)) + i - 1;
                        belowMax = curX < R.maxX(idx);
                        if(sum(belowMax) > 0)
                            X(belowMax, i) = curX(belowMax);
                            [rows, ~] = ind2sub(size(R), idx(belowMax));
                            fgX(belowMax, i) = fg(X(belowMax, i), Eta(rows,:));
                        end
                    end
                    
                    R.logL(idx) = mrfs.utils.logsumexp(fgX,2);
                    R.logU(idx) = R.logL(idx);
                end
                
                function R = newRegion( R, sel, ri,  Eta, minX, maxX, concavity )
                    if(isempty(R)); return; end
                    evalMinX = ceil(minX)+1;
                    evalMaxX = ceil(maxX)-1;
                    % Set variables
                    R.minX(sel,ri) = minX; R.maxX(sel,ri) = maxX;
                    R.evalMinX(sel,ri) = evalMinX; R.evalMaxX(sel,ri) = evalMaxX;
                    R.concavity(sel,ri) = concavity;
                    % Use integral fg
                    R.fgMin(sel,ri) = fg(ceil(minX), Eta); R.fgMax(sel,ri) = fg(ceil(maxX)-1, Eta);
                    % Use internal integral points for gMin and dgMin since only used for Taylor/secant approximations
                    R.gMin(sel,ri) = g(evalMinX, Eta); R.gMax(sel,ri) = g(evalMaxX, Eta);
                    R.dgMin(sel,ri) = dg(evalMinX, Eta); R.dgMax(sel,ri) = dg(evalMaxX, Eta);
                    
                    R.logU(sel,ri) = NaN; R.logL(sel,ri) = NaN;
                end

                function R = splitRegion( R, selSplit, ri, Eta )
                    nSplit = sum(selSplit);
                    if(nSplit==0); return; end
                    idxSplit = find(selSplit);
                    % Find biggest difference between L and U (must convert logL and logU)
                    [~,iSplit] = max(log(1-exp(R.logL(selSplit,:)-R.logU(selSplit,:)))+R.logU(selSplit,:),[],2);
                    %RSplitIdx = sub2ind(size(R),idxSplit,iSplit);
                    rsi = sub2ind(size(R),idxSplit,iSplit);

                    % Find split and derivatives (handle tails specially)
                    %RSplit = R(rsi);
                    negInfBool = R.minX(rsi) == -Inf;
                    posInfBool = R.maxX(rsi) == Inf;
                    interior = ~negInfBool & ~posInfBool;
                    assert(all(negInfBool | posInfBool | interior), 'Not all selected');
                    
                    % Handle inf splits
                    splitX = NaN(size(rsi));
                    splitX(negInfBool) = min( 4*R.maxX(rsi(negInfBool)), -1 ); % Double or make -1 (i.e. ensure negativity, if too far than log(n) more points will find good place by subdividing)
                    splitX(posInfBool) = max( 4*R.minX(rsi(posInfBool)), 1); % Double or make 1 (i.e. ensure positivity, if too far than log(n) more points will find good place by subdividing)
                    
                    % Get weighted sum for interior points
                    logW = [R.fgMin(rsi(interior)), R.fgMax(rsi(interior))];
                    W = exp(bsxfun(@minus, logW, max(logW,[],2)));
                    minWeight = 0.1; % When W = [1,0] we need to add something to make it [1-minWeight, minWeight]
                    W = W + minWeight/(1-2*minWeight); % This will ensure that even if weight is 0, the smoothed weight will be minWeight
                    W = bsxfun(@rdivide, W, sum(W,2));
                    splitX(interior) = sum(W.*[R.minX(rsi(interior))+1, R.maxX(rsi(interior))-1], 2); % Weighted mean if interior region
                    
                    % At least have a gap of 2 since 2 is exact
                    diff = [splitX-R.minX(rsi), R.maxX(rsi)-splitX];
                    nearMin = (diff(:,1) < 2 & diff(:,2) >= 3);
                    nearMax = (diff(:,1) >= 3 & diff(:,2) < 2);
                    splitX(nearMin) = splitX(nearMin)+1;
                    splitX(nearMax) = splitX(nearMax)-1;
                    
                    % Reset U and L, add region by making copy and shifting minX,maxX
                    R.logU(rsi) = NaN;
                    R.logL(rsi) = NaN;

                    % Copy and set new region (concavity and min/max)
                    R.concavity(selSplit,ri) = R.concavity(rsi);
                    R.maxX(selSplit,ri) = R.maxX(rsi);
                    R.fgMax(selSplit,ri) = R.fgMax(rsi);
                    R.evalMaxX(selSplit,ri) = R.evalMaxX(rsi);
                    R.gMax(selSplit,ri) = R.gMax(rsi);
                    R.dgMax(selSplit,ri) = R.dgMax(rsi);
                    
                    % Set min for new region
                    R.minX(selSplit,ri) = splitX;
                    R.fgMin(selSplit,ri) = fg(ceil(splitX), Eta(selSplit,:)); % Use integral calculations for fg
                    evalMinX = ceil(splitX)+1;
                    R.evalMinX(selSplit,ri) = evalMinX;
                    R.gMin(selSplit,ri) = g(evalMinX, Eta(selSplit,:)); % Use internal integral points for gMin and dgMin since only used for Taylor/secant approximations
                    R.dgMin(selSplit,ri) = dg(evalMinX, Eta(selSplit,:));
                    
                    % Shift max of old region
                    R.maxX(rsi) = splitX;
                    R.fgMax(rsi) = fg(ceil(splitX)-1, Eta(selSplit,:)); % Use integral calculations for fg
                    evalMaxX = ceil(splitX)-1;
                    R.evalMaxX(rsi) = evalMaxX;
                    R.gMax(rsi) = g(evalMaxX, Eta(selSplit,:));
                    R.dgMax(rsi) = dg(evalMaxX, Eta(selSplit,:));
                end

                % Vector of regions to update
                function R = updateRegions(R, updateIdx)
                    if(isempty(updateIdx)); return; end
                    
                    % Determine whether tail or interior region
                    infBool = isinf(R.minX(updateIdx)) | isinf(R.maxX(updateIdx));
                    infIdx = updateIdx(infBool);
                    normalIdx = updateIdx(~infBool);

                    % Update each case
                    R = tail(R, infIdx);
                    R = normal(R, normalIdx);

                    function R = tail(R, idx)
                        if(isempty(idx)); return; end
                        % Get q, g(q) and g'(q) for non-infs
                        [~,nonInfI] = min([isinf(R.minX(idx)), isinf(R.maxX(idx)) ],[],2);
                        minSel = nonInfI == 1;
                        maxSel = nonInfI == 2;
                        q = zeros(size(idx)); gq = q; dgq = q; % Setup with zeros
                        q(minSel) = R.evalMinX(idx(minSel));
                        gq(minSel) = R.gMin(idx(minSel));
                        dgq(minSel) = R.dgMin(idx(minSel));
                        q(maxSel) = R.evalMaxX(idx(maxSel));
                        gq(maxSel) = R.gMax(idx(maxSel));
                        dgq(maxSel) = R.dgMax(idx(maxSel));

                        % Assume convex (will be decreasing since tail)
                        R.bu(idx) = zeros(size(gq));
                        R.cu(idx) = gq;
                        R.bl(idx) = dgq;
                        R.cl(idx) = gq - q.*dgq;

                        % Switches for concave and computes upper and lower bounds
                        R = logRegionM(R, idx);
                    end

                    function R = normal(R, idx)
                        if(isempty(idx)); return; end
                        % Get convexity and Taylor (which requires argmax_j g(q_j))
                        [~,iTaylor] = max([R.fgMin(idx),R.fgMax(idx)],[],2); % Use fg combo rather than just g (i.e. focus on areas of the function that are larger)
                        subSelMin = iTaylor == 1;
                        idxMin = idx(subSelMin);
                        idxMax = idx(~subSelMin);
                        gTaylor = zeros(size(iTaylor));
                        gTaylor(subSelMin) = R.gMin(idxMin);
                        gTaylor(~subSelMin) = R.gMax(idxMax);
                        qTaylor = zeros(size(iTaylor));
                        qTaylor(subSelMin) = R.evalMinX(idxMin);
                        qTaylor(~subSelMin) = R.evalMaxX(idxMax);
                        dgTaylor = zeros(size(iTaylor));
                        dgTaylor(subSelMin) = R.dgMin(idxMin);
                        dgTaylor(~subSelMin) = R.dgMax(idxMax);

                        % Assume convex Upper = secant, Lower = Taylor at max q
                        idxZero = idx(R.evalMaxX(idx) == R.evalMinX(idx)); % If evalMaxX=evalMinX, then same point and thus slope is 0
                        R.bu(idx) = (R.gMax(idx) - R.gMin(idx))./(R.evalMaxX(idx)-R.evalMinX(idx));
                        R.bu(idxZero) = 0;
                        R.cu(idx) = R.gMin(idx) - R.bu(idx).*R.evalMinX(idx);
                        
                        R.bl(idx) = dgTaylor;
                        R.bl(idxZero) = 0;
                        R.cl(idx) = gTaylor - qTaylor.*R.bl(idx);
                        
                        % Switches for concave and computes upper and lower bounds
                        R = logRegionM( R, idx );
                    end

                    function R = logRegionM( R, idx )
                        %NOTE: eta1 is embedded in R
                        convex = R.concavity(idx) >= 0;

                        % Switch if concave
                        tempB = R.bu(idx(~convex));
                        tempC = R.cu(idx(~convex));
                        R.bu(idx(~convex)) = R.bl(idx(~convex));
                        R.cu(idx(~convex)) = R.cl(idx(~convex));
                        R.bl(idx(~convex)) = tempB;
                        R.cl(idx(~convex)) = tempC;

                        % Get upper and lower bounds
                        R.logU(idx) = linApprox(R, idx, true);
                        R.logL(idx) = linApprox(R, idx, false);
                        
                        %find(~( relDiff > -1e-10 | (relDiff > -1e-6 & R.logU(idx) < -100 ) | (relDiff > -0.1 & R.logU(idx) > 15 ) )))
                        %[tempB, tempC] = ind2sub(size(R),idx(find(~(  relDiff > -1e-10 | (relDiff > -0.1 & R.logU(idx) < -100 ) | (relDiff > -0.1 & R.logU(idx) > 15 )   ))))
                        relDiff = (R.logU(idx)-R.logL(idx))./abs(R.logU(idx)+eps);
                        % 3 cases 
                        % (1) the relDiff is small because of approximation error
                        % (2) the logU is so close to 0 that the relative difference is too small to measure (again approximation error)
                        % (3) the logU is very large (i.e. exp(logU) is the approximate mean of the Poisson, thus exp(15) \approx 3,000,000 )
                        if(~all( relDiff > -1e-10 | (R.logU(idx) < -100 ) | (R.logU(idx) > 15 ) ))
                            warning('logU is smaller than logL outside of thresholds');
                        end
                        roundOff = R.logU(idx) < R.logL(idx); 
                        R.logU(idx(roundOff)) = R.logL(idx(roundOff)); % Ensure logU >= logL even with roundoff error (error check above)

                        function logY = linApprox( R, idx, isUpper )
                            % Get b and c from isUpper
                            if(isUpper)
                                b = R.bu(idx); c = R.cu(idx);
                            else
                                b = R.bl(idx); c = R.cl(idx);
                            end
                            % Form etaHat
                            etaHat = R.eta1(idx) + b;
                            lamHat = exp(etaHat);
                            AetaHat = lamHat; % For Poisson this is the same

                            % Use CDF for approximation (Note that poisscdf is <= x)
                            intMaxX = ceil(R.maxX(idx)) - 1; % CDF with maxX excluded if integer (closest integer point in interval while being exclusive at end point since open interval)
                            intMinX = ceil(R.minX(idx)); % CDF with ceil(minX) included (and thus removed by subtraction)
                            selLower = intMaxX <= lamHat & intMinX <= lamHat;
                            cdfDiff = NaN(size(lamHat));
                            cdfDiff(selLower) = poisscdf(intMaxX(selLower),lamHat(selLower)) - poisscdf(intMinX(selLower),lamHat(selLower));
                            cdfDiff(~selLower) = poisscdf(intMinX(~selLower),lamHat(~selLower),'upper') - poisscdf(intMaxX(~selLower),lamHat(~selLower),'upper');
                            
                            logY = mrfs.utils.logsumexp([ R.fgMin(idx), c + AetaHat + log(max(cdfDiff,0)) ], 2); % Clip to 0 because sometimes maxX and minX are too close
                            logY(ceil(R.maxX(idx))-1 < ceil(R.minX(idx))) = -Inf; % Special case where region does not contain any integers
                            %Debug code
                            %{R.fgMin(idx(905)),c(905),AetaHat(905),log(max(cdfDiff(905),0))}
                            %{R.fgMin(idx),c,AetaHat,log(max(cdfDiff,0))}
                        end
                    end
                end
            end
        end
        
        %% Testing code for Poisson
        function testPoisson()
            % Setup
            check=true;
            n = 1000;
            k = 4;
            nQuery = max(10, k+1);
            a = 1;
            
            rng(20);
            Eta = bsxfun(@times, 2*randn(n,k), 1:k);
            
            [MuBatch, MlBatch] = mrfs.grm.univariate.Poisson.approxM(a, Eta, nQuery);
            
            if(check)
                %MBatch = mrfs.grm.univariate.Poisson.exactM(a, Eta);
                M = NaN(size(Eta,1),1);
                Mu = NaN(size(M));
                Ml = NaN(size(M));
                for i = 1:n
                    M(i) = mrfs.grm.univariate.Poisson.exactM(a, Eta(i,:));
                    % Skip absurdly large etas
                    if(M(i) > 100)
                        fprintf('   Skipping i = %d because M(i) = %g, roughly a mean of exp(M(i)) = %g\n',i, M(i), exp(M(i)));
                        continue;
                    end
                    [Mu(i), Ml(i)] = mrfs.grm.univariate.Poisson.approxM(a, Eta(i,:), nQuery);

                    % Display debug info
                    %fprintf('Eta(%d,:)\n', i);
                    %disp(Eta(i,:));
                    %fprintf('Values\n'); disp([Mii, Muii, Mlii]); fprintf('Diff (should be > -1e-12 )\n');disp([Mii-Mii, Muii-Mii, Mii-Mlii]);

                    % Check batch to single are the same
                    %assert(M(i) == MBatch(i),'M different than batchM');
                    assert(Mu(i) == MuBatch(i),'Mu different than batchMu');
                    assert(Ml(i) == MlBatch(i),'Ml different than batchMl');

                    % Check that properly upper and lower bounds
                    assert((Mu(i) - M(i))./abs(M(i)) > -1e-6,'Upper bound error Muii');
                    assert((M(i) - Ml(i))./abs(M(i)) > -1e-6,'Lower bound error Mlii');
                    %assert((MuBatch(i) - MBatch(i))./abs(MBatch(i)) > -1e-10,'Upper bound error batchMu(ii)');
                    %assert((MBatch(i) - MlBatch(i))./abs(MBatch(i)) > -1e-10,'Lower bound error batchMl(ii)');
                    fprintf('%d passed!\n', i);
                end
                fprintf('<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>\n');
                fprintf('<<< All assertions passed :-D !!!!! >>>\n');
                fprintf('<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>\n');
            end
        end
        
        %% Visualize approximation
        function [grm] = visApprox()
            %% Construct dataset
            rng(2);
            prob = [0,0.1,2,3.5,6,3.5,2,0.5,0.1,0.1];
            x = (0:(length(prob)-1))*5;
            bar(x,prob);
            
            prob = prob/sum(prob);
            
            nSamples = 100;
            mnSample = mnrnd(nSamples, prob);
            Xt = NaN(nSamples,1);
            curI = 0;
            for i = 1:length(prob)
                Xt((1:mnSample(i)) + curI) = x(i);
                curI = curI + mnSample(i);
            end
            %%
            rng(1);
            Xt = nbinrnd(6,0.4,100,1);
            %Xt = max(round(randn(50)*5+4),0);
            histogram(Xt,0:40);
            
            % Find fit
            k = 2;
            grm = mrfs.grm.fitgrm(Xt, k, struct('lam',0,'nWorkers',1));
            
            %% Visualize fit
            a = 0;
            if(k == 3)
                eta = [grm.Psi{1,1}.full(), grm.Psi{1,2}.full(), grm.Psi{1,3}.full()]
            elseif(k == 2)
                eta = [grm.Psi{1,1}.full(), grm.Psi{1,2}.full()]
            end
            nQuery = 10;
            [Mu, Ml] = mrfs.grm.univariate.Poisson.approxM( a, eta, nQuery, true);
        end
        
        %% Naive summation to test
        function [M, fCell, xCell] = exactM(a, Eta)
            [n,k] = size(Eta);
            % Super naive and dumb but should work reasonably well
            EtaT = Eta';
            xCell = cell(n,1);
            fCell = cell(n,1);
            M = NaN(n,1);
            for i = 1:n
                eta = EtaT(:,i);
                
                upperEta = eta(1)+sum(max(abs(eta(2:end)),0));
                maxX = max(ceil(4*exp(upperEta)),10); % At least 10
                clip = 1e4;
                if(maxX >= clip)
                    fprintf('maxX =%d, being clipped at = %d\n',maxX, clip);
                    maxX = min(clip, maxX);
                end
                
                x = (0:maxX)'; % Very large value that should definitely be past the normal tail
                xCell{i} = x;
                X = bsxfun(@power, x, (1./(1:k)));
                fCell{i} = f(X,eta,a);
                M(i) = mrfs.utils.logsumexp(fCell{i});
            end
            
            function v = f(X,eta,a)
                if(a == 0)
                    v = X*eta - gammaln(X(:,1)+1) + log(X(:,1).^a);
                else
                    v = X*eta - gammaln(X(:,1)+1) + log(X(:,1).^a);
                end
            end
        end
    end
    
end

function visScatter( XtSample )
    XtSample = XtSample + 0.1*randn(size(XtSample));
    C = sqrt(sum(bsxfun(@minus, XtSample, mean(XtSample)).^2,2));
    %temp = mean(pdist2(XtSample,XtSample));
    %[~, C] = sort(temp,'descend');
    scatter3(XtSample(:,1), XtSample(:,2), XtSample(:,3), 30, C, 'o','filled');
end
