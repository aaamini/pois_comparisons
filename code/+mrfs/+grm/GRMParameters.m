classdef GRMParameters < handle
    %GRMPARAMETERS Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (SetAccess = protected)
        Psi;
        optSubs = [];
        lock = false;
        PsiCache = {};
    end

    methods (Static)
        function T = outerProduct( x, power )
            x = x(:); % Ensure column vector
            p = length(x);
            if(power < 0)
                error('Outer product not defined for power < 0');
            elseif(power == 0)
                T = ones(size(x));
            elseif(power == 1)
                T = x;
            elseif(power == 2)
                T = x*x';
            else
                T = zeros(p*ones(1,power)); % p x p x p x ... tensor
                Pcell = num2cell( allPerms( p, power-2 ) ); % Needed for each matrix
                nPerms = size(Pcell,1);
                X = x*x';
                % Fill nd array in a matrix wise fashion
                for pi = 1:nPerms
                    T(:,:,Pcell{pi,:}) = prod(x([Pcell{pi,:}])) * X;
                end
            end
            
            function P = allPerms( p, k )
                if(k < 1)
                    error('Not defined for k < 1');
                elseif(k == 1)
                    P = (1:p)';
                else
                    Xcell = cell(1, k);
                    [Xcell{:}] = ndgrid(1:p);
                    P = cat(k+1, Xcell{:});
                    P = reshape(P, [p^k, k]);
                end
            end
        end
        
        % Convert Psi full matrix to averaged and symmetric version
        function Tsym = psi2avgsym(PsiFull)
            % Handle special case when Psi is a vector (i.e. ell = 1)
            if(isvector(PsiFull))
                Tsym = PsiFull;
                return;
            end
            
            % Otherwise find ell
            ell = length(size(PsiFull)); % Get the number of dimensions
            
            % Get permutations
            P = perms(1:ell);
            
            % Symmetrize
            Tsym = zeros(size(PsiFull));
            for pi = 1:size(P,1)
                Tsym = Tsym + permute(PsiFull,P(pi,:));
            end
            Tsym = Tsym./ell; % e.g. for ell=3, Tsym_perm(123) = (Psi_123 + Psi_213 + Psi_312 + 0 + 0 + 0)/3 --- i.e. the average of the 3 estimates
        end
        
        function relLogL = relLogLikelihood( x, PsiExt )
            relLogL = -sum(gammaln(x+1)); % Base measure
            k = size(PsiExt,1);
            for j = 1:k
                for ell = 1:j
                    X = mrfs.grm.GRMParameters.outerProduct( x.^(1/j), ell);
                    innerProd = X.*PsiExt{ell,j};
                    relLogL = relLogL + sum(innerProd(:));
                end
            end
        end
    end
    
    methods
        
        function obj = GRMParameters( k, p, initType )
            if(nargin < 3); initType = 1; end
            obj.Psi = cell(k,k);
            for j = 1:k
                for ell = 1:j
                    obj.Psi{ell,j} = mrfs.grm.sparsetensor( p*ones(1,ell) );
                end
            end
            obj.initOptSubs(initType);
        end
        
        %% Get external representation of Psi
        function PsiExt = getPsiExt(o)
            % Create cached version of full Psi if it does not exist
            k = o.getK();
            PsiExt = cell(k,k);
            for j2 = 1:k
                for ell2 = 1:j2
                    PsiExt{ell2,j2} = o.psi2avgsym(o.Psi{ell2,j2}.full());
                end
            end
        end
        
        function setLock( o )
            o.lock = true;
        end
        
        function releaseLock( o )
            o.lock = false;
        end
        
        function showParams( o, labels, nTop, Xt, docs, pairedLabels )
            if(nargin < 3); nTop = 20; end
            if(nargin < 4); Xt = []; end
            if(nargin < 5); docs = {}; end
            if(nargin < 6); pairedLabels = {'Unknown_Label'}; end
            
            p = o.getP();
            for j = 1:o.getK()
                for ell = 1:j
                    [~, Vuniq, subsUniq] = o.Psi{ell,j}.getAllUniq();
                    % Need to adjust value since merely sum of values i.e. V = (Psi_123 + Psi_213 + Psi_312 + 0 + 0 + 0) 
                    %  but we want (V / ell) * perm(ell) = psi_avg(123) * perm(ell)
                    Vuniq = (factorial(ell)/ell) * Vuniq;
                    if(~isempty(Vuniq))
                        [Vuniq, sortedI] = sort(Vuniq,1,'descend');
                        subsUniq = subsUniq(sortedI,:);
                        if(length(pairedLabels) == 2)
                            if( ell == 1 ) % Special case for top words since parameter can be negative
                                % Get nTop/2 of first vocab set
                                for pi = 1:2
                                    ti = 1;
                                    topTuples = cell(1,nTop);
                                    topTuples(:) = {''};
                                    for ii = 1:length(Vuniq)
                                        if( (pi == 1 && subsUniq(ii) <= p/2) || (pi == 2 && subsUniq(ii) >= p/2 + 1) ) % Check to make sure it's in first set
                                            valStr = sprintf('%.4f', Vuniq(ii));
                                            tupleStr = sprintf('%7s  %s', valStr, labels{subsUniq(ii)});
                                            
                                            %[suffStat,docI] = sort(Xt(:,subsUniq(ii)),'descend');
                                            %tupleStr = sprintf('%-25s  NNZ docs: %d, Example: %s\n', tupleStr, nnz(suffStat), docs{docI(1)});
                                            
                                            topTuples{ti} = tupleStr; 
                                            ti = ti + 1;
                                        end
                                        if(ti > nTop); break; end
                                    end
                                    fprintf('<<< %s Largest %d-tuples, j = %d, ell = %d >>>\n', pairedLabels{pi}, ell, j, ell);
                                    disp(cell2mat(topTuples));
                                    fprintf('\n');
                                end
                            elseif( ell == 2) % Interaction terms
                                tuples = cell(3,nTop); % 1 = 1x1, 2 = 2x2, 3 = 1x2
                                tuples(:) = {''}; % Fill with emptys
                                i1 = 1; i2 = 1; i12 = 1;
                                for ii = 1:length(Vuniq)
                                    if(Vuniq(ii) <= 0); break; end % Only consider positive
                                    tupleLabels = labels(subsUniq(ii,:));
                                    valStr = sprintf('%.4f', Vuniq(ii));

                                    tupleStr = sprintf(['%7s  %s', repmat(' + %s',1,ell-1)], valStr, tupleLabels{:});
                                    
                                    %[suffStat,docI] = sort(prod(Xt(:,subsUniq(ii,:)), 2),'descend');
                                    %tupleStr = sprintf('%-35s  NNZ docs: %d, Example: %s\n', tupleStr, nnz(suffStat), docs{docI(1)});
                                    
                                    if( all(subsUniq(ii,:) <= p/2) && i1 <= nTop ) % 1
                                        tuples{1,i1} = tupleStr;
                                        i1 = i1 + 1;
                                    elseif( all(subsUniq(ii,:) > p/2) && i2 <= nTop ) % 1
                                        tuples{2,i2} = tupleStr;
                                        i2 = i2 + 1;
                                    elseif( ( (subsUniq(ii,1) <= p/2 && subsUniq(ii,2) >  p/2) ...
                                            || (subsUniq(ii,1) >  p/2 && subsUniq(ii,2) <= p/2) )...
                                            && i12 <= nTop ) % 1
                                        tuples{3,i12} = tupleStr;
                                        i12 = i12 + 1;
                                    else
                                        %fprintf('Ignoring tuple: %s, i = (%d,%d,%d)\n', tupleStr, i1,i2,i12);
                                    end
                                end
                                fprintf('<<< %s Internal %d-tuples (parameter > 0), j = %d, ell = %d >>>\n', pairedLabels{1}, ell, j, ell);
                                disp(cell2mat(tuples(1,:)));
                                fprintf('\n');
                                fprintf('<<< %s Internal %d-tuples (parameter > 0), j = %d, ell = %d >>>\n', pairedLabels{2}, ell, j, ell);
                                disp(cell2mat(tuples(2,:)));
                                fprintf('\n');
                                fprintf('<<< %s Cross %s %d-tuples (parameter > 0), j = %d, ell = %d >>>\n', pairedLabels{1}, pairedLabels{2}, ell, j, ell);
                                disp(cell2mat(tuples(3,:)));
                                fprintf('\n');
                                
                                %%%%%%%%%%%%%% SUPER NAIVE COPY OF ABOVE %%%%%%%%%%%%%%
                                tuples = cell(3,nTop); % 1 = 1x1, 2 = 2x2, 3 = 1x2
                                tuples(:) = {''}; % Fill with emptys
                                i1 = 1; i2 = 1; i12 = 1;
                                for ii = length(Vuniq):-1:1
                                    if(Vuniq(ii) >= 0); break; end % Only consider negative
                                    tupleLabels = labels(subsUniq(ii,:));
                                    valStr = sprintf('%.4f', Vuniq(ii));
                                    tupleStr = sprintf(['%7s  %s', repmat(' - %s',1,ell-1), '\n'], valStr, tupleLabels{:});
                                    
                                    if( all(subsUniq(ii,:) <= p/2) && i1 <= nTop ) % 1
                                        tuples{1,i1} = tupleStr;
                                        i1 = i1 + 1;
                                    elseif( all(subsUniq(ii,:) > p/2) && i2 <= nTop ) % 1
                                        tuples{2,i2} = tupleStr;
                                        i2 = i2 + 1;
                                    elseif( ( (subsUniq(ii,1) <= p/2 && subsUniq(ii,2) >  p/2) ...
                                            || (subsUniq(ii,1) >  p/2 && subsUniq(ii,2) <= p/2) )...
                                            && i12 <= nTop ) % 1
                                        tuples{3,i12} = tupleStr;
                                        i12 = i12 + 1;
                                    else
                                        %fprintf('Ignoring tuple: %s, i = (%d,%d,%d)\n', tupleStr, i1,i2,i12);
                                    end
                                end
                                fprintf('<<< %s Internal %d-tuples (parameter < 0), j = %d, ell = %d >>>\n', pairedLabels{1}, ell, j, ell);
                                disp(cell2mat(tuples(1,:)));
                                fprintf('\n');
                                fprintf('<<< %s Internal %d-tuples (parameter < 0), j = %d, ell = %d >>>\n', pairedLabels{2}, ell, j, ell);
                                disp(cell2mat(tuples(2,:)));
                                fprintf('\n');
                                fprintf('<<< %s Cross %s %d-tuples (parameter < 0), j = %d, ell = %d >>>\n', pairedLabels{1}, pairedLabels{2}, ell, j, ell);
                                disp(cell2mat(tuples(3,:)));
                                fprintf('\n');
                                %%%%%%%%%%%%%% SUPER NAIVE COPY OF ABOVE %%%%%%%%%%%%%%
                            elseif( ell == 4) % Special interaction terms
                                tuples = cell(2,nTop); % 1 = same words, 2 = different words
                                tuples(:) = {''}; % Fill with emptys
                                i1 = 1; i2 = 1;
                                for ii = 1:length(Vuniq)
                                    if(Vuniq(ii) <= 0); break; end % Only consider positive
                                    % Check to make sure words are paired correctly
                                    assert(subsUniq(ii,1) + p/2 == subsUniq(ii,3), 'Words are not paired');
                                    assert(subsUniq(ii,2) + p/2 == subsUniq(ii,4), 'Words are not paired');
                                    
                                    tupleLabels = labels(subsUniq(ii,:)); % Only need first and second labels
                                    valStr = sprintf('%.4f', Vuniq(ii));
                                    tupleStr = sprintf('%7s  (%s+%s) + (%s+%s)', valStr, tupleLabels{:});
                                    
                                    %[suffStat,docI] = sort(prod(Xt(:,subsUniq(ii,:)), 2),'descend');
                                    %tupleStr = sprintf('%-70s  NNZ docs: %d, Example: %s\n',tupleStr, nnz(suffStat), docs{docI(1)});
                                    
                                    if( subsUniq(ii,1) == subsUniq(ii,2) && i1 <= nTop ) % Same words
                                        error('Should not be the same word');
                                        tuples{1,i1} = tupleStr;
                                        i1 = i1 + 1;
                                    elseif( i2 <= nTop ) % Different words
                                        tuples{2,i2} = tupleStr;
                                        i2 = i2 + 1;
                                    else
                                        %fprintf('Ignoring tuple: %s, i = (%d,%d,%d)\n', tupleStr, i1,i2,i12);
                                    end
                                end
                                fprintf('<<< Edges shared between %s and %s %d-tuples (parameter > 0), j = %d, ell = %d >>>\n', pairedLabels{1}, pairedLabels{2}, ell, j, ell);
                                disp(cell2mat(tuples(2,:)));
                                fprintf('\n');
                                
                                %%%%%%%%%%%%%% SUPER NAIVE COPY OF ABOVE %%%%%%%%%%%%%%
                                tuples = cell(2,nTop); % 1 = same words, 2 = different words
                                tuples(:) = {''}; % Fill with emptys
                                i1 = 1; i2 = 1;
                                for ii = length(Vuniq):-1:1
                                    if(Vuniq(ii) > 0); break; end % Only consider negative
                                    % Check to make sure words are paired correctly
                                    assert(subsUniq(ii,1) + p/2 == subsUniq(ii,3), 'Words are not paired');
                                    assert(subsUniq(ii,2) + p/2 == subsUniq(ii,4), 'Words are not paired');
                                    
                                    tupleLabels = labels(subsUniq(ii,:)); % Only need first and second labels
                                    valStr = sprintf('%.4f', Vuniq(ii));
                                    tupleStr = sprintf('%7s  (%s-%s) + (%s-%s) \n', valStr, tupleLabels{:});
                                    
                                    if( subsUniq(ii,1) == subsUniq(ii,2) && i1 <= nTop ) % Same words
                                        error('Should not be the same word');
                                        tuples{1,i1} = tupleStr;
                                        i1 = i1 + 1;
                                    elseif( i2 <= nTop ) % Different words
                                        tuples{2,i2} = tupleStr;
                                        i2 = i2 + 1;
                                    else
                                        %fprintf('Ignoring tuple: %s, i = (%d,%d,%d)\n', tupleStr, i1,i2,i12);
                                    end
                                end
                                fprintf('<<< Edges shared between %s and %s %d-tuples (parameter < 0), j = %d, ell = %d >>>\n', pairedLabels{1}, pairedLabels{2}, ell, j, ell);
                                disp(cell2mat(tuples(2,:)));
                                fprintf('\n');
                                %%%%%%%%%%%%%% SUPER NAIVE COPY OF ABOVE %%%%%%%%%%%%%%
                                
                            else
                                posTupleStr = cell(1,nTop);
                                negTupleStr = cell(1,nTop);
                                for ii = 1:nTop
                                    if(ii <= length(Vuniq) && Vuniq(ii) > 0 )
                                        tupleLabels = labels(subsUniq(ii,:));
                                        valStr = sprintf('%.4f', Vuniq(ii));
                                        posTupleStr{ii} = sprintf(['%7s  %s', repmat(' + %s',1,ell-1), '\n'], valStr, tupleLabels{:});

                                        %[suffStat,docI] = sort(prod(Xt(:,subsUniq(ii,:)),2),'descend');
                                        %posTupleStr{ii} = sprintf('%-35s  NNZ docs: %d, Example: %s\n', posTupleStr{ii}, nnz(suffStat), docs{docI(1)});
                                    else
                                        posTupleStr{ii} = '';
                                    end
                                    revI = length(Vuniq)-(ii-1);
                                    if(revI <= length(Vuniq) && revI >= 1 && Vuniq(revI) < 0 )
                                        tupleLabels = labels(subsUniq(revI,:));
                                        valStr = sprintf('%.4f', Vuniq(revI));
                                        negTupleStr{ii} = sprintf(['%7s  %s', repmat(' - %s',1,ell-1), '\n'], valStr, tupleLabels{:});
                                    else
                                        negTupleStr{ii} = '';
                                    end
                                end
                                fprintf('<<< %d-tuples (parameter > 0), j = %d, ell = %d >>>\n', ell, j, ell);
                                disp(cell2mat(posTupleStr));
                                fprintf('\n');
                                fprintf('<<< %d-tuples (parameter < 0), j = %d, ell = %d >>>\n', ell, j, ell);
                                disp(cell2mat(negTupleStr));
                                fprintf('\n');
                            end
                            
                        else
                            if( ell == 1 ) % Special case for top words since parameter can be negative
                                topTuples = cell(1,nTop);
                                for ii = 1:nTop
                                    if(ii <= length(Vuniq))
                                        valStr = sprintf('%.4f', Vuniq(ii));
                                        topTuples{ii} = sprintf('%7s  %s\n', valStr, labels{subsUniq(ii)});

                                        %[suffStat,docI] = sort(Xt(:,subsUniq(ii)),'descend');
                                        %topTuples{ii} = sprintf('%-25s  NNZ docs: %d, Example: %s\n', topTuples{ii}, nnz(suffStat), docs{docI(1)});
                                    else
                                        topTuples{ii} = '';
                                    end
                                end
                                fprintf('<<< %s Largest %d-tuples, j = %d, ell = %d >>>\n', pairedLabels{1}, ell, j, ell);
                                disp(cell2mat(topTuples));
                                fprintf('\n');
                            else % Interaction terms
                                posTupleStr = cell(1,nTop);
                                negTupleStr = cell(1,nTop);
                                for ii = 1:nTop
                                    if(ii <= length(Vuniq) && Vuniq(ii) > 0 )
                                        tupleLabels = labels(subsUniq(ii,:));
                                        valStr = sprintf('%.4f', Vuniq(ii));
                                        posTupleStr{ii} = sprintf(['%7s  %s', repmat(' + %s',1,ell-1), '\n'], valStr, tupleLabels{:});

                                        %[suffStat,docI] = sort(prod(Xt(:,subsUniq(ii,:)),2),'descend');
                                        %posTupleStr{ii} = sprintf('%-35s  NNZ docs: %d, Example: %s\n', posTupleStr{ii}, nnz(suffStat), docs{docI(1)});
                                    else
                                        posTupleStr{ii} = '';
                                    end
                                    revI = length(Vuniq)-(ii-1);
                                    if(revI <= length(Vuniq) && revI >= 1 && Vuniq(revI) < 0 )
                                        tupleLabels = labels(subsUniq(revI,:));
                                        valStr = sprintf('%.4f', Vuniq(revI));
                                        negTupleStr{ii} = sprintf(['%7s  %s', repmat(' - %s',1,ell-1), '\n'], valStr, tupleLabels{:});
                                    else
                                        negTupleStr{ii} = '';
                                    end
                                end
                                fprintf('<<< %s Internal %d-tuples (parameter > 0), j = %d, ell = %d >>>\n', pairedLabels{1}, ell, j, ell);
                                disp(cell2mat(posTupleStr));
                                fprintf('\n');
                                fprintf('<<< %s Internal %d-tuples, j = %d, ell = %d >>>\n', pairedLabels{1}, ell, j, ell);
                                disp(cell2mat(negTupleStr));
                                fprintf('\n');
                            end
                        end
                    end
                end
            end
        end
        
        % Display all parameters
        function disp(o)
            fprintf('<<<< OptSubs >>>>\n');
            disp(o.optSubs);
            
            k = o.getK();
            for j = 1:k
                fprintf('<<<< j/k = %d/%d >>>>\n', j, k);
                for ell = 1:j
                    fprintf('  << ell = %d >>\n', ell);
                    if(nnz(full(o.Psi{ell,j})) > 0)
                        disp(full(o.Psi{ell,j}));
                    else
                        fprintf('   (all zeros)\n');
                    end
                end
            end
        end
        
        % Initialize optimization subscripts (i.e. possible non-zeros)
        function initOptSubs( o, initType )
            if(o.lock); error('Cannot initialize optSubs when lock is set.'); end
            if(nargin < 2); initType = 1; end
            if(initType == 0); return; end
            if(initType == 10); o.initPaired('largest-terms'); return; end
            if(initType == 11); o.initPaired('fourth-root-terms'); return; end
            % initType == 0: No parameters are free
            % initType == 1: All possible values under GRM model (default)
            % initType == 2: Special case of only constant or largest interaction terms
            p = o.getP(); k = o.getK();
            subCell = cell(k*(k+1)/2,1);
            ii = 1;
            for j = 1:k
                if(initType == 3 && (j > 1 && j < k)); continue; end % Skip in the middle
                for ell = 1:j
                    if((initType == 2 || initType == 3) && ell ~= j); continue; end % Skip except when ell == j
                    if( ell <= p )
                        C = nchoosek(1:p, ell);
                        subCell{ii} = [...
                            repmat([ell,j],size(C,1),1), ...
                            C, ...
                            -ones(size(C,1),k-ell) ];
                    else
                        subCell{ii} = [];
                    end
                    ii = ii + 1;
                end
            end
            o.optSubs = unique(cell2mat(subCell),'rows');
        end
        
        function initPaired( o, type )
            if(o.lock); error('Cannot initialize optSubs when lock is set.'); end
            %% Setup: We assume that p is even so that 1:(p/2) is x and (p/2+1):p is y
            if(nargin < 2); type = 'largest-terms'; end
            p = o.getP(); k = o.getK();
            assert( mod(p,2) == 0, 'p must be even for initPaired');
            assert( k == 4, 'k should be 4 for initPaired');
            xI = 1:(p/2);
            yI = (p/2+1):p;
            subCell = cell(3,1);
            
            %% Get indices for initType
            switch type
                case 'largest-terms'
                    jInd = 1;
                    jPairwise = 2;
                case 'fourth-root-terms'
                    jInd = 4;
                    jPairwise = 4;
                otherwise
                    error('Not a correct paired type');
            end
            
            %% Independent terms
            subCell{1} = [repmat([1,jInd],p,1), (1:p)', -ones(p,k-1)];
            
            %% Pairwise terms
            ell = 2;
            C = nchoosek(1:p, ell);
            subCell{2} = [...
                repmat([ell,jPairwise],size(C,1),1), ...
                C, ...
                -ones(size(C,1),k-ell) ];
            
            %% Special 4-wise "paired" terms
            ell = 4;
            Ct = NaN(4,nchoosek(p/2,2));
            ii = 1;
            for s = 1:(p/2)
                for t = 1:(s-1)
                    Ct(:,ii) = [ [s,t], [s,t] + (p/2) ]; % Paired terms
                    ii = ii + 1;
                end
            end
            C = Ct';
            
            subCell{3} = [...
                repmat([ell,4],size(C,1),1), ...
                C, ...
                -ones(size(C,1),k-ell) ];
            
            %% Final init step
            o.optSubs = unique(cell2mat(subCell),'rows');
        end
        
        function addOptSubs( o, newOptSubs)
            if(o.lock); error('Cannot modify optSubs when lock is set.'); end
            k = o.getK();
            assert(newOptSubs(1) == size(newOptSubs,2) - 2, 'Tensor is of size ell but number of indices is not ell');
            newOptSubs = [newOptSubs(:,1:2), sort(newOptSubs(:,3:end),2), -ones(size(newOptSubs,1), k+2-size(newOptSubs,2) )];
            o.optSubs = unique([o.optSubs; newOptSubs],'rows');
        end
        
        function removeOptSubs( o, removeOptSubs)
            error('For now removing is not allowed because of the assumptions for betaSet and ZtSet construction');
            if(o.lock); error('Cannot modify optSubs when lock is set.'); end
            k = o.getK();
            assert(removeOptSubs(1) == size(removeOptSubs,2) - 2, 'Tensor is of size ell but number of indices is not ell');
            removeOptSubs = [removeOptSubs(:,1:2), sort(removeOptSubs(:,3:end),2), -ones(size(removeOptSubs,1), k+2-size(removeOptSubs,2) )];

            % Duplicates in ic need to be removed
            if(size(removeOptSubs,1) == 1)
                LIA = ismember(o.optSubs, repmat(removeOptSubs,2,1), 'rows');
            else
                LIA = ismember(o.optSubs, removeOptSubs, 'rows');
            end
            o.optSubs(LIA, :) = []; % Remove rows
        end
        
        % Get sparse beta as defined in paper
        function betaSet = getBetaSet(o, s)
            if(~o.lock); error('Cannot get beta set without lock set.'); end
            k = o.getK();
            betaSet = cell(k,1);
            for j = 1:k
                Vcell = cell(j,1);
                for ell = 1:j
                    % Get optSubs
                    subTensor = o.Psi{ell,j}.getSubTensor( s );
                    VsCell = cell(ell,1);
                    for si = 1:ell
                        rowSel = all( bsxfun(@eq, o.optSubs(:,[1,2,si+2]), [ell,j,s]), 2);
                        colSel = true(ell+2,1);
                        colSel(1:2) = false;
                        colSel(si+2) = false;
                        subsS = o.optSubs(rowSel,colSel); % Subtensor subs
                        if(sum(rowSel) >= 1 && sum(colSel) < 1)
                            VsCell{si} = full(subTensor.get(1)); % Scalar case
                        else
                            VsCell{si} = full(subTensor.get(subsS));
                        end
                    end
                    Vcell{ell} = cell2mat(VsCell);
                end
                betaSet{j} = cell2mat(Vcell);
                %{
                Icell = cell(j,1);
                Vcell = cell(j,1);
                totalSize = 0;
                for ell = 1:j
                    % Vectorize each tensor
                    subTensor = o.Psi{ell,j}.getSubTensor( s ); 
                    [I,V,subs] = subTensor.getAll();
                    Icell{ell} = I + totalSize; % Shift index values
                    Vcell{ell} = ell*V; % Implicit de-averaging (see setBetaSet)
                    totalSize = totalSize + prod( subTensor.size() );
                end
                Iall = cell2mat(Icell);
                Vall = cell2mat(Vcell);
                betaSet{j} = sparse(Iall, ones(size(Iall)), Vall, totalSize, 1);
                %}
            end
        end
        
        % Set Psi based on sparse beta defined in paper
        function setBetaSet(o, s, betaSet)
            if(~o.lock); error('Cannot set beta set without lock set.'); end
            for j = 1:o.getK()
                curI = 1;
                for ell = 1:j
                    % Create subtensor
                    subSz = o.Psi{ell,j}.getSubTensorSize( s );
                    subTensor = mrfs.grm.sparsetensor( subSz );
                    
                    % Update subtensor
                    for si = 1:ell
                        rowSel = all( bsxfun(@eq, o.optSubs(:,[1,2,si+2]), [ell,j,s]), 2);
                        colSel = true(ell+2,1);
                        colSel(1:2) = false;
                        colSel(si+2) = false;
                        if(any(rowSel)) % Only if any non-zero
                            % NOTE: Only rowSel is actually used... (no need for special scalar case)
                            subsS = o.optSubs(rowSel,colSel); % Subtensor subs
                            Vs = betaSet{j}(curI:(curI + size(subsS,1) - 1));
                            curI = curI + size(subsS,1);
                            if(sum(rowSel) >= 1 && sum(colSel) < 1)
                                subTensor.update(1, Vs); % Scalar special case
                            else
                                subTensor.update(subsS, Vs);
                            end
                        end
                    end
                    
                    % Update tensor with subtensor
                    o.Psi{ell,j}.setSubTensor( s, subTensor );
                end
                %{
                beta = betaSet{j};
                ii = 0;
                for ell = 1:j
                    % Create subtensors from beta
                    % Create subtensor of correct size
                    subSz = o.Psi{ell,j}.getSubTensorSize( s );
                    subTensor = mrfs.grm.sparsetensor( subSz );
                    
                    % Extract corresponding indices
                    subElem = prod( subTensor.size());
                    sel = (betaI > ii & betaI <= ii + subElem);
                    
                    % Get filtered and shifted indices/values
                    curI = betaI(sel) - ii;
                    curV = betaV(sel)./ell; % Implicit averaging over ell estimates
                    curSubs = subTensor.getSubs(curI);
                    
                    % Update subTensor and then tensor
                    if(~isempty(curSubs)) % Only if any non-zero
                        subTensor.update(curSubs, curV);
                    end
                    o.Psi{ell,j}.setSubTensor( s, subTensor );
                    ii = ii + subElem;
                end
                %}
            end
        end
        
        % Get optimization set over the beta set
        function betaSetOptI = getBetaSetOptI( o, s )
            if(~o.lock); error('Cannot get beta set optI without lock set.'); end
            k = o.getK();
            betaSetOptI = cell(k,1);
            for j = 1:k
                optICell = cell(j,1);
                nElem = 0;
                for ell = 1:j
                    sel = (o.optSubs(:,1) == ell) & (o.optSubs(:,2) == j);
                    C = o.optSubs(sel,(1:ell)+2);
                    subSz = o.Psi{ell,j}.getSubTensorSize( s );
                    
                    % Check all columns for s
                    temp = cell(ell,1);
                    for c = 1:ell
                        % Create row and column selectors
                        cSel = false(1,ell);
                        cSel(c) = true;
                        rSel = (C(:,cSel) == s); % Filter based on s
                        % Check to make sure there are indices to add
                        if(sum(rSel) > 0)
                            if(subSz == 1)
                                temp{c} = 1;
                            elseif(length(subSz) == 1)
                                temp{c} = C(rSel, ~cSel); % Just index directly for scalars
                            else
                                filtC = C(rSel, ~cSel); % Get other columns
                                cellFiltCperm = mat2cell(filtC, size(filtC,1), ones(1,size(filtC,2)));
                                temp{c} = sub2ind(subSz, cellFiltCperm{:});
                            end
                        end
                    end
                    
                    % Add shifted indices to optICell
                    optICell{ell} = cell2mat(temp) + nElem;
                    nElem = nElem + prod(subSz);
                end
                betaSetOptI{j} = sort(cell2mat(optICell));
            end
        end
        
        % Get bounds for each subparameter of beta vectors
        function Ibounds = getIBounds(o)
            p = o.getP(); k = o.getK();
            Ibounds = NaN(k,2);
            nElem = 0;
            for j = 1:k
                Ibounds(j,1) = nElem+1;
                nElem = nElem+p^(j-1);
                Ibounds(j,2) = nElem;
            end
        end

                % Get the set of z vectors
        function ZtSet = getPairedZtSet(o, s, Xt, pairedIdx )
            if(~o.lock); error('Cannot get Zt set without lock set.'); end
            %betaSetOptI = o.getBetaSetOptI(s);
            k = o.getK(); p = size(Xt,2);
            n = length(pairedIdx{1});
            m = length(pairedIdx{2});
            
            % Setup XtCell
            XtCell = mat2cell(Xt(pairedIdx{1},:), n, ones(p,1));
            YtCell = mat2cell(Xt(pairedIdx{2},:), m, ones(p,1));
            ZtSet = cell(k,1);
            for j = 1:k
                ZIJVcell = cell(j,3);
                nElem = 0;
                for ell = 1:j
                    % Get optSubs
                    ZsCell = cell(1,ell);
                    for si = 1:ell
                        rowSel = all( bsxfun(@eq, o.optSubs(:,[1,2,si+2]), [ell,j,s]), 2);
                        colSel = true(ell+2,1);
                        colSel(1:2) = false;
                        colSel(si+2) = false;
                        subsS = o.optSubs(rowSel,colSel); % Subtensor subs
                        nSubs = size(subsS,1);
                        if(nSubs == 0); ZsCell{si} = []; continue; end
                        
                        % Create ZsCell
                        if(ell == 1) % Special scalar case
                            if(s <= p)
                                ZsCell{si} = sparse(ones(n,nSubs)); % \sqrt{x_s0} o^(l-1) = \sqrt{x_s0} o^0 = ones
                            else
                                ZsCell{si} = sparse(ones(m,nSubs)); % \sqrt{x_s0} o^(l-1) = \sqrt{x_s0} o^0 = ones
                            end
                        elseif(ell == 2)
                            if(s <= p)
                                xSel = subsS <= p;
                                xOnly = reshape( cell2mat( XtCell(subsS(xSel)') ), n, sum(xSel)).^(1/j);
                                C = mean(reshape( cell2mat( YtCell(subsS(~xSel)'-p) ), m, sum(~xSel)).^(1/j), 1); % Average over all in Y
                                yConst = repmat( C, n, 1 );
                                tempZs = NaN(n,nSubs);
                                tempZs(:,xSel) = xOnly;
                                tempZs(:,~xSel) = yConst;
                            else
                                ySel = subsS > p;
                                yOnly = reshape( cell2mat( YtCell(subsS(ySel)'-p) ), m, sum(ySel)).^(1/j);
                                C = mean(reshape( cell2mat( XtCell(subsS(~ySel)') ), n, sum(~ySel)).^(1/j), 1); % Average over all in Y
                                xConst = repmat( C, m, 1 );
                                tempZs = NaN(m,nSubs);
                                tempZs(:,ySel) = yOnly;
                                tempZs(:,~ySel) = xConst;
                            end
                            ZsCell{si} = tempZs;
                        elseif(ell == 4)
                            if(s <= p)
                                xOnly = reshape( cell2mat( XtCell(subsS(:,1)) ), n, nSubs).^(1/j);
                                C = mean(reshape( prod( cell2mat(YtCell(subsS(:,2:3)-p)), 2), m, nSubs ).^(1/j), 1);
                                yConst = repmat( C, n, 1 );
                                tempZs = xOnly.*yConst;
                            else
                                yOnly = reshape( cell2mat( YtCell(subsS(:,3)-p) ), m, nSubs).^(1/j);
                                C = mean(reshape( prod( cell2mat(XtCell(subsS(:,1:2))), 2), n, nSubs ).^(1/j), 1);
                                xConst = repmat( C, m, 1 );
                                tempZs = yOnly.*xConst;
                            end
                            ZsCell{si} = tempZs;
                        else
                            error('Only 1,2,4 are implemented for getPairedZtSet');
                        end
                    end
                     
                    % Get Zs
                    Zs = cell2mat(ZsCell);
                    [ZIJVcell{ell,1}, Jsubs, ZIJVcell{ell,3}] = find(Zs);
                    ZIJVcell{ell,2} = Jsubs + nElem;
                    nElem = nElem + size(Zs,2); % Update for next ell
                end
                if(s <= p)
                    ZtSet{j} = sparse(cell2mat(ZIJVcell(:,1)), cell2mat(ZIJVcell(:,2)), cell2mat(ZIJVcell(:,3)), n, nElem);
                else
                    ZtSet{j} = sparse(cell2mat(ZIJVcell(:,1)), cell2mat(ZIJVcell(:,2)), cell2mat(ZIJVcell(:,3)), m, nElem);
                end
            end
        end
        
        % Get the set of z vectors
        function ZtSet = getZtSet(o, s, Xt )
            if(~o.lock); error('Cannot get Zt set without lock set.'); end
            %betaSetOptI = o.getBetaSetOptI(s);
            k = o.getK(); p = o.getP();
            n = size(Xt,1);
            assert(p == size(Xt,2), 'Xt wrong size');
            
            % Get Ibounds  and XtCell for computation later
            %Ibounds = o.getIBounds();
            XtCell = mat2cell(Xt, n, ones(p,1));
            ZtSet = cell(k,1);
            for j = 1:k
                ZIJVcell = cell(j,3);
                nElem = 0;
                for ell = 1:j
                    % Get optSubs
                    ZsCell = cell(1,ell);
                    for si = 1:ell
                        rowSel = all( bsxfun(@eq, o.optSubs(:,[1,2,si+2]), [ell,j,s]), 2);
                        colSel = true(ell+2,1);
                        colSel(1:2) = false;
                        colSel(si+2) = false;
                        subsS = o.optSubs(rowSel,colSel); % Subtensor subs
                        nSubs = size(subsS,1);
                        % Create ZsCell
                        if(ell == 1) % Special scalar case
                            ZsCell{si} = sparse(ones(n,nSubs)); % \sqrt{x_s0} o^(l-1) = \sqrt{x_s0} o^0 = ones
                        elseif(ell == 2)
                            ZsCell{si} = reshape( cell2mat(XtCell(subsS)'), n, nSubs ).^(1/j); % Just extract cell mat (weird issue when ziSelSubs is vector)
                        else
                            ZsCell{si} = reshape( prod( cell2mat(XtCell(subsS)), 2), n, nSubs ).^(1/j);
                        end
                    end
                     
                    % Get Zs
                    %NOTE: Since we only consider one unique parameter we need factorial(ell).
                    % For example, when ell = 3, it would usually be ell but then there are 2 since the inner product is with a matrix so 3*2 = 6 = 3!.
                    Zs = factorial(ell) * cell2mat(ZsCell); % Push factorial(ell) term into the Z matrix for simplicity
                    [ZIJVcell{ell,1}, Jsubs, ZIJVcell{ell,3}] = find(Zs);
                    ZIJVcell{ell,2} = Jsubs + nElem;
                    nElem = nElem + size(Zs,2); % Update for next ell
                end
                ZtSet{j} = sparse(cell2mat(ZIJVcell(:,1)), cell2mat(ZIJVcell(:,2)), cell2mat(ZIJVcell(:,3)), n, nElem);
                
                %{
                % Get indices for this j
                zi = betaSetOptI{j};
                ZIJVcell = cell(j,3);
                nElem = 0;
                for ell = 1:j
                    % Extract subscripts using Ibounds
                    ziSel = zi(zi >= Ibounds(ell,1) & zi <= Ibounds(ell,2));
                    subSz = o.Psi{ell,j}.getSubTensorSize( s );
                    subsCell = cell(1,length(subSz));
                    [subsCell{:}] = ind2sub(subSz, ziSel - nElem);
                    ziSelSubs = cell2mat(subsCell);
                    
                    % Get Z for the given subscripts
                    if(ell == 1)
                        Zs = sparse(ones(n,length(ziSel))); % \sqrt{x_s0} o^(l-1) = \sqrt{x_s0} o^0 = ones
                    elseif(ell == 2)
                        Zs = reshape( cell2mat(XtCell(ziSelSubs)'), n, length(ziSel) ).^(1/j); % Just extract cell mat (weird issue when ziSelSubs is vector)
                    else
                        Zs = reshape( prod( cell2mat(XtCell(ziSelSubs)), 2), n, length(ziSel) ).^(1/j);
                    end
                    
                    % Get sparse coordinates but modify Jsubs according to zi(sel)
                    [ZIJVcell{ell,1}, Jsubs, ZIJVcell{ell,3}] = find(Zs);
                    ZIJVcell{ell,2} = ziSel(Jsubs); % Map Jsubs to correct indices
                    
                    nElem = nElem + prod(subSz); % Update for next ell
                end
                
                ZtSet{j} = sparse(cell2mat(ZIJVcell(:,1)), cell2mat(ZIJVcell(:,2)), cell2mat(ZIJVcell(:,3)), n, nElem);
                %fprintf('Debug testing code\n'); A = {sparse(1:3)',sparse(4:6)',sparse(7:9)',sparse(10:12)'}; p = length(A); n = length(A{1}); subs = [1 2; 2 3; 3 4]; Zsubs = reshape(prod(cell2mat(A(subs)),2), n, size(subs,1));
                %}
            end
        end
        
        % Utility functions
        function k = getK(o)
            k = size(o.Psi,1);
        end
        
        function p = getP(o)
            p = o.Psi{1,1}.size();
        end
        
        % Debug fill
        function debugFill(o)
            k = o.getK();
            for j = 1:k
                for ell = 1:j
                    % Get subscripts for all entries
                    sz = o.Psi{ell,j}.size();
                    subCell = cell(1,ell);
                    [ subCell{:} ] = ind2sub( sz, (1:prod(sz))' );
                    subs = cell2mat(subCell);
                    
                    % Create special numbers {ell}{j}.{s1}{s2}{s3}...
                    V = sum(bsxfun(@times, [ell*ones(size(subs,1),1), j*ones(size(subs,1),1), subs], 10.^(1:-1:(-size(subs,2))) ),2);
                    
                    % Fill sparsetensor with these values
                    o.Psi{ell,j}.update(subs,V);
                end
            end
        end
        
        % Debug to modify an entry directly
        function debugUpdate(o, ell, j, subs, vals)
            o.Psi{ell,j}.update(subs, vals);
        end
    end
end

%clear grm; k=3; p=3; grm = mrfs.grm.GRMParameters(k,p); grm.debugFill(); fprintf('\n\n<<Psi(all)>>\n\n'); for j = 1:size(grm.Psi,1); for ell = 1:j; fprintf('ell=%d,j=%d\n',ell,j); disp(grm.Psi{ell,j}.full());end;end; disp(grm.getBetaSet(1)); grm.setBetaSet(2, {1,(2:5)',(6:18)'} ); fprintf('After setting beta_2 to 1:18\n'); for j = 1:k; for ell = 1:j; fprintf('ell=%d,j=%d\n',ell,j); disp(grm.Psi{ell,j}.full());end;end; betaSetOptI = grm.getBetaSetOptI(2); fprintf('betaSetOptI for s = 2\n'); for j = 1:k; fprintf('betaSetOptI_%d\n',j); disp(betaSetOptI{j}); end; fprintf('boundsI\n'); disp(grm.getIBounds()); n=10; Xt=reshape((1:(n*p)),n,p); fprintf('Xt\n'); disp(Xt); Zset = grm.getZtSet(2,Xt); for j = 1:k; fprintf('Zset_2(%d)\n',j); disp(full(Zset{j})); end; grm.initOptSubs(2); fprintf('Optimization subscripts with initType=2 special case\n'); disp(grm.optSubs); grm.addOptSubs([3 3 2 2 2]); fprintf('after adding 33222\n'); disp(grm.optSubs); fprintf('Removing 33222, 33123, 111, 221 \n'); grm.removeOptSubs([3,3,2,2,2;3,3,1,2,3]); grm.removeOptSubs([1,1,1; 2,2,1]); disp(grm.optSubs);
