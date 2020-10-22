function mmdPairValue = pair_complement_mmd(X, Y, sigmaVec, nBasis, max_nPairs, aggType)

    if (nargin < 5), max_nPairs = Inf;  end
    if (nargin < 6), aggType = 'max';   end
    % Compute mmd pairs in parallel
    %  max_nPairs = 200;
    tic
    p = size(X,2);
    if nchoosek(p, 2) >  max_nPairs
        % a sample of pairwise combinations (rough estimate)
        C = unique(reshape(randsample(p, 2*max_nPairs, true), [], 2), 'rows');
    else
        % all pairwise combinations
        C = nchoosek(1:p,2);
    end
    nC = size(C,1);

    switch aggType
        case 'mean'
            aggFunc = @mean;
        case 'max'
            aggFunc = @max;
        case 'none'
            aggFunc = @(x) x;
    end
    % mmdPairValue = NaN(nC,1);
    mmdPairValue = cell(nC,1);

    fprintf('  [Computing pair-comp. MMD (nBasis = %d, nPairs = %d) ... ', nBasis, nC)
    parfor ii = 1:nC
        ind = setdiff(1:p, C(ii,:));
        % [~, mmdPair] = MMDFourierFeature(X(:,ind), Y(:,ind), sigmaVec, nBasis);
        % mmdPair is a vector of the same size as sigmaVec
        [~, mmdPair] = fast_mmd(X(:,ind), Y(:,ind), sigmaVec, nBasis);
        
        %mmdPairValue(ii) = mean(mmdPair);  
        % mmdPairValue(ii) = aggFunc(mmdPair);
        mmdPairValue{ii} = aggFunc(mmdPair);
        %fprintf('  Finished pair = [%d, %d] with value = %g in %g s\n', s, t, mmdPairValueVec(ii), toc(tts));
    end
    %fprintf('%2.1f (s).  Type:%s = %2.2f]\n', toc, aggType, mean(mmdPairValue))
    fprintf('%2.1f (s).  Type:%s = %2.2f]\n', toc, aggType, mean(vertcat(mmdPairValue{:}))) %mean([mmdPairValue{:}]))

    % Make symmetric matrix from the values of parallel loop
    %     mmdPairValue = NaN(p,p);
    %     ind = sub2ind(size(mmdPairValue),C(:,1),C(:,2));
    %     mmdPairValue(ind) = mmdPairValueVec;
    %     ind = sub2ind(size(mmdPairValue),C(:,2),C(:,1));
    %     mmdPairValue(ind) = mmdPairValueVec;
