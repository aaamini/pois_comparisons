function [trainIdxArray, testIdxArray] = cvsplit( n, nSplits, rndSeed )
if(nargin < 2); nSplits = 10; end
if(nargin < 3); rndSeed = []; end
if(~isempty(rndSeed))
    rng(rndSeed);
end

% Determine split index
permutation = randperm( n );
splitVec = round(linspace(0, n, nSplits+1));
trainIdxArray = cell(nSplits,1);
testIdxArray = cell(nSplits,1);
for j = 1:nSplits
    trainIdxArray{j} = [];
    for j2 = 1:nSplits
        slice = permutation((splitVec(j2)+1):splitVec(j2+1));
        if(j2 == j)
            testIdxArray{j} = slice;
        else
            trainIdxArray{j} = [trainIdxArray{j}, slice];
        end
    end
    trainIdxArray{j} = sort(trainIdxArray{j});
    testIdxArray{j} = sort(testIdxArray{j});
end

end

