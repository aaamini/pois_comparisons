function grm = fitgrm( Xt, k, opts )
assert( nargin < 3 || isempty(opts) || isstruct(opts), 'opts should be struct or empty or not given');
if(nargin < 3 || isempty(opts))
    opts = struct();
end
assert(isstruct(opts), 'Opts should be a struct by now');
if(~isfield(opts,'initType'))
    opts.initType = 1;
end

%% Initialize
[n,p] = size(Xt);
if(isfield(opts,'pairedIdx') && ~isempty(opts.pairedIdx))
    p = 2*p;
end
if(isstruct(opts) && isfield(opts, 'grm'))
    grm = opts.grm;
else
    grm = mrfs.grm.GRMParameters( k, p, opts.initType );
end

%% Extract needed parameters
betaSet = cell(p,1);
onlyNodeP = false(p,1);
for s = 1:p
    % Only node parameter (trivial case)
    if sum(sum(grm.optSubs(:,3:end) == s)) <= 1 
        betaSet{s} = cell(k,1);
        nElem = 0;
        for j = 1:k
            nElem = nElem + p^(j-1);
            if(j == 1)
                betaSet{s}{j} = log(mean(Xt(:,s)));
            else
                betaSet{s}{j} = sparse(nElem,1);
            end
        end
        onlyNodeP(s) = true;
    end
end
fprintf('Finished initialization of betaSet and ZtSet\n');


%% Fit each node-wise regression
% Filter only to non-only-nodes
grm.setLock(); % Just a error checking mechanism to make sure optSubs does not change
idxMap = find(~onlyNodeP);
betaSetMod = betaSet(~onlyNodeP);
prefixFormat = sprintf('s = %%%dd, ',floor(log10(p))+1);
if(isfield(opts,'nWorkers') && opts.nWorkers == 1)
    %% With just one worker, merely run for loop
    for sMod = 1:length(betaSetMod)
        betaSetMod{sMod} = loopFunc(idxMap(sMod), grm, Xt, opts, prefixFormat);
    end
else
    %% Initialize parallel pool of workers and run in parallel
    if(isfield(opts,'nWorkers'))
        nWorkers = opts.nWorkers;
    else
        [~,nStr] = system('grep --count "^processor" /proc/cpuinfo');
        nWorkers = str2double(nStr);
    end
    % addpath('matlab-utils');
    initParallel(nWorkers);
    parfor sMod = 1:length(betaSetMod)
        betaSetMod{sMod} = loopFunc(idxMap(sMod), grm, Xt, opts, prefixFormat);
    end
end

betaSet(~onlyNodeP) = betaSetMod;

%% Update grm parameters
for s = 1:p
    grm.setBetaSet(s, betaSet{s});
end
grm.releaseLock();

end

function betaSet = loopFunc(s, grm, Xt, opts, prefixFormat)
    opts.outPrefix = sprintf(prefixFormat, s);
    if(isfield(opts,'pairedIdx') && ~isempty(opts.pairedIdx))
        p = size(Xt,2);
        % Give different data and scaled ZtSet depending on s
        if(s <= p)
            curXt = Xt(opts.pairedIdx{1}, s );
        else
            curXt = Xt(opts.pairedIdx{2}, s-p );
        end
        ZtSet = grm.getPairedZtSet(s, Xt, opts.pairedIdx);
        betaSet = mrfs.grm.fitnodereg( grm.getBetaSet(s), curXt, ZtSet, opts );
    else
        betaSet = mrfs.grm.fitnodereg( grm.getBetaSet(s), Xt(:,s), grm.getZtSet(s, Xt), opts );
    end
end

