function nAdded = update2k( grm, k )
assert(k > 2, 'update2k must be have k > 2');

%% Get nonzero cliques of size k-1
[~, ~, subsUniq] = grm.Psi{k-1,k-1}.getAllUniq();
if(isempty(subsUniq)); nAdded = 0; return; end

%% Add appropriate cliques
% For each clique of size k-1 
% check every other clique that intersects in at least k-2 nodes
% if there are k such cliques then add

nSubs = size(subsUniq,1);
newCliques = cell(nSubs);
p = grm.getP();
for i1 = 1:nSubs
    nSubIntersect = zeros(p,1); % Number of cliques that intersect with at least k-2 nodes
    for i2 = 1:nSubs
        if(i1 == i2); continue; end % Skip itself
        inter = intersect(subsUniq(i1,:), subsUniq(i2,:));
        diff = setdiff(subsUniq(i2,:), subsUniq(i1,:));
        assert(length(diff) >= 1, 'Set difference is too small');
        assert(length(inter) <= k-2, 'Intersection is too large');
        
        % Check if intersection is at least k-2
        if(length(inter) == k-2)
            assert(length(diff) == 1, 'Diff should be 1');
            nSubIntersect(diff) = nSubIntersect(diff) + 1;
        end
    end
    assert(all(nSubIntersect <= k-1), 'Not possible to intersect more than k times');
    addedNodes = find(nSubIntersect == k-1);
    if(~isempty(addedNodes))
        % Add new clique for every clique such that the node had k-1 intersections with subsets
        newCliques{i1} = [repmat(subsUniq(i1,:),length(addedNodes),1), addedNodes ];
    end
end

% Prepare new subs
newSubs = unique( sort(cell2mat(newCliques(:)),2), 'rows');
nAdded = size(newSubs,1);

%% Update grm
ell = k; j = k;
if(~isempty(newSubs))
    grm.addOptSubs( [ell*ones(size(newSubs,1),1), j*ones(size(newSubs,1),1), newSubs] );
end

end

% Test code
% grm = mrfs.grm.GRMParameters(4,10,0); mrfs.grm.stage.init1(grm); mrfs.grm.stage.init2( grm, 10); grm.debugUpdate(2,2,nchoosek(1:4,2),ones(nchoosek(4,2),1)); mrfs.grm.stage.update2k(grm,3); fprintf('grm after 3 update\n'); disp(grm.optSubs); grm.debugUpdate(3,3,nchoosek(1:4,3),ones(nchoosek(4,3),1)); mrfs.grm.stage.update2k(grm,4); fprintf('grm after 4 update\n'); disp(grm.optSubs);