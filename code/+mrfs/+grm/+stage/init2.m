function nAdded = init2( grm, pThresh )

%% Get the max pThresh nodes from current grm
p = grm.getP();
I = (1:p)';
V = zeros(size(I)); % By assumption of sparse vectors
[Inz, Vnz] = grm.Psi{1,1}.getAll();
V(Inz) = Vnz;
[~,sortI] = sort(V,1,'descend');
topNodes = I(sortI);

%% Add all combos of these nodes
C = nchoosek(topNodes(1:pThresh),2);
ell = 2; j = 2;
nAdded = size(C,1);
grm.addOptSubs( [ell*ones(nAdded,1), j*ones(nAdded,1), C] );

end

% Test code
% grm = mrfs.grm.GRMParameters(3,10,0); mrfs.grm.stage.init1(grm); grm.debugUpdate(1,1,8,10); mrfs.grm.stage.init2( grm, 5); grm