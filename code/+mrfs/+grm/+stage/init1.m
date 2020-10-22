function nAdded = init1( grm )

%% Initialize all the node parameters to 1
p = grm.getP();
ell = 1; j = 1;
nAdded = p;
grm.addOptSubs( [ell*ones(p,1), j*ones(p,1), (1:p)'] );

end

% Test code
% grm = mrfs.grm.GRMParameters(3,10,0); mrfs.grm.stage.init1(grm); grm