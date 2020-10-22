function methodCell = map_method(methodCell)
% Maps method names
method_map = struct();
method_map.bootstrap = 'Bootstrap';
method_map.mixture_tune = sprintf('Mix Pois');
method_map.po_is_glmnet = 'POIS-g';
method_map.po_is_tune = sprintf('POIS');
method_map.ind_negbin = 'Ind Neg Bin';
method_map.copula_poi = 'Copula Pois';
method_map.poisson_sqr_tune = 'Pois SQR';
method_map.tpgm_tune = 'T-PGM';
method_map.copula_mult = 'Copula Mult';
method_map.ind_mult = 'Ind Mult';

methodCell = vis.map_keys(methodCell, method_map);