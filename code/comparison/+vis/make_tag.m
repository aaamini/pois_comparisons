function tag = make_tag(exset)
tag = sprintf('_%s_%d_%d_cv%d_%s_%s', ...
    exset.datasetLabel, exset.nDim, exset.nSamples, exset.nCV, exset.mmdAggType, ...
    vis.code_method_names(exset.batch));