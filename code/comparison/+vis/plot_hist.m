function aggEMV = plot_hist(cvAll, datasetLabel, nDim, exclude_methods, labelPos, sig_agg_method, sig_index_include)

if nargin < 6, sig_agg_method = 'mean'; end
if nargin < 7, sig_index_include = []; end
if nargin < 5, labelPos = 'on'; end

if nargin >= 4
    % filter exlucde_methods
    ex_idx = [];
    for mi = 1:size(cvAll, 2) % number of methods
       if any(strcmp(cvAll(1, mi).methodName, exclude_methods))
            ex_idx = [ex_idx mi];
       end
    end 
    cvAll(:, ex_idx) = []; % remove methods' data
end

batch = {cvAll(1,:).methodName};
methodNames = vis.map_method(batch);
dataName = vis.map_dataset(datasetLabel);
xLabelStr = 'Pair-complement Maximum Mean Discrepancy';

aggEMV = experim.aggregate_eval_metric(cvAll, sig_agg_method, sig_index_include);
[~, I] = sort(mean(aggEMV),'descend');
aggEMV = aggEMV(:, I);
methodNames = methodNames(I);

titleStr = sprintf('%s (d = %d)', dataName, nDim);
mhist(aggEMV, methodNames, xLabelStr, titleStr, labelPos);