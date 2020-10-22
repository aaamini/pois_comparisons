function aggEMV = aggregate_eval_metric(cvAll, sig_agg_method, sig_index_include)
% Averages the eval metric values across cross-validation batches

if nargin < 2, sig_agg_method = 'mean'; end
if nargin < 3, sig_index_include = []; end
    
aggEMV = []; % average eval metric values
if ~iscell(cvAll(1,1).evalMetricValues)  
    % already aggregated across bandwith (sigma)
    fprintf('Already sig-aggregated. Only averaging over CV batches...')
    for j = 1:size(cvAll,2) 
        aggEMV = [aggEMV mean([cvAll(:,j).evalMetricValues], 2)];
    end
else 
    % not aggregated across sigma
    n_methods = size(cvAll,2);
    n_cv = size(cvAll,1);
    for j = 1:n_methods % loop over methods
        tempMat = [];
        for i = 1:n_cv % loop over CV batch
            temp = cvAll(i,j).evalMetricValues;
            temp = [temp{:}];  % temp will have dimensions length(sigmaVec) x (# of MMD pairs)
            if ~isempty(sig_index_include)
                if i==1 && j== 1, fprintf('Only including sig indices...', sig_index_include), end
                temp = temp(sig_index_include, :); % include only the desired bandwidths
            end           
            if strcmp(sig_agg_method, 'none')
                if i==1 && j== 1, fprintf('not aggregating over sigma.'), end
                % temp = vertcat(temp{:});
            else
                switch sig_agg_method
                case 'max'
                    if i==1 && j== 1, fprintf('max-pooling over sigma.'), end
                    temp = max(temp, [], 1);
                case 'mean'
                    if i==1 && j== 1, fprintf('mean-pooling over sigma.\n'), end
                    temp = mean(temp, 1);
                end
                
            end
            temp = temp(:);
            tempMat = [tempMat temp];
        end
        aggEMV = [aggEMV mean(tempMat, 2)];
    end
end
fprintf('\n')