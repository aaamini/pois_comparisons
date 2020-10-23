function [Xt, labels, nDim] = load_data(dataset, nDim)
%load_count_dataset - Loads count dataset and filters to nDim
%  With the exception of "brca", each dataset is reduced to nDim
%  by sorting the variables by the sum/mean over all instances.
%  For "brca", the variables are sorted by variance instead of 
%  sum/mean because variance is often more important in biology.

Xt = []; labels = {}; % Placeholders

if(strcmp(dataset,'brca'))
    %% If "brca", first filter variables by variance, then take log of raw counts
    % Load raw BRCA counts
    load(sprintf('../data/%s-raw.mat',dataset),'Xt','labels');
    
    % Filter based on variance
    [~,sortedI] = sort(var(log(Xt+1)),'descend');
    Xt = Xt(:,sortedI(1:nDim));
    labels = labels(sortedI(1:nDim));
    
    % Sort by sum/mean
    [Xt, labels] = transform( Xt, labels, nDim );

    % Take log(x+1) transformation
    Xt = floor(log(Xt+1));
else
    %% Otherwise simply load dataset and filter dimensions
    load(sprintf('../data/%s.mat',dataset),'Xt','labels');
    if ~exist('labels')
        labels = {};
    end
    if ~isempty(nDim)
        [Xt, labels] = transform( Xt, labels, nDim );
    end
end

% Error check
if(nDim > size(Xt,2)); warning('Requested number of dimensions (nDim = %d) is larger than the loaded dataset \n which has dimension (d = %d)', nDim, size(Xt,2)); end
% fprintf('\nSuccessfully loaded dataset %s with\nd = %d, n = %d, sparsity = %3.1f\% and sum of all counts = %d\n\n', dataset, size(Xt,2), size(Xt,1),  100*nnz(Xt)/prod(size(Xt)), full(sum(Xt(:))));
fprintf('\nSuccessfully loaded dataset "%s"\nsize        = %d x %d\nsparsity    = %3.1f %%\n', ...
    vis.map_dataset(dataset), size(Xt,1), size(Xt,2),  100*(1-nnz(Xt)/prod(size(Xt))));
nDim = size(Xt,2);

vis.print_freq_info(Xt)

end

%% Sort and filter to nDim dimensions
function [ transXt, transLabels] = transform( Xt, labels, nDim)
    % Determine wordCounts
    wordCounts = full(sum(Xt));
    [~, idx] = sort(wordCounts, 'descend');

    transXt = Xt(:, idx);
    % Sort rows based on wordCounts
    if (nDim > 0)
        transXt = transXt(:,1:nDim);
    end
    
    if isempty(labels)
        transLabels = {};
    else    
        transLabels = labels(idx);
        % Filter words if not -1 or negative
        if(nDim > 0 && nDim <= length(transLabels))
            % transXt = transXt(:,1:nDim);
            transLabels = transLabels(1:nDim);
        end
    end
    % % Determine wordCounts
    % wordCounts = full(sum(Xt));
    % [~, idx] = sort(wordCounts, 'descend');

    % % Sort rows based on wordCounts
    % transXt = Xt(:, idx);
    % if isempty(labels)
    %     transLabels = {};
    % else    
    %     transLabels = labels(idx);
    %     % Filter words if not -1 or negative
    %     if(nDim > 0 && nDim <= length(transLabels))
    %         transXt = transXt(:,1:nDim);
    %         transLabels = transLabels(1:nDim);
    %     end
    % end
end