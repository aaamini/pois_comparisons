function test_seq_func(data_cell)

SAVE_DATA = true;
SAVE_FIG = true;
rng('default')

%% Experiment setup
exset = struct();
exset.batch = {...
        'po-is-tune',  'ind-mult', 'mixture-tune', ... 
        'copula-mult', 'bootstrap', ... %'po-is-glmnet', ... 
        'ind-negbin',  'copula-poi','tpgm-tune'};

exset.nWorkers = 7;
myCluster = parcluster('local');
myCluster.NumWorkers = exset.nWorkers; 
saveProfile(myCluster);    

exset.nSamples = 1000; 
exset.nCV = 5;
exset.mmdAggType = 'none';  % aggregation method across Gaussian kernel bandwidth (max | mean | none) during simulation
exset.sigmaVec = 10.^(-2:0.2:0.8)';


for i = 1:length(data_cell)/2
    % Set up data
    exset.datasetLabel = data_cell{2*i-1};
    exset.nDim = data_cell{2*i};
    
    [Xt, labels, exset.nDim] = experim.load_data(exset.datasetLabel, exset.nDim);
    exset.tag = vis.make_tag(exset);

    %% Run experiments
    total_time = tic;
    cvAll = experim.create_cv_array(exset.nCV, length(exset.batch));
    for mi = 1:length(exset.batch)
        % Compute model and evaluate
        cvAll(:,mi) = experim.test(mi, Xt, exset);
    end
    fprintf('Total time = %2.2f (h)\n', toc(total_time)/3600)

    dat_fname = sprintf('res%s.mat', exset.tag);
    if SAVE_DATA 
        save(dat_fname, 'cvAll','exset')
    end

    %%
    fig_fanme = sprintf('fig%s', exset.tag); % Add filename to export figure to file

    sig_agg_method = 'mean'; % aggregation method across Gaussian kernel bandwidth (max | mean | none) during visualization
    % sig_index_include = 1:length(exset.sigmaVec);
    figure(1), aggEMV = vis.plot_hist(cvAll, exset.datasetLabel, ...
                            exset.nDim, {}, 'on', sig_agg_method);
    if SAVE_FIG
        paper_size = [6 8];
        % paper_size = [14 6];
        % paper_size = [12 6];
        set(gcf, 'Color', 'w', 'PaperUnits','inches', ...
            'PaperPosition',[0.1 0.025 paper_size], 'PaperSize', paper_size + [0.2 0.05])
        print('-dpdf', [fig_fanme '.pdf'])
    end
end