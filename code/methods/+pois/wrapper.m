function [XtSample, trainTime, sampleTime] = ... 
    wrapper(XtTrain0, nSamples0, pois_method0, bisect_flag0)
% wrapper for R code fitting the POIS model

    curr_path = fileparts(mfilename('fullpath')); % retreive the path the function file lives in
    input_fpath = fullfile(curr_path, 'pois_input.mat');
    output_fpath = fullfile(curr_path, 'pois_output.mat');
    XtTrain = XtTrain0;
    nSamples = nSamples0;
    method = pois_method0;
    flag = bisect_flag0;
    save(input_fpath, 'XtTrain', 'nSamples', 'flag', 'method')
    
    % system(sprintf('cd %s && Rscript --vanilla run_pois.R', curr_path))
    if ~system(sprintf('cd %s && Rscript run_pois.R', curr_path))
        fprintf('  R script ran successfully.\n')
    end
    
    S = load(output_fpath);
    XtSample = S.XtSample;
    trainTime = S.trainTime; 
    sampleTime = S.sampleTime;

    delete(input_fpath)
    delete(output_fpath)