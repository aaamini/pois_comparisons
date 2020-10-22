function [XtSampleS, trainTime, sampleTime] = fit_and_sample(XtTrain0, lamVec0, nSamples0)

curr_path = fileparts(mfilename('fullpath')); % retreive the path the function file lives in
input_fpath = fullfile(curr_path, 'pois_input.mat');
output_fpath = fullfile(curr_path, 'pois_output.mat');

% define local variables se to allow to save to .mat file
XtTrain = XtTrain0;
lamVec = lamVec0;
nSamples = nSamples0;
% save(input_fpath, 'XtTrain','lamVec','nSamplesTune')
save(input_fpath, 'XtTrain','lamVec','nSamples')
if ~system(sprintf('cd %s && Rscript fit_and_sample.R', curr_path))
    fprintf('  R script ran successfully.\n')
end
S = load(output_fpath);
XtSampleS = S.XtSampleS;
trainTime = S.trainTime;
sampleTime = S.sampleTime;

delete(input_fpath)
delete(output_fpath)