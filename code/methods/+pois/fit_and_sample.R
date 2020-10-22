#! /usr/bin/Rscript
library(pois)

# out <- R.matlab::readMat("pois_input.mat")
out <- rmatio::read.mat("pois_input.mat")
Xt <- out$XtTrain
lamVec <- out$lamVec
nSamples <- out$nSamples

trainTime = system.time( 
    models <- fit_pois_glmnet_nocv(Xt, lambda = lamVec)
)["elapsed"]

n_mods = length(models)
sampleTime = system.time( 
    XtSampleS <- lapply(models, function(model) 
        sample_pois(nSamples, model$theta, model$gamh, burn_in = 5000, spacing = 100, verb = F))
)["elapsed"]

# print(XtSampleS)

# R.matlab::writeMat("pois_output.mat",
#          XtSampleS = XtSampleS,
#          trainTime = trainTime,
#          sampleTime = sampleTime)

rmatio::write.mat(list(XtSampleS = XtSampleS, trainTime = trainTime, sampleTime = sampleTime), 
                filename = "pois_output.mat")