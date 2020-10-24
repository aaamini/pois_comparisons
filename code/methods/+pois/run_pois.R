#! /usr/bin/Rscript

# library(R.matlab)
library(pois)

# setwd("../..")
# source("potts_ising.R")

# print(getwd())

out <- R.matlab::readMat('pois_input.mat')
Xt <- out$XtTrain
n <-  as.numeric(out$nSamples)
bisect_flag <-as.logical(out$flag)
method <- as.character(out$method)

# print(method)
# set.seed(123)
set.seed(123)
# gam_est = gam_estim(as.data.frame(as.matrix(Xt)))
# theta_est = theta_estim(Xt,gam_est)
#   method = "glmnet"
#   method = "firth"

solver = ifelse(bisect_flag, "coord", "global")
# if (bisect_flag) {
#   trainTime = system.time( out <- fit_by_bisection(Xt, method=method) )["elapsed"]
# } else {
#   trainTime = system.time( out <- fit_pott_ising(Xt, method=method) )["elapsed"]
# }

trainTime = system.time( out <- fit_pois(Xt, solver = solver, method = method) )["elapsed"]
theta_est = out$Theta
gam_est = out$Gamma
gam_est = (gam_est + t(gam_est))/2
# image(Matrix(gam_est))
# sampleTime = system.time( XtSample <- toxGenCpp(n, theta_est, gam_est, burn_in = 5000) )["elapsed"]
# sampleTime = system.time( 
#   XtSample <- toxGenCpp2(n, theta_est, gam_est, burn_in = 5000, spacing = 100)
#   )["elapsed"]
sampleTime = system.time(
  XtSample <- sample_pois(n, theta_est, gam_est, burn_in = 5000, spacing = 100))["elapsed"]


R.matlab::writeMat("pois_output.mat",
         XtSample = XtSample,
         gam_est = gam_est,
         theta_est = theta_est,
         sampleTime = sampleTime,
         trainTime = trainTime)


