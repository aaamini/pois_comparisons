l2norm_squared = function(X) Re(sum(Conj(t(X)) %*% X))

fast_mmd = function(X, Y, nBasis = 2^6, seed = NULL) {
  
  if (!is.null(seed)) set.seed(seed)
  d = ncol(X)
  if (ncol(Y) != d) stop('# of columns of X and Y should match.')
  
  n = nrow(X)
  m = nrow(Y)
  N = nBasis
  Z = matrix(rnorm(N*d), nrow=N) 
  ZXt = Z %*% t(as.matrix(X))
  ZYt = Z %*% t(as.matrix(Y))
  phi = function(sigma) rowMeans( exp(1i*ZXt/sigma) / sqrt(N))
  psi = function(sigma) rowMeans( exp(1i*ZYt/sigma) / sqrt(N))
  l2norm_squared( phi(1) - psi(1) )
}


d = 43
X = matrix(rnorm(100*d), nrow=100)
Y = X[50:60, ]
ncores = 4
nBasis = 2^6

seed = 1
all_indices = 1:ncol(X)  
system.time( out <-  parallel::mclapply(combn(all_indices, 2, simplify = F),
                    function(ind) {
                       ind =  setdiff(all_indices, ind)
                       fast_mmd(X[, ind], Y[, ind], nBasis, seed)
                    },
                    mc.cores = ncores) )["elapsed"]
length(unlist(out))


doParallel::registerDoParallel(ncores)
p = ncol(X)
C = combn(1:p, 2, simplify = F)
nC = length(C)
library(foreach)  
foreach(i=1:nC, .combine = "c") %dopar% {
    ind = setdiff(1:p, C[[i]])
    fast_mmd(X[, ind], Y[, ind], nBasis, seed)
}

