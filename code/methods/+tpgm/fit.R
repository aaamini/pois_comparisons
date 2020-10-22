# Fits the XMRF, formerly xmrf.wrapper.R

# install.packages("XMRF")
# myPaths <- .libPaths() 
# myPaths <- c(myPaths[2], myPaths[1])
# .libPaths(c("/home/arash/.R/lib",myPaths))

require(methods)

# Extract arguments
args = commandArgs(trailingOnly=TRUE)
# if(length(args) < 1) { paramsFile = "data/params.csv" } else { paramsFile = args[1] }
# if(length(args) < 2) { dataFile = "data/X.csv" } else { dataFile = args[2] }
# if(length(args) < 3) { thetaFile = "data/theta.csv" } else { thetaFile = args[3] }
paramsFile = args[1]
dataFile = args[2]
thetaFile = args[3]

# Load params
lamTable <- read.table(paramsFile,
                 header = FALSE,
                 sep = ",")
method = toString(lamTable[1,1])
lam = as.numeric(lamTable[1,2])
nWorkers = as.integer(lamTable[1,3])
R <- as.numeric(lamTable[1,4])
R0 <- as.numeric(lamTable[1,5])

# Load data
Xtable <- read.table(dataFile,
                 header = FALSE,
                 sep = ",")
X <- as.matrix(Xtable)
#show(method)
#show(lam)
#show(nWorkers)
#show(X)

# Load library and source files for XMRF
require(XMRF)
source('PGM.network.mod.R')
source('PGM.path.neighborhood.mod.R')
source('LPGM.network.mod.R')
source('LPGM.path.neighborhood.R')
source('Bsublin.R')

# Run program
if(method == 'spgm') {
	# Generate the matrix with values sublinearly truncated between R (upper bound) and R0 (lower bound)
	X <- round(Bsublin(X, R, R0))
    theta <- LPGM.network.mod(X,nlams=1,parallel=TRUE,nCpus=nWorkers,lambda=lam,sym=FALSE,th=0)
} else if(method == 'tpgm') {
    # Truncate values to R
	X[X > R] <- R
    theta = LPGM.network.mod(X,nlams=1,parallel=TRUE,nCpus=nWorkers,lambda=lam,sym=FALSE,th=0)
} else {
    theta = PGM.network.mod(X,nlams=1,parallel=TRUE,ncores=nWorkers,lambda=lam)
}
theta <- theta[[1]]

# Display results
#show(theta)
write.table(theta,thetaFile,row.names=FALSE,col.names=FALSE,sep=",")
