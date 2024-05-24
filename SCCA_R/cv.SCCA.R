cv.SCCA <- function(x, y, lambda.alpha, lambda.beta, nfolds=5, alpha.init, beta.init, eps = 1e-3, niter = 10, standardize = TRUE){
  
  n <- nrow(x)
  id.folds <- cut(seq(1:n), breaks = nfolds, labels = 1:nfolds)
  id.folds <- sample(id.folds, n, replace = FALSE)
  id.folds <- as.numeric(id.folds)
  rho <- matrix(0, length(lambda.alpha), length(lambda.beta))
  
  
  for(i.lambda in 1:length(lambda.alpha)){
    for(j.lambda in 1:length(lambda.beta)){
      for(i in 1:nfolds){
        
        obj <- SCCA(x[id.folds!=i, ], y[id.folds!=i, ], alpha.init = alpha.init, beta.init = beta.init, lambda.alpha = lambda.alpha[i.lambda], lambda.beta = lambda.beta[j.lambda], eps = eps, niter = niter, standardize = standardize)
        
        rho[i.lambda, j.lambda] <- rho[i.lambda, j.lambda]+abs(cor(x[id.folds==i, ]%*%obj$beta, y[id.folds==i,] %*% obj$alpha))
      }
    }
  }
  rho[is.na(rho)] <- 0
  id.alpha.max<-max(which(apply(rho, 1, max)==max(rho)))
  id.beta.max<-max(which(apply(rho, 2, max)==max(rho)))
  rho[is.na(rho)] <- 0
  
  list(rho = rho, lambda.alpha = lambda.alpha, beta.beta = lambda.beta, bestlambda.alpha = lambda.alpha[id.alpha.max], bestlambda.beta = lambda.beta[id.beta.max])
}