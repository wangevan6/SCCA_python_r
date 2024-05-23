

#The function cv.SCCA.equal is the cross validation function for the first canonical pair, assuming that alpha and beta uses the same tuning parameter.
cv.SCCA.equal <- function(x, y, lambda, nfolds=5, alpha.init, beta.init, eps = 1e-3, niter = 20){
  n <- nrow(x)
  id.folds <- cut(seq(1:n), breaks = nfolds, labels = 1:nfolds)
  id.folds <- sample(id.folds, n, replace = FALSE)
  id.folds <- as.numeric(id.folds)
  rho <- matrix(0, length(lambda), nfolds)
  for(i.lambda in 1:length(lambda)){
    for(i in 1:nfolds){
      obj <- SCCA(x[id.folds!=i, ], y[id.folds!=i, ], alpha.init = alpha.init, beta.init = beta.init, lambda.alpha = lambda[i.lambda], lambda.beta = lambda[i.lambda], eps = eps, niter = niter)
      
      rho[i.lambda, i]<- cor(x[id.folds==i, ]%*%obj$beta, y[id.folds==i,] %*% obj$alpha)
    }
  }
  rho <- apply(rho, 1, mean)
  list(rho = rho, lambda = lambda, bestlambda = lambda[which.max(rho)])
}