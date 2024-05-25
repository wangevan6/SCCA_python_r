library(glmnet)


#### SCCA is the main functions that produces estimates of the canonical pairs. 
SCCA <- function(x, y, alpha.init = NULL, beta.init = NULL, lambda.alpha, lambda.beta, niter = 100, npairs = 1, init.method = c("sparse","uniform","svd","random"), alpha.current = NULL, beta.current = NULL, standardize = TRUE, eps=1e-4){
  p <- ncol(x)
  q <- ncol(y)
  n <- nrow(x)
  
  x <- scale(x, center = T,scale = standardize)
  y <- scale(y, center = T,scale = standardize)
  
  sigma.YX.hat <- cov(y, x)
  sigma.X.hat <- cov(x)
  sigma.Y.hat <- cov(y)
  
  alpha <- matrix(0, q, npairs)
  beta <- matrix(0, p, npairs)
  rho <- matrix(0, npairs, npairs)
  
  if(length(init.method)>1){init.method <- init.method[1]}
    
  if(missing(alpha.current)){npairs0 <- 0}else{ 
    npairs0 <- ncol(alpha.current)
    
    alpha[ , 1:npairs0] <- alpha.current
    beta[ , 1:npairs0] <- beta.current
    }
  

  if(missing(alpha.init)){
    if(missing(alpha.current)){
      
      obj.init <- init0(sigma.YX.hat, sigma.X.hat, sigma.Y.hat, init.method = init.method, npairs, n = n)

      
      alpha.init <- obj.init$alpha.init
      beta.init <- obj.init$beta.init
      }else{
        alpha.current <- as.matrix(alpha.current)
        beta.current <- as.matrix(beta.current)

      obj.init <- init1(sigma.YX.hat = sigma.YX.hat, sigma.X.hat=sigma.X.hat, sigma.Y.hat=sigma.Y.hat, init.method = init.method, npairs = npairs, npairs0 = npairs0, alpha.current = alpha.current, beta.current = beta.current, n = n, eps = eps)
      alpha.init <- obj.init$alpha.init
      beta.init <- obj.init$beta.init
      
      alpha[ , 1:npairs0] <- alpha.current
      beta[ , 1:npairs0] <- beta.current
    }
  }
  
  
  n.iter.converge <- rep(0, npairs - npairs0)
  
  for(ipairs in (npairs0 + 1):npairs){
    
    alpha.init <- as.matrix(alpha.init)
    beta.init <- as.matrix(beta.init)
    
    omega <- find.Omega(sigma.YX.hat, ipairs, alpha = alpha[ , 1:(ipairs - 1)], beta = beta[ , 1:(ipairs - 1)], y = y, x = x)
    
    x.tmp <- omega %*% x
    y.tmp <- t(omega) %*% y
    
    lambda.alpha0 <- lambda.alpha[ipairs - npairs0] 
    lambda.beta0 <- lambda.beta[ipairs - npairs0] 
    
    alpha0 <- alpha.init
    beta0 <- beta.init
    
    obj <- SCCA.solution(x = x, y = y, x.Omega = x.tmp, y.Omega = y.tmp, alpha0, beta0, lambda.alpha0, lambda.beta0, niter = niter, eps = eps)

    alpha[ , ipairs] <- obj$alpha
    beta[ , ipairs] <- obj$beta
    n.iter.converge[ipairs - npairs0] <- obj$niter
    
    if((ipairs < npairs)&(init.method == "sparse")){
      obj.init <- init1(sigma.YX.hat, sigma.X.hat, sigma.Y.hat, init.method=init.method, npairs, npairs0 = ipairs, alpha.current = alpha[ , 1:ipairs], beta.current = beta[ , 1:ipairs])
      alpha.init <- obj.init$alpha.init
      beta.init <- obj.init$beta.init
      }
  }
  
  list(alpha = alpha, beta = beta, alpha.init = alpha.init, beta.init = beta.init, n.iter.converge = n.iter.converge)
}

#### The function init0 finds the initial value when no canonical pairs have been obtained. If init.method="sparse", 
#### only one pair of initial value will be returned. For other options of init.method, the number of pairs of initial
#### values can be specified with the argument npairs.
init0 <- function(sigma.YX.hat, sigma.X.hat, sigma.Y.hat, init.method, npairs, n, d = NULL){
  if(init.method == "svd"){
    obj.svd <- svd(sigma.YX.hat,nu = npairs,nv = npairs)
    alpha.init <- obj.svd$u[ , 1:npairs, drop = F]
    beta.init <- obj.svd$v[ , 1:npairs, drop = F]
    }
  if(init.method == "uniform"){
    alpha.init <- matrix(1, q, npairs)
    beta.init <- matrix(1, p, npairs)
  }
  if(init.method == "random"){
    alpha.init <- matrix(rnorm(q * npairs), q, npairs)
    beta.init <- matrix(rnorm(p * npairs), p, npairs)
  }
  if(init.method == "sparse"){
    alpha.init <- matrix(0, q, npairs)
    beta.init <- matrix(0, p, npairs)
    if(missing(d)) d <- sqrt(n)
    thresh <- sort(abs(sigma.YX.hat), decreasing = T)[d]
    row.max <- apply(abs(sigma.YX.hat), 1, max)
    col.max <- apply(abs(sigma.YX.hat), 2, max)
    obj.svd <- svd(sigma.YX.hat[row.max > thresh, col.max > thresh])
    
    
    alpha1.init <- rep(0, q)
    beta1.init <- rep(0, p)
    alpha1.init[row.max > thresh] <- obj.svd$u[ , 1]
    beta1.init[col.max > thresh] <- obj.svd$v[ , 1]
    
    alpha.init[ , 1] <- alpha1.init
    beta.init[ , 1] <- beta1.init
  }
  alpha.scale <- diag(t(alpha.init) %*% sigma.Y.hat %*% alpha.init)[1:npairs, drop = F]
  alpha.init <- sweep(alpha.init[ , 1:npairs, drop = F], 2, sqrt(alpha.scale), "/")
  beta.scale <- diag(t(beta.init) %*% sigma.X.hat %*% beta.init)[1:npairs, drop = F]
  beta.init <- sweep(beta.init[ , 1:npairs, drop = F], 2, sqrt(beta.scale), "/")
  list(alpha.init = alpha.init, beta.init = beta.init)
}



#### The function init1 finds the initial value when npairs0 canonical pairs have been obtained. If init.method="sparse", 
#### only one pair of initial value will be returned. For other options of init.method, init1 returns npairs - npairs0 pairs
#### of initial values.

init1 <- function(sigma.YX.hat, sigma.X.hat, sigma.Y.hat, init.method, npairs, npairs0, alpha.current, beta.current, n=n, eps = 1e-4, d = NULL){
 
  alpha.init <- matrix(0, q, 1)
  beta.init <- matrix(0, p, 1)
  alpha.current <- as.matrix(alpha.current)
  beta.current <- as.matrix(beta.current)
  
   npairs0 <- ncol(alpha.current)
  
  if(init.method == "svd"){
    obj.svd <- svd(sigma.YX.hat)
    alpha.init <- obj.svd$u[ , npairs0 + 1, drop = F]
    beta.init[ , npairs0:npairs] <- obj.svd$v[ , npairs0 + 1, drop = F]
  }
  if(init.method == "uniform"){
    alpha.init <- matrix(1, q, 1)
    beta.init <- matrix(1, p, 1)
  }
  if(init.method == "random"){
    alpha.init <- matrix(rnorm(q * npairs), q, 1)
    beta.init <- matrix(rnorm(p * npairs), p, 1)
  }
  if(init.method == "sparse"){
    
    
    id.nz.alpha <- which(apply(abs(alpha.current), 1, sum) > eps)
    id.nz.beta <- which(apply(abs(beta.current), 1, sum) > eps)
    
    rho.tmp <- t(alpha.current) %*% sigma.YX.hat %*% beta.current
    
    sigma.YX.tmp <- sigma.YX.hat - sigma.Y.hat %*% alpha.current %*% rho.tmp %*% t(beta.current) %*% sigma.X.hat
    
    if(missing(d)) d <- sqrt(n)
    
    thresh <- sort(abs(sigma.YX.tmp), decreasing = T)[d]
    row.max <- apply(abs(sigma.YX.tmp), 1, max)
    col.max <- apply(abs(sigma.YX.tmp), 2, max)
    
    id.row <- unique(c(id.nz.alpha, which(row.max > thresh)))
    id.row <- sort(id.row, decreasing = FALSE)
    id.col <- unique(c(id.nz.beta, which(col.max > thresh)))
    id.col <- sort(id.col, decreasing = FALSE)
    
    sigma.tmp <- sigma.YX.tmp[id.row, id.col]  
    obj.svd <- svd(sigma.tmp)
    
    alpha.init[id.row] <- obj.svd$u[ , 1]
    beta.init[id.col] <- obj.svd$v[ , 1]
  }
  alpha.scale <- drop(t(alpha.init) %*% sigma.Y.hat %*% alpha.init)
  alpha.init <- alpha.init/sqrt(alpha.scale)
  beta.scale <- drop(t(beta.init) %*% sigma.X.hat %*% beta.init)
  beta.init <- beta.init/sqrt(beta.scale)
  
  list(alpha.init = alpha.init, beta.init = beta.init)
}


#### The function find.Omega is used by SCCA.solution. 
find.Omega <- function(sigma.YX.hat, npairs, alpha = NULL, beta = NULL,y = NULL, x = NULL){
  n <- nrow(y)
  if(npairs > 1){
    rho <- t(alpha) %*% sigma.YX.hat %*% beta
    omega <- diag(rep(1, n))-y %*% alpha %*% rho %*% t(beta) %*% t(x)/n
}else{
  omega <- diag(rep(1, n))
}
  omega
}


#### The function SCCA.solution is used by SCCA. 
SCCA.solution <- function(x, y, x.Omega, y.Omega, alpha0, beta0, lambda.alpha, lambda.beta, niter = 100, glmnet.alg = NULL, eps=1e-4){
  n <- nrow(x)
  p <- ncol(x)
  q <- ncol(y)
  
  for(i in 1:niter){
    x0 <- x.Omega %*% beta0
    
    m <- glmnet(y, x0, standardize = FALSE, intercept = FALSE, lambda = lambda.alpha)
    
    alpha1 <- coef(m, s = lambda.alpha)[-1]
    
    if(sum(abs(alpha1)) < eps){
      alpha0 <- rep(0, q)
      break}
    id.nz <- which(alpha1 != 0)
    alpha1.scale <- y[ , id.nz, drop = F] %*% alpha1[id.nz, drop = F]
    
    alpha1 <- alpha1/drop(sqrt(t(alpha1.scale) %*% alpha1.scale/(n - 1)))
    
    y0 <- y.Omega %*% alpha1
    

      m <- glmnet(x, y0, standardize = FALSE, intercept = FALSE, lambda = lambda.beta)

    beta1 <- coef(m, s = lambda.beta)[-1]

    if(sum(abs(beta1)) < eps){
      beta0 <- rep(0, p)
      break}
    id.nz <- which(beta1 != 0)
    beta1.scale <- x[ , id.nz, drop = F] %*% beta1[id.nz, drop = F]
    
    beta1 <- beta1/drop(sqrt(t(beta1.scale) %*% beta1.scale/(n - 1)))
    
    if(sum(abs(alpha1 - alpha0)) < eps&sum(abs(beta1 - beta0) < eps)) break
    alpha0 <- alpha1
    beta0 <- beta1
    
  }
  

  list(alpha = alpha0, beta = beta0, niter = i)
}


#The function cv.SCCA is the cross validation function for the first canonical pair.

cv.SCCA <- function(x, y, lambda.alpha, lambda.beta, nfolds=5, alpha.init, beta.init, eps = 1e-3, niter = 10, standardize = TRUE){
  
  n <- nrow(x)
  id.folds <- cut(seq(1:n), breaks = nfolds, labels = 1:nfolds)
  id.folds <- sample(id.folds, n, replace = FALSE)
  id.folds <- as.numeric(id.folds)
  rho <- matrix(0, length(lambda.alpha), length(lambda.beta))
    
  
  for(i.lambda in 1:length(lambda.alpha)){
    for(j.lambda in 1:length(lambda.beta)){
      for(i in 1:nfolds){
        
          obj <- SCCA(x[id.folds!=i, ], y[id.folds!=i, ], alpha.init = alpha.init, beta.init = beta.init, lambda.alpha = lambda[i.lambda], lambda.beta = lambda[j.lambda], eps = eps, niter = niter, standardize = standardize)
        
          rho[i.lambda, j.lambda] <- rho[i.lambda, j.lambda]+abs(cor(x[id.folds==i, ]%*%obj$beta, y[id.folds==i,] %*% obj$alpha))
      }
    }
  }
  rho[is.na(rho)] <- 0
  id.alpha.max<-max(which(apply(rho, 1, max)==max(rho)))
  id.beta.max<-max(which(apply(rho, 2, max)==max(rho)))
  rho[is.na(rho)] <- 0
  
  list(rho = rho, lambda = lambda, bestlambda.alpha = lambda.alpha[id.alpha.max], bestlambda.beta = lambda.beta[id.beta.max])
}


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