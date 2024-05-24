
#### The function init0 finds the initial value when no canonical pairs have been obtained. If init.method="sparse", 
#### only one pair of initial value will be returned. For other options of init.method, the number of pairs of initial
#### values can be specified with the argument npairs.
init0 <- function(sigma.YX.hat, sigma.X.hat, sigma.Y.hat, init.method, npairs, n, d = NULL){
  if(init.method == "svd"){
    obj.svd <- svd(sigma.YX.hat,nu = npairs,nv = npairs)
    alpha.init <- obj.svd$u[ , 1:npairs, drop = FALSE]
    beta.init <- obj.svd$v[ , 1:npairs, drop = FALSE]
  }
  p=ncol(sigma.X.hat)
  q=ncol(sigma.Y.hat)
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










