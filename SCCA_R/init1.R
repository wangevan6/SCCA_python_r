
#### The function init1 finds the initial value when npairs0 canonical pairs have been obtained. If init.method="sparse", 
#### only one pair of initial value will be returned. For other options of init.method, init1 returns npairs - npairs0 pairs
#### of initial values.

init1 <- function(sigma.YX.hat, sigma.X.hat, sigma.Y.hat, init.method, npairs, npairs0, alpha.current, beta.current, n=n, eps = 1e-4, d = NULL){
  p=ncol(sigma.X.hat)
  q=ncol(sigma.Y.hat)
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
