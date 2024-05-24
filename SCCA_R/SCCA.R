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
