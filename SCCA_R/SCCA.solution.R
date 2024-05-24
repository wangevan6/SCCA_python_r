
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