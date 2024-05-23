
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
