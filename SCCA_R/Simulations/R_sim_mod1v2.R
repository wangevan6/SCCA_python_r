library(glmnet)
library(Matrix)

p <- 300
q <- 300 
n <- 500

# Model 1: Identity Covariances
sigma.X <- diag(p)
sigma.Y <- diag(q)

# True Canonical Weight Vectors
theta <- matrix(0, q, 2)
eta <- matrix(0, p, 2)

# Indices for non-zero entries
id <- c(1, 6, 11, 16, 21)

# Coefficients for the first canonical variable
theta[id, 1] <- c(-2, -1, -1, 2, 2)
# Coefficients for the second canonical variable
theta[id, 2] <- c(0, 0, 0, 1, 1)

# Normalizing the true canonical vectors
for (i in 1:2) {
  theta[, i] <- theta[, i] / as.vector(sqrt(t(theta[, i]) %*% sigma.Y %*% theta[, i]))
}

eta <- theta

# Cross-covariance matrix
sigma.YX <- 0.9 * sigma.Y %*% theta[, 1] %*% t(eta[, 1]) %*% sigma.X
          + 0.8 * sigma.Y %*% theta[, 2] %*% t(eta[, 2]) %*% sigma.X

# Full covariance matrix
sigma <- matrix(0, p + q, p + q)
sigma[1:q, 1:q] <- sigma.Y
sigma[(q + 1):(p + q), (q + 1):(p + q)] <- sigma.X
sigma[1:q, (q + 1):(p + q)] <- sigma.YX
sigma[(q + 1):(p + q), 1:q] <- t(sigma.YX)

# Eigen decomposition for the square root of the covariance matrix
sigma.eigen <- eigen(sigma)
sigma.sqrt <- sigma.eigen$vectors %*% sqrt(diag(sigma.eigen$values)) %*% t(sigma.eigen$vectors)

set.seed(4399)

################################################################################
#Check Point 1 < ----
# To be filled ....
################################################################################

Z <- matrix(rnorm(2*n*(p+q)), nrow = 2*n)
Z <- Z%*%sigma.sqrt


Y <- Z[ , 1:q]
X <- Z[ , (q + 1):(p + q)]

id.train <- sample(1:(2*n), n, replace = FALSE)

X.train <- X[id.train, ]
Y.train <- Y[id.train, ]
X.tune <- X[-id.train, ]
Y.tune <- Y[-id.train, ]


sigma.Y.hat <- cov(Y.train)
#print(sigma.Y.hat)
sigma.X.hat <- cov(X.train) 
#print(sigma.X.hat)
sigma.YX.hat <- cov(Y.train, X.train)
#print(sigma.YX.hat)


X.train <- sweep(X.train, 2, apply(X.train, 2, mean))
Y.train <- sweep(Y.train, 2, apply(Y.train, 2, mean))

lambda <- (1:20)/100
init.opt <- c("sparse", "svd")

rho.lambda <- matrix(0, length(lambda), 2)

dim(rho.lambda)

for(i.init in 1:length(init.opt)){
  
  obj.init <- init0(sigma.YX.hat = sigma.YX.hat, sigma.X.hat = sigma.X.hat, sigma.Y.hat = sigma.Y.hat, 
                    npairs = 1, n = n, init.method = init.opt[i.init])
  alpha.init <- obj.init$alpha.init
  beta.init <- obj.init$beta.init
  
  for(i.lambda in 1:length(lambda)){
    lambda.alpha <- lambda[i.lambda]
    lambda.beta <- lambda[i.lambda]
    
    obj <- SCCA(x = X.train, y = Y.train, lambda.alpha = lambda.alpha, lambda.beta = lambda.beta,
                npairs = 1, alpha.init = alpha.init, beta.init = beta.init)
    rho.lambda[i.lambda, i.init] <- cor(X.tune %*% obj$beta[ , 1], Y.tune %*% obj$alpha[ , 1])^2
    #if(rho.lambda[i.lambda, i.init] < 1e-6) break
  }
}

################################################################################
#Check Point 2 < ----
# To be filled ....
################################################################################

id.best <- which(rho.lambda == max(rho.lambda), arr.ind = TRUE)
bestlambda <- lambda[id.best[1]]
bestinit <- init.opt[id.best[2]]

obj <- SCCA(x = X.train, y = Y.train, lambda.alpha = bestlambda, lambda.beta = bestlambda, 
            npairs = 1, init.method = bestinit)


alpha.hat <- obj$alpha
beta.hat <- obj$beta



sum((alpha.hat%*%solve(t(alpha.hat)%*%alpha.hat)%*%t(alpha.hat) 
     - theta%*%solve(t(theta)%*%theta)%*%t(theta))^2)^0.5

sum((beta.hat%*%solve(t(beta.hat)%*%beta.hat)%*%t(beta.hat)
     - eta%*%solve(t(eta)%*%eta)%*%t(eta))^2)^0.5 


