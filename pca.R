time = seq(0,9,len=1025)
series = sin(2*time*pi) + rnorm(mean = 0, sd = 0.05, n=1025)

X = cbind(series[1:1000], series[26:1025])
X = rbind(X, cbind(rnorm(mean=0, sd=0.15, n=1000), rnorm(mean=0, sd=0.15, n=1000)))

Y = c(rep(1,1000), rep(2,1000))

par(mfrow=c(2,1))
plot(X, col=Y, pch=20, cex=2)

C = cov(X)
R = eigen((C))

plot(X %*% R$vectors, col=Y)