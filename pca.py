import numpy as np
import matplotlib.pyplot as plt

time = np.linspace(0,10,1025)
series = np.sin(2*time*np.pi) + np.random.normal(0,0.05,1025)

# plt.plot(time, series)

X1 = np.c_[series[0:1000], series[25:1025]]
X = np.concatenate((X1, np.c_[np.random.normal(0,0.15,1000), \
                              np.random.normal(0,0.15,1000)]), axis=0)

Y = np.concatenate((np.tile(1,1000), np.tile(2,1000)), axis=0)


# plt.plot(X, Y)

## eigen
C = np.cov(X, rowvar=False)
w,v = np.linalg.eig(C)

# print(R)
fig = plt.figure()
# ax = plt.axes(projection='2d')
plt.scatter(np.matmul(X,v)[:,0], np.matmul(X,v)[:,1],
           cmap='viridis', linewidth=0.5)