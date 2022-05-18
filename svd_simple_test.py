import numpy as np

X = np.random.rand(5,3)
print(X)

U, S, V = np.linalg.svd(X, full_matrices=True) #full_matrices = full SVD
print('U', U, '\n', 'S', S, '\n', 'V', V,)

print('========== Economy mode =========')

U_hat, S_hat, V_hat = np.linalg.svd(X, full_matrices=False) #full_matrices = full SVD
print('U_hat', U_hat, '\n', 'S_hat', S_hat, '\n', 'V_hat', V_hat) #economy SVD - returns m columns