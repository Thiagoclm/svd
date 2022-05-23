from cv2 import normalize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, plot_confusion_matrix

seed = np.random.RandomState(42)

# Create synthetic dataset
X, y = make_classification(n_samples=10000, 
    n_features=30, 
    n_informative=10,
    n_redundant=10, 
    n_classes= 2,
    weights=[0.2,0.8], # 20% of the targets will be 0, 80% will be 1. default is 50/50
    flip_y=0.1,
    random_state=seed)

# Modelling original matrix X
print('Shape of X: ', np.shape(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=seed)

clf = RandomForestClassifier(random_state=seed)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print('========== Result model applied: X without SVD ==========')
print(classification_report(y_test, y_pred))
plot_confusion_matrix(clf, X_test, y_test, normalize='true')
plt.show()

# Applying SVD 
U, Sigma, V_T = np.linalg.svd(X, full_matrices = False) #economy
Sigma = np.diag(Sigma)

# Plots
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.semilogy(np.diag(Sigma))
plt.title('Singular Values (Sigma)')

plt.subplot(1,2,2)
plt.semilogy(np.cumsum(np.diag(Sigma))/np.sum(np.diag(Sigma)))
plt.title('Singular Values (Cumulative Sum)')
plt.show()

# Modelling matrix X after SVD

r = 20 # number of selected features
X_hat = U[:,:r] @ Sigma[:r,:r] @ V_T[:r,:] # @ = np.matmul
print('Shape of X after SVD: ', np.shape(X_hat))

# Applying MI to select features

mi = mutual_info_classif(X, y, random_state=seed)
df_mi = pd.DataFrame(mi, columns=['MI'])
df_mi.reset_index(inplace=True)

df_mi = df_mi.sort_values(by='MI', ascending=False)
print(df_mi)

columns_selected = df_mi.iloc[:,1].head(r)
indexes = columns_selected.index

X_selected = pd.DataFrame(X).iloc[:,indexes]
print(X_selected.shape)


# Modelling smaller dataset

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size = 0.30, random_state=seed)

clf_svd = RandomForestClassifier(random_state=seed)

clf_svd.fit(X_train, y_train)

y_pred = clf_svd.predict(X_test)

print('========== Result model applied: X after SVD ==========')
print(classification_report(y_test, y_pred))
plot_confusion_matrix(clf_svd, X_test, y_test, normalize='true')
plt.show()