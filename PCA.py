import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,decomposition,manifold

# def load_data():
#     iris=datasets.load_iris()
#     return iris.data,iris.target
#
# def test_PCA(*data):
#     X,y=data
#     pca=decomposition.PCA(n_components=None)
#     pca.fit(X)
#     print('explained variance ratio : %s'%str(pca.explained_variance_ratio_))
#
# X,y=load_data()
# test_PCA(X,y)
#
# def plot_PCA(*data):
#     X,y=data
#     pca=decomposition.PCA(n_components=2)
#     pca.fit(X)
#     X_r=pca.transform(X)
#
#     fig=plt.figure()
#     ax=fig.add_subplot(1,1,1)
#     colors=((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0,0.5,0.5),(0.5,0,0.5),(0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),(0.5,0.3,0.2))
#     for label,color in zip(np.unique(y),colors):
#         position=y==label
#         ax.scatter(X_r[position,0],X_r[position,1],label="target=%d"%label,color=color)
#     ax.set_xlabel("X[0]")
#     ax.set_ylabel("Y[0]")
#     ax.legend(loc="best")
#     ax.set_title("PCA")
#     plt.show()
# plot_PCA(X,y)

# pca = decomposition.PCA(n_components=None)
# X_pca = model_pca.fit(iris_X).transform(iris_X)