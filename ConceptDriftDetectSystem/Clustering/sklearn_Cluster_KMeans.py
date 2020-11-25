'''
In practice, the k-means algorithm is very fast (one of the fastest clustering algorithms available), 
but it falls in local minima.
That’s why it can be useful to restart it several times.
'''

from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print(kmeans.labels_)

kmeans.predict([[0, 0], [4, 4]])

kmeans.cluster_centers_
