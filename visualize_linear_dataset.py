from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np
from numpy.random  import uniform
from sklearn.cluster import KMeans

low=-1.0
high=1.1
n_samples=1000

# create N samples, uniformly distributed between low and high
x_samples    = uniform(low=low, high=high, size=(n_samples,))
y_samples    = uniform(low=low, high=high, size=(n_samples,))

points = np.array([list(x) for x in zip(x_samples, y_samples)])

# create kmeans object
kmeans = KMeans(n_clusters=2)

# fit kmeans object to data
kmeans.fit(points)

# cluster the points
points_clusters = kmeans.fit_predict(points)

cluster_1_x_axis = points[points_clusters == 0,0]
cluster_1_y_axis = points[points_clusters == 0,1]

cluster_2_x_axis = points[points_clusters == 1,0]
cluster_2_y_axis = points[points_clusters == 1,1]

plt.scatter(cluster_1_x_axis, cluster_1_y_axis, c='red')
plt.scatter(cluster_2_x_axis, cluster_2_y_axis, c='blue')

plt.show()