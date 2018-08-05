from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np
from numpy.random  import uniform
from sklearn.cluster import KMeans
import torch

low=-1.0
high=1.1
n_samples=1000

# create N samples, uniformly distributed between low and high
x_samples    = uniform(low=low, high=high, size=(n_samples,))
y_samples    = uniform(low=low, high=high, size=(n_samples,))
points = np.array([list(x) for x in zip(x_samples, y_samples)])

#linear function y=mx + b
m = .3
b = 0
f = lambda x: x * m + b

# The labels for the clusters (positive integers)
cluster1_label = 0
cluster2_label = 1

cluster1_labeled_points = []
cluster2_labeled_points = []

cluster1_x = []
cluster1_y = []

cluster2_x = []
cluster2_y = []

for point in points:
	x, y = point
	diff = y - f(x)

	if diff > 0:
		point_tensor_cluster1 = (torch.from_numpy(point).float(), cluster1_label)
		cluster1_labeled_points.append(point_tensor_cluster1)
		cluster1_x.append(x)
		cluster1_y.append(y)
	else:
		point_tensor_cluster2 = (torch.from_numpy(point).float(), cluster2_label)
		cluster2_labeled_points.append(point_tensor_cluster2)
		cluster2_x.append(x)
		cluster2_y.append(y)

resulting_labeled_points = cluster1_labeled_points + cluster2_labeled_points

plt.scatter(cluster1_x, cluster1_y, c='blue')
plt.scatter(cluster2_x, cluster2_y, c='red')
plt.title('Uniform Linear Dataset')
plt.show()