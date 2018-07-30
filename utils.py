from numpy.random import uniform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from pandas import DataFrame
from matplotlib import pyplot
from sklearn.datasets.samples_generator import make_blobs

class Net4(nn.Module):
	def __init__(self):
		super(Net4, self).__init__()
		self.fc1 = nn.Linear(2, 30)
		self.fc2 = nn.Linear(30, 30)
		self.fc3 = nn.Linear(30, 2)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return F.log_softmax(x, dim=1)

# generate a linear dataset with two centers (using sklearn's make_blobs)
# making a linear function to separate the two cluster of points possible
def generate_linear_dataset(n_samples=10, centers=2):
	X_Y_values, label_values = make_blobs(n_samples=n_samples, centers=centers, n_features=2,
		cluster_std=1.2, shuffle=True)
	
	X = torch.from_numpy(X_Y_values[:, 0])
	Y = torch.from_numpy(X_Y_values[:, 1])
	XY = torch.Tensor([x for x in zip(X, Y)])
	labels = torch.from_numpy(label_values)

	return [ x for x in zip(XY, labels)]

# Generate a plot of the decision boundry by creating 1000 samples, running the NN
# on them, then color samples them based on NN's classification
def plot_boundry(NN, N=1000, low=-2, high=2, sample_loader=None):

	if sample_loader is not None:
		# create N samples, uniformly distributed between low and high
		x_samples    = uniform(low=low, high=high, size=(N,))
		y_samples    = uniform(low=low, high=high, size=(N,))
		
		x_tensors = torch.from_numpy(x_samples).float()
		y_tensors = torch.from_numpy(y_samples).float()

		samples = torch.Tensor([ x for x in zip(x_tensors, y_tensors) ])

		sample_loader = torch.utils.data.DataLoader(samples)

	# feed it to the neural net
	labels = []
	for x in sample_loader:
		y = NN(x)
		class_1, class_2 = y[0][0], y[0][1]
		label = 1 if  class_1 > class_2 else 0
		labels.append(label)

	# scatter plot, dots colored by class value
	df = DataFrame(dict(x=x_samples, y=y_samples, label=labels))
	colors = {0:'red', 1:'blue'}
	fig, ax = pyplot.subplots()
	grouped = df.groupby('label')
	ax.set_title('Net{} - decision boundry'.format(str(NN.id)))

	for key, group in grouped:
	    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
	pyplot.show()