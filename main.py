import torch
import torch.nn as nn
from sklearn.datasets.samples_generator import make_blobs
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from random import shuffle
from math import sqrt

acceptable_float_error = 1
learning_rate = 0.01
epochs = 10
n_samples = 10
batch_size = 1

class Net4(nn.Module):
	def __init__(self):
		super(Net4, self).__init__()
		self.fc0 = nn.Linear(1, 2)
		self.fc1 = nn.Linear(2, 30)
		self.fc2 = nn.Linear(30, 30)
		self.fc3 = nn.Linear(30, 2)
		self.fc4 = nn.Linear(2, 1)

	def forward(self, x):
		x = F.relu(self.fc0(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return F.log_softmax(x)

# generate a linear dataset with two centers (using sklearn's make_blobs)
# making a linear function to separate the two cluster of points possible
def generate_linear_dataset(n_samples=10, centers=2):
	X_Y, _ = make_blobs(n_samples=n_samples, centers=centers, n_features=2)
	
	X = torch.Tensor.float(torch.from_numpy(X_Y[:, 0]))
	y = torch.Tensor.float(torch.from_numpy(X_Y[:, 1]))

	return [x for x in zip(X, y)]

def train(net, optimizer, criterion, train_loader, epoch, log_interval=1):
	net.train()

	train_loss = 0.0
	correct    = 0

	for batch_idx, data in enumerate(train_loader, 0):
		# get the inputs
		inputs, target = data

		# zero the parameter gradients
		optimizer.zero_grad()

		# foraward, backward, optimize
		output = net(inputs)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()

		correct += torch.le(torch.abs(output - target), acceptable_float_error).sum().item()

		train_loss += loss.item()
		# log the training details on every log_interval (default=10)
		if batch_idx % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			epoch, batch_idx, len(train_loader.dataset),
			100. * batch_idx / len(train_loader), loss.item()))

	print('Training loss: {:.6f} Accuracy: {}/{}\n'.format(train_loss/len(train_loader.dataset), correct, len(train_loader.dataset) ))

def test(net, criterion, test_loader):
	net.eval()
	correct = 0
	test_loss = 0.0

	with torch.no_grad():
		for data in test_loader:
			inputs, label = data
			predicted = net(inputs)
			test_loss += criterion(predicted, label).item() # sum up batch loss

			correct += torch.le(torch.abs(predicted - label) , acceptable_float_error).sum().item()
	
	test_loss /= len(test_loader.dataset)
	test_accuracy   = correct/len(test_loader.dataset) 
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, 
		len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

def main():
	# generate the dataset, shuffle it
	dataset_data = generate_linear_dataset(n_samples=n_samples)
	shuffle(dataset_data)

	train_test_divide = int(len(dataset_data) * .8)

	# partition into train and test sets
	train_dataset = dataset_data[:train_test_divide]
	test_dataset  = dataset_data[train_test_divide:]

	train_loader = dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader  = dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

	net = Net4()
	print(len(train_dataset), len(test_dataset))
	optimizer = optim.Adam(net.parameters(), lr=learning_rate)
	criterion = nn.MSELoss()

	for epoch in range(1, epochs + 1):
		train(net, optimizer, criterion, train_loader, epoch)
		test(net, criterion, test_loader)

if __name__ == '__main__':
    main()