import torch
import torch.nn as nn
from sklearn.datasets.samples_generator import make_blobs
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from random import shuffle
from math import sqrt

acceptable_float_error = 3
learning_rate = 0.01
epochs = 20
n_samples = 1000
batch_size = 1

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

class Net3(nn.Module):
	def __init__(self):
		super(Net3, self).__init__()
		self.fc1 = nn.Linear(2, 20)
		self.fc2 = nn.Linear(20, 20)
		self.fc3 = nn.Linear(20, 20)
		self.fc4 = nn.Linear(20, 2)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return F.log_softmax(x, dim=1)

class Net2(nn.Module):
	def __init__(self):
		super(Net2, self).__init__()
		self.fc1 = nn.Linear(2, 15)
		self.fc2 = nn.Linear(15, 15)
		self.fc3 = nn.Linear(15, 15)
		self.fc4 = nn.Linear(15, 15)
		self.fc5 = nn.Linear(15, 15)
		self.fc6 = nn.Linear(15, 2)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))
		x = F.relu(self.fc5(x))
		x = self.fc6(x)
		return F.log_softmax(x, dim=1)

class Net1(nn.Module):
	def __init__(self):
		super(Net1, self).__init__()
		self.fc1 = nn.Linear(2, 10)
		self.fc2 = nn.Linear(10, 10)
		self.fc3 = nn.Linear(10, 10)
		self.fc4 = nn.Linear(10, 10)
		self.fc5 = nn.Linear(10, 10)
		self.fc6 = nn.Linear(10, 10)
		self.fc7 = nn.Linear(10, 10)
		self.fc8 = nn.Linear(10, 10)
		self.fc9 = nn.Linear(10, 10)
		self.fc10 = nn.Linear(10, 10)
		self.fc11 = nn.Linear(10, 2)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))
		x = F.relu(self.fc5(x))
		x = F.relu(self.fc6(x))
		x = F.relu(self.fc7(x))
		x = F.relu(self.fc8(x))
		x = F.relu(self.fc9(x))
		x = F.relu(self.fc10(x))
		x = self.fc11(x)
		return F.log_softmax(x, dim=1)

class Net0(nn.Module):
	def __init__(self):
		super(Net0, self).__init__()
		self.fc1 = nn.Linear(2, 7)
		self.fc2 = nn.Linear(7, 7)
		self.fc3 = nn.Linear(7, 7)
		self.fc4 = nn.Linear(7, 7)
		self.fc5 = nn.Linear(7, 7)
		self.fc6 = nn.Linear(7, 7)
		self.fc7 = nn.Linear(7, 7)
		self.fc8 = nn.Linear(7, 7)
		self.fc9 = nn.Linear(7, 7)
		self.fc10 = nn.Linear(7, 7)
		self.fc11 = nn.Linear(7, 7)
		self.fc12 = nn.Linear(7, 7)
		self.fc13 = nn.Linear(7, 7)
		self.fc14 = nn.Linear(7, 7)
		self.fc15 = nn.Linear(7, 7)
		self.fc16 = nn.Linear(7, 7)
		self.fc17 = nn.Linear(7, 7)
		self.fc18 = nn.Linear(7, 7)
		self.fc19 = nn.Linear(7, 7)
		self.fc20 = nn.Linear(7, 7)
		self.fc21 = nn.Linear(7, 2)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))
		x = F.relu(self.fc5(x))
		x = F.relu(self.fc6(x))
		x = F.relu(self.fc7(x))
		x = F.relu(self.fc8(x))
		x = F.relu(self.fc9(x))
		x = F.relu(self.fc10(x))
		x = F.relu(self.fc11(x))
		x = F.relu(self.fc12(x))
		x = F.relu(self.fc13(x))
		x = F.relu(self.fc14(x))
		x = F.relu(self.fc15(x))
		x = F.relu(self.fc16(x))
		x = F.relu(self.fc17(x))
		x = F.relu(self.fc18(x))
		x = F.relu(self.fc19(x))
		x = F.relu(self.fc20(x))
		x = self.fc21(x)
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

def train(net, optimizer, criterion, train_loader, epoch, log_interval=1):
	net.train()

	train_loss = 0.0
	correct    = 0

	for batch_idx, (data, target) in enumerate(train_loader):
		optimizer.zero_grad()
		output = net(data)
		loss = F.nll_loss(output, target)
		train_loss += loss.item()
		pred = output.max(1, keepdim=True)[1]
		correct += pred.eq(target.view_as(pred)).sum().item()
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			epoch, batch_idx , len(train_loader.dataset),
			100. * batch_idx / len(train_loader), loss.item()))
	
	train_loss /= len(train_loader.dataset)
	train_accuracy   = correct/len(train_loader.dataset) 

	print('\nTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(train_loss, correct, 
		len(train_loader.dataset), 100. * correct / len(train_loader.dataset)))

def test(net, criterion, test_loader, epoch):
	net.eval()
	correct = 0
	test_loss = 0.0
	log_interval = 1

	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(test_loader):
			output = net(data)
			loss = F.nll_loss(output, target, size_average=False)
			test_loss += loss.item() # sum up batch loss
			pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()

			# log the training details on every log_interval (default=10)
			if batch_idx % log_interval == 0:
				print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx, len(test_loader.dataset),
				100. * batch_idx / len(test_loader), loss.item()))

	test_loss /= len(test_loader.dataset)
	test_accuracy   = correct/len(test_loader.dataset) 
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, 
		len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

def main(net):
	# generate the dataset, shuffle it
	dataset_data = generate_linear_dataset(n_samples=n_samples)
	shuffle(dataset_data)
	train_test_divide = int(len(dataset_data) * .8)

	# partition into train and test sets
	train_dataset = dataset_data[:train_test_divide]
	test_dataset  = dataset_data[train_test_divide:]

	train_loader = dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader  = dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

	optimizer = optim.Adam(net.parameters(), lr=learning_rate)
	criterion = nn.CrossEntropyLoss()

	for epoch in range(1, epochs + 1):
		train(net, optimizer, criterion, train_loader, epoch)
		test(net, criterion, test_loader, epoch)

if __name__ == '__main__':

	# run all 5 models
	models_num = 5
	models = [Net0(), Net1(), Net2(), Net3(), Net4(), Net4()]
	for model_idx in range(5):
		print('___Net{}___\n'.format(model_idx))
		main(models[model_idx])
		print('\n___________')

