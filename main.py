import torch
import torch.nn as nn
from sklearn.datasets.samples_generator import make_blobs
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from random import shuffle
from utils import plot_boundry, generate_linear_dataset, \
generate_sample_linear_dataset, generate_uniform_linear_dataset, \
plot_loss, generate_uniform_circular_dataset, generate_uniform_parabolic_dataset, \
generate_uniform_stripes_dataset
from numpy.random import uniform

#0.005 is current best
learning_rate = 0.003
epochs = 100
n_samples = 20000
batch_size = 256

SHOW_TRAIN_LOGS = False
SHOW_TEST_LOGS  = False

class Net5(nn.Module):
	def __init__(self):
		super(Net5, self).__init__()
		self.id  = 5
		self.fc1 = nn.Linear(2, 11)
		self.fc2 = nn.Linear(11, 11)
		self.fc3 = nn.Linear(11, 11)
		self.fc4 = nn.Linear(11, 11)
		self.fc5 = nn.Linear(11, 2)

	def forward(self, x):
		x = F.elu(self.fc1(x))
		x = (self.fc2(x))
		x = (self.fc3(x))
		x = (self.fc4(x))
		x = self.fc5(x)
		return F.log_softmax(x, dim=1)


class Net4(nn.Module):
	def __init__(self):
		super(Net4, self).__init__()
		self.id  = 4
		self.fc1 = nn.Linear(2, 30)
		self.fc2 = nn.Linear(30, 30)
		self.fc3 = nn.Linear(30, 2)

	def forward(self, x):
		x = F.elu(self.fc1(x))
		x = F.elu(self.fc2(x))
		x = self.fc3(x)
		return F.log_softmax(x, dim=1)

class Net3(nn.Module):
	def __init__(self):
		super(Net3, self).__init__()
		self.id  = 3
		self.fc1 = nn.Linear(2, 20)
		self.fc2 = nn.Linear(20, 20)
		self.fc3 = nn.Linear(20, 20)
		self.fc4 = nn.Linear(20, 2)

	def forward(self, x):
		x = F.elu(self.fc1(x))
		x = F.elu(self.fc2(x))
		x = F.elu(self.fc3(x))
		x = self.fc4(x)
		return F.log_softmax(x, dim=1)

class Net2(nn.Module):
	def __init__(self):
		super(Net2, self).__init__()
		self.id  = 2
		self.fc1 = nn.Linear(2, 15)
		self.fc2 = nn.Linear(15, 15)
		self.fc3 = nn.Linear(15, 15)
		self.fc4 = nn.Linear(15, 15)
		self.fc5 = nn.Linear(15, 15)
		self.fc6 = nn.Linear(15, 2)

	def forward(self, x):
		x = F.elu(self.fc1(x))
		x = F.elu(self.fc2(x))
		x = F.elu(self.fc3(x))
		x = F.elu(self.fc4(x))
		x = F.elu(self.fc5(x))
		x = self.fc6(x)
		return F.log_softmax(x, dim=1)

class Net1(nn.Module):
	def __init__(self):
		super(Net1, self).__init__()
		self.id  = 1
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
		x = F.elu(self.fc1(x))
		x = F.elu(self.fc2(x))
		x = F.elu(self.fc3(x))
		x = F.elu(self.fc4(x))
		x = F.elu(self.fc5(x))
		x = F.elu(self.fc6(x))
		x = F.elu(self.fc7(x))
		x = F.elu(self.fc8(x))
		x = F.elu(self.fc9(x))
		x = F.elu(self.fc10(x))
		x = self.fc11(x)
		return F.log_softmax(x, dim=1)

class Net0(nn.Module):
	def __init__(self):
		super(Net0, self).__init__()
		self.id  = 0
		self.fc1 = nn.Linear(2, 9)
		self.fc2 = nn.Linear(9, 9)
		self.fc3 = nn.Linear(9, 9)
		self.fc4 = nn.Linear(9, 9)
		self.fc5 = nn.Linear(9, 9)
		self.fc6 = nn.Linear(9, 9)
		self.fc7 = nn.Linear(9, 9)
		self.fc8 = nn.Linear(9, 9)
		self.fc9 = nn.Linear(9, 9)
		self.fc10 = nn.Linear(9, 9)
		self.fc11 = nn.Linear(9, 9)
		self.fc12 = nn.Linear(9, 9)
		self.fc13 = nn.Linear(9, 9)
		self.fc14 = nn.Linear(9, 9)
		self.fc15 = nn.Linear(9, 9)
		self.fc16 = nn.Linear(9, 9)
		self.fc17 = nn.Linear(9, 9)
		self.fc18 = nn.Linear(9, 9)
		self.fc19 = nn.Linear(9, 9)
		self.fc20 = nn.Linear(9, 9)
		self.fc21 = nn.Linear(9, 2)

	def forward(self, x):
		x = F.elu(self.fc1(x))
		x = F.elu(self.fc2(x))
		x = F.elu(self.fc3(x))
		x = F.elu(self.fc4(x))
		x = F.elu(self.fc5(x))
		x = F.elu(self.fc6(x))
		x = F.elu(self.fc7(x))
		x = F.elu(self.fc8(x))
		x = F.elu(self.fc9(x))
		x = F.elu(self.fc10(x))
		x = F.elu(self.fc11(x))
		x = F.elu(self.fc12(x))
		x = F.elu(self.fc13(x))
		x = F.elu(self.fc14(x))
		x = F.elu(self.fc15(x))
		x = F.elu(self.fc16(x))
		x = F.elu(self.fc17(x))
		x = F.elu(self.fc18(x))
		x = F.elu(self.fc19(x))
		x = F.elu(self.fc20(x))
		x = self.fc21(x)
		return F.log_softmax(x, dim=1)

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
		if batch_idx % log_interval == 0 and SHOW_TRAIN_LOGS:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			epoch, batch_idx , len(train_loader.dataset),
			100. * batch_idx / len(train_loader), loss.item()))
	
	train_loss /= len(train_loader.dataset)
	train_accuracy   = 100. * correct / len(train_loader.dataset)

	if SHOW_TRAIN_LOGS:
		print('\nTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(train_loss, correct, 
			len(train_loader.dataset), train_accuracy))

	return train_accuracy, train_loss

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
			if batch_idx % log_interval == 0 and SHOW_TEST_LOGS:
				print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx, len(test_loader.dataset),
				100. * batch_idx / len(test_loader), loss.item()))

	test_loss /= len(test_loader.dataset)
	test_accuracy   = 100. * correct / len(test_loader.dataset)

	if SHOW_TEST_LOGS:
		print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, 
			len(test_loader.dataset), test_accuracy))

	return test_accuracy, test_loss

def main(net, train_loader, test_loader, sample_loader):
	optimizer = optim.Adam(net.parameters(), lr=learning_rate)
	criterion = nn.CrossEntropyLoss()
	
	training_accuracy = 0
	test_accuracy     = 0

	training_loss_array = []
	test_loss_array		= []

	for epoch in range(1, epochs + 1):
		training_accuracy, training_loss = train(net, optimizer, criterion, train_loader, epoch)
		test_accuracy, test_loss     = test(net, criterion, test_loader, epoch)
		
		training_loss_array.append( training_loss )
		test_loss_array.append( test_loss )

	print('Train Accuracy: {} \nTest Accuracy: {}'.format(training_accuracy, test_accuracy))
	plot_loss(net, training_loss_array, 'training', test_loss_array, 'validation')
	plot_boundry(net, N=n_samples, sample_loader=sample_loader,low=-10, high=10.1)


if __name__ == '__main__':
	
	# generate the dataset, shuffle it
	train_dataset = generate_uniform_stripes_dataset(n_samples=n_samples, low=-3.0, high=3.1, slope=.1, plot_db=True)
	validation_dataset  = generate_uniform_stripes_dataset(n_samples=n_samples, low=-3.0, high=3.1, slope=.1, plot_db=True)

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	validation_loader  = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

	# run all 5 models
	models = [Net0(), Net1(), Net2(), Net3(), Net4(), Net5()]
	models_num = len(models)
	
	for model_idx in range(models_num):
		repeat_times = 1
		for i in range(1, repeat_times + 1):
			print('\n___Net{}___{}/{}\n'.format(model_idx, i, repeat_times))
			main(models[model_idx], train_loader, validation_loader, generate_sample_linear_dataset())
			print('\n___________')

