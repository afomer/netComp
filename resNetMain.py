'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import argparse
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
learning_rate = 0.007
momentum = 0.5
weight_decay = 0.5
epochs = 20
n_samples = 1000
batch_size = 256

# training parameters
SHOW_TRAIN_LOGS = True
SHOW_TEST_LOGS  = True

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.elu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.elu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.elu(self.bn1(self.conv1(x)))
        out = F.elu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.elu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.id = 'Res'
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.elu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])



def train(net, optimizer, criterion, train_loader, epoch, log_interval=1):
    net.train()

    train_loss = 0.0
    correct    = 0

    for batch_idx, (raw_data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = torch.zeros(1,1,25,25)
        #data[0][0][0][0] = raw_data[0][0]
        #data[0][0][0][1] = raw_data[0][1]
        #print(data.size(), raw_data.size(), data[0][0][0][0], raw_data)
        #data[0][0] = raw_data[0][0]
        #data[1][0] = raw_data[0][0]
        #data[2][0] = raw_data[0][0]

        output = net(raw_data)

        loss = torch.exp(F.nll_loss(output, target))
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

def test(net, criterion, test_loader, epoch, log_interval = 1):
    net.eval()
    correct = 0
    test_loss = 0.0


    with torch.no_grad():
        for batch_idx, (raw_data, target) in enumerate(test_loader):
            #data = torch.zeros(1,1,2,2)
            #data[0][0][0][0] = raw_data[0][0]
            #data[0][0][0][1] = raw_data[0][1]
            output = net(raw_data)
            loss = torch.exp( F.nll_loss(output, target) )
            test_loss += loss.item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            # log the training details on every log_interval (default=10)
            if batch_idx % log_interval == 0 and SHOW_TEST_LOGS:
                print('Test Epoch: {} [{}/{} ({:.6f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(test_loader.dataset),
                100. * batch_idx / len(test_loader), loss.item()))

    test_loss /= len(test_loader.dataset)
    test_accuracy   = 100. * correct / len(test_loader.dataset)

    if SHOW_TEST_LOGS:
        print('\nTest set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, 
            len(test_loader.dataset), test_accuracy))

    return test_accuracy, test_loss

def main(net, train_loader, test_loader, sample_loader):
    
    optimizer = torch.optim.SGD(net.parameters(),learning_rate,
                                momentum=momentum,
                                weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    training_accuracy = 0
    test_accuracy     = 0

    training_loss_array = []
    test_loss_array     = []

    for epoch in range(1, epochs + 1):
        training_accuracy, training_loss = train(net, optimizer, criterion, train_loader, epoch)
        test_accuracy, test_loss     = test(net, criterion, test_loader, epoch)
        
        training_loss_array.append( training_loss )
        test_loss_array.append( test_loss )

    print('Train Accuracy: {} \nTest Accuracy: {}'.format(training_accuracy, test_accuracy))
    plot_loss(net, training_loss_array, 'training', test_loss_array, 'validation')
    plot_boundry(net, N=n_samples, sample_loader=sample_loader,low=-10, high=10.1)

def test2():
    net = ResNet18()
    torch.zeros(32, 32)
    y = net(torch.randn(1,1,32,32))
    print(y.size(), torch.randn(1,3,2,2))


if __name__ == '__main__':
    
    # generate the dataset, shuffle it
    train_dataset = []
    validation_dataset  = []
    
    for sample in generate_uniform_stripes_dataset(n_samples=n_samples, low=-3.0, high=3.1, slope=.1, plot_db=False):
        label = sample[1]

        new_sample = [[ [sample[0][0].item(), sample[0][1].item()] ]]
        padded_sample = F.pad(torch.Tensor(new_sample), (10, 13, 14, 10), 'constant', 0)

        modified_sample = (padded_sample, label)
        train_dataset.append(modified_sample)

    for sample in generate_uniform_stripes_dataset(n_samples=n_samples, low=-10.0, high=10.1, slope=.1, plot_db=False):
        label = sample[1]

        new_sample = [[ [sample[0][0].item(), sample[0][1].item()] ]]
        padded_sample = F.pad(torch.Tensor(new_sample), (10, 13, 14, 10), 'constant', 0)

        modified_sample = (padded_sample, label)
        validation_dataset.append(modified_sample)

    #print('train_dataset: ', train_dataset[0][0][0].size(), train_dataset[0][0][0])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader  = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    # run all 5 models
    models = [ResNet18()]
    models_num = len(models)
    
    for model_idx in range(models_num):
        repeat_times = 1
        for i in range(1, repeat_times + 1):
            print('\n___ResNet{}___{}/{}\n'.format(model_idx, i, repeat_times))
            main(models[model_idx], train_loader, validation_loader, generate_sample_linear_dataset())
            print('\n___________')

