import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.utils import accuracy
from src.batch_norm import Batch_norm_2d

class Fruit_conv_net(nn.Module):
    def __init__(self):
        super(Fruit_conv_net, self).__init__()
        self.number_of_filters_0 = 3 # cannot be changed
        self.number_of_filters_1 = 15 
        self.number_of_filters_2 = 30
        
        self.number_of_neurons_in_fc_0 = 2000
        self.number_of_neurons_in_fc_1 = 1000
        self.number_of_neurons_in_fc_2 = 95 # cannot be changed
        
        self.conv_1_1 = nn.Conv2d(self.number_of_filters_0, 
                                  self.number_of_filters_1, 
                                  5, 
                                  padding = 2, 
                                  bias=False)
        self.bn_1 = Batch_norm_2d(self.number_of_filters_1)
        self.pool_1 = nn.MaxPool2d(2)
        self.conv_1_2 = nn.Conv2d(self.number_of_filters_1, 
                                  self.number_of_filters_2, 
                                  5, 
                                  padding = 2, 
                                  bias=False)
        self.bn_2 = Batch_norm_2d(self.number_of_filters_2)
        self.pool_2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(self.number_of_filters_2 * 25 * 25, self.number_of_neurons_in_fc_0)
        self.fc2 = nn.Linear(self.number_of_neurons_in_fc_0, self.number_of_neurons_in_fc_1)
        self.fc3 = nn.Linear(self.number_of_neurons_in_fc_1, self.number_of_neurons_in_fc_1)
        
    def initialize_weights(self):
        nn.init.xavier_uniform_(self.conv_1_1.weight)
        nn.init.xavier_uniform_(self.conv_1_2.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        
    def forward(self, x):
        x = F.relu(self.pool_1(self.bn_1(self.conv_1_1(x))))
        x = F.relu(self.pool_2(self.bn_2(self.conv_1_2(x))))
        x = x.view(-1, self.number_of_filters_2 * 25 * 25)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def epoch_training(network, optimizer, criterion, train_set_loader, validation_set_loader):
    network.train()
    for i, data in enumerate(train_set_loader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        outputs = network(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    network.eval()
    
def training(network, train_set_loader, validation_set_loader, number_of_epochs = 10, optimizer=None):
    print('Training started')
    if optimizer is None:
        optimizer = optim.Adam(network.parameters(), lr=0.0001, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(number_of_epochs):
        epoch_training(network, optimizer, criterion, train_set_loader, validation_set_loader)
        print('Epoch ' + str(epoch) + ': Accuracy of the network:' + str(100*accuracy(network, validation_set_loader)) + '%')
    print('Finished Training')
    
def train_to_threshold(network, train_set_loader, validation_set_loader, threshold = 0.97, max_number_of_epochs=15, optimizer=None):
    print('Training started')
    history_of_accuracy = []
    if optimizer is None:
        optimizer = optim.Adam(network.parameters(), lr=0.0001, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(max_number_of_epochs):
        epoch_training(network, optimizer, criterion, train_set_loader, validation_set_loader)
        network_accuracy = accuracy(network, validation_set_loader)
        history_of_accuracy.append(network_accuracy)
        print('Epoch ' + str(epoch) + ': Accuracy of the network:' + str(100*network_accuracy) + '%')
        if(network_accuracy > threshold):
            break
    print('Finished Training after ' + str(epoch) + ' epochs')
    return history_of_accuracy

def get_gradients_of_image(network, images, labels, optimizer=None, criterion=None):
    if optimizer is None:
        optimizer = optim.Adam(network.parameters(), lr=0.0001, weight_decay=1e-2)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    images_grad = torch.autograd.Variable(images, requires_grad=True)
    optimizer.zero_grad()
    outputs = network(images_grad.cuda())
    loss = criterion(outputs.cpu(), labels)
    loss.backward()
    return images_grad.grad

def get_loss_of_original(network, images, labels, optimizer=None, criterion=None):
    if optimizer is None:
        optimizer = optim.Adam(network.parameters(), lr=0.0001, weight_decay=1e-2)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    outputs = network(images.cuda())
    return criterion(outputs, labels.cuda()).cpu()

def get_losses_of_cutouts(network, multiplied_images, labels, optimizer=None, criterion=None):
    if optimizer is None:
        optimizer = optim.Adam(network.parameters(), lr=0.0001, weight_decay=1e-2)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    batch_size = 100
    aggregate_losses = None
    labels_mult = labels.view(1,1).float().matmul(torch.ones(batch_size).view(1,batch_size)).view(batch_size).long().cuda()
    dataset = torch.utils.data.TensorDataset(multiplied_images)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            outputs = network(data[0].cuda())
            data[0].cpu()
            loss = criterion(outputs, labels_mult)
            if aggregate_losses is None:
                aggregate_losses = loss
            else:
                aggregate_losses = torch.cat((aggregate_losses.view(-1), loss.view(-1)), 0)
    return aggregate_losses