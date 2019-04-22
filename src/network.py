import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.utils import accuracy
from src.batch_norm import Batch_norm_2d

class Fruit_conv_net(nn.Module):
    def __init__(self):
        super(Fruit_conv_net, self).__init__()
        self.conv_1_1 = nn.Conv2d(3, 15, 5, padding = 2, bias=False)
        self.bn_1 = Batch_norm_2d(15)
        nn.init.xavier_uniform_(self.conv_1_1.weight)
        self.pool_1 = nn.MaxPool2d(2)
        self.conv_1_2 = nn.Conv2d(15, 30, 5, padding = 2, bias=False)
        self.bn_2 = Batch_norm_2d(30)
        nn.init.xavier_uniform_(self.conv_1_2.weight)
        self.pool_2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(30 * 25 * 25, 2000)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(2000, 1000)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(1000, 95)
        nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        x = F.relu(self.pool_1(self.bn_1(self.conv_1_1(x))))
        x = F.relu(self.pool_2(self.bn_2(self.conv_1_2(x))))
        x = x.view(-1, 30 * 25 * 25)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def training(network, train_set_loader, validation_set_loader, number_of_epochs = 10, optimizer=None):
    print('Training started')
    if optimizer is None:
        optimizer = optim.Adam(network.parameters(), lr=0.0001, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(number_of_epochs):
        for i, data in enumerate(train_set_loader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print('Epoch ' + str(epoch) + ': Accuracy of the network:' + str(100*accuracy(network, validation_set_loader)) + '%')

    print('Finished Training')
