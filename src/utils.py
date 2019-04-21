import torch

def accuracy(network, validation_set_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in validation_set_loader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct/total