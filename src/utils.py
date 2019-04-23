import matplotlib.pyplot as plt
import numpy as np

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

def imshow(img, title=None):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.title(title)
    plt.show()
    
def reverse_dict(dictionary):
    return dict((v, k) for k, v in dictionary.items())

def get_first_image(image_set_loader):
    return iter(image_set_loader).next() 

def get_fruit_name(id_to_class, labels):
    return str(id_to_class[labels.numpy()[0]])
    
def multiply_images(images):
    _, _, height, width = images.shape
    return (torch.ones([width*height,1])).matmul(images.view(1,-1)).view(width*height,3,height,width)  

def make_cutouts(multiplied_images, radius_of_kernel, colours):
    _, _, height, width = multiplied_images.shape
    for i in range(100):
        for j in range(100):
            multiplied_images[100*i+j][0][max(i-radius_of_kernel,0):min(i+radius_of_kernel,height), max(j-radius_of_kernel,0):min(j+radius_of_kernel, width)]=colours['R']
            multiplied_images[100*i+j][1][max(i-radius_of_kernel,0):min(i+radius_of_kernel,height), max(j-radius_of_kernel,0):min(j+radius_of_kernel, width)]=colours['G']
            multiplied_images[100*i+j][2][max(i-radius_of_kernel,0):min(i+radius_of_kernel,height), max(j-radius_of_kernel,0):min(j+radius_of_kernel, width)]=colours['B']
            
def get_max_min_abs_max(colour):
    colour_max = torch.max(colour)
    colour_min = torch.min(colour)
    colour_abs_max = torch.max(torch.abs(colour_max),torch.abs(colour_min))
    return colour_max, colour_min, colour_abs_max

def multiply_gray_picture(picture):
    return (torch.ones([3,1]).matmul((picture).view(1, -1))).view(3,100,100)