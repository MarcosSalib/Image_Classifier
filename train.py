import time
import os
import torch
from torch import nn, optim
import torch.nn.functional as F

import json
import numpy as np
import model_utils.py
from model_utils import get_input_args, make_folder

from torch.autograd import Variable
from collections import OrderedDict
from torchvision import datasets, transforms, models


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

model = models.densenet121(pretrained=True)
device = 'cuda'
data_transforms = {}
image_datasets = {}
dataloaders = {}
hidden_units = 500
optimizer = 0
loss = 0


def transformations():
    global data_transforms, image_datasets, dataloaders
    global data_dir, train_dir, valid_dir, test_dir
    data_transforms = {
        'train_set': transforms.Compose([transforms.Resize(256),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])]),
        'valid_set' : transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])]),
        'test_set' : transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    }

image_datasets = {
        'train_data': datasets.ImageFolder(train_dir, transform = data_transforms['train_set']),
        'valid_data': datasets.ImageFolder(valid_dir, transform = data_transforms['valid_set']),
        'test_data':  datasets.ImageFolder(test_dir,  transform = data_transforms['test_set'])
    }

dataloaders = {
    'train_loader': torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=64, shuffle=True),
    'valid_loader': torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=64, shuffle=True),
    'test_loader': torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=64, shuffle=True)
    }


def label_mapping():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)


def validation(model, validloaders, criterion):
    validating_loss = 0
    accuracy = 0
    for images, labels in validloaders:
        images = Variable(images)
        labels = Variable(labels)
        images, labels = images.to(device), labels.to(device)
        
        log_ps = model.forward(images)
        validating_loss += criterion(log_ps, labels).item()
        
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    return validating_loss, accuracy



def train_network(arch, lr=0.003, hidden_units= 500):
    global model, device, optimizer
    global data_transforms, image_datasets, dataloaders

    #turn off the parameters; to avoid backpropagation
    for param in model.parameters():
        param.requires_grad = False

    if (arch == 'vgg16'):
        num_features = model.classifier[0].in_features
    else:
        num_features = model.classifier.in_features
    #defining the feedforward classifier
    model.classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(1024, 500)),
                            ('relu', nn.ReLU()),
                            ('fc2', nn.Linear(500, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    classifier = model.classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

    #Do Training on the train set
    epochs = 4
    steps = 0
    running_loss = 0
    print_step = 20

    start_time = time.time()
    print('The training starts\n')
    model = model.train()

    for epoch in range(epochs):
        model.to(device)
        for images, labels in trainloaders:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad() 

            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
            if steps % print_step == 0:
                #turning model to evaluation mode
                model.eval()
                model.to(device)
            
                with torch.no_grad():
                    validating_loss, accuracy = validation(model, dataloaders['valid_loader'], criterion)
                #print the loss
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train Loss: {running_loss/print_step:.3f}.. "
                      f"Validation Loss: {validating_loss/len(dataloaders['valid_loader']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders['test_loader']):.3f}")
                running_loss = 0
                #model back to training mode
                model.train()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Completed {:02d} epoch{} using device: {}'.format(epochs, 's' if epochs > 1 else '', device))
    print("\n** Elapsed Runtime:",
              str(int((elapsed_time/3600)))+":"+str(int((elapsed_time%3600)/60))+":"
              +str(int((elapsed_time%3600)%60)) )

def test_network():
    global model, data_transforms, image_datasets, dataloaders
    accurate = 0
    total = 0
    with torch.no_grad():
        model.to(device)
        model.eval()
        for ii, (inputs, labels) in enumerate(testloaders['test_loader']):
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accurate += (predicted == labels).sum().item()
    print('Accuracy: %d %%' % (100 * accurate / total))


def save_checkpoint(checkpoint_path, arch):
    global model, device, optimizer
    checkpoint = {'input_size': 1024,
                  'output_size': 102,
                  'epochs': 4,
                  'batch_size': 64,
                  'model': models.densenet121(pretrained=True),
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'arch_name': 'densenet121',
                  'optimizer_state': optimizer.state_dict(),
                  'class_to_idx': train_datasets.class_to_idx}
    os.mkdir(checkpoint_path)
    torch.save(checkpoint, checkpoint_path+'/checkpoint.pth')


def main():
    global data_dir, train_dir, valid_dir, test_dir
    global model, device, epochs, hidden_units
    
    in_arg = get_input_args()
    data_dir = in_arg.data_dir
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    lr = float(in_arg.learn_rate.strip().strip("'"))
    if (in_arg.arch == 'densenet121'):
        model = models.densenet121(pretrained=True)
    elif (in_arg.arch == 'vgg16'):
        model = models.vgg16(pretrained=True)
    else:
        print('Error')
        return
        
    if (in_arg.to_device == 'gpu'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    epochs = in_arg.epochs
    hidden_units = in_arg.hidden_units
    
    transformations()
    label_mapping()
    build_train_network(in_arg.arch, lr, hidden_units)
    test_network()
    return 

if __name__ == '__main__':
    main()
