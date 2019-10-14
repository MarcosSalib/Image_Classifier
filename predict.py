import torch 
import torchvision
from torch import nn, optim
from torchvision import datasets, transforms, models

import json
import time

import numpy as np
import pandas as pd
import argparse
import os
from PIL import Image

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    
    arch_name = saved_model['arch_name']
    if (arch_name == 'densenet121'):
        model = models.densenet121(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
    
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = saved_model['optimizer_state']
    epochs = checkpoint['epochs']
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model, checkpoint['class_to_idx']


def process_image(image):
    img = Image.open(image)
    width, height = img.size
    
    #resize
    shortest = min(width, height)
    pil_image = img.resize((int((width/shortest)*256), int((height/shortest)*256)))

    #crop out
    left = (width - 224) / 2
    upper = (height - 224)/ 2
    top = (width + 224) / 2
    lower = (height + 224) / 2
    img = img.crop((left, upper, top, lower))
    
    #normalize
    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])
    img = np.array(pil_image) / 225
    pil_image = ( img - img_mean) / img_std
    pil_image = np.transpose(pil_image, (2,0,1))
    return torch.Tensor(pil_image)

def predict(image_path, model, to_device, topk=5):
    
    if (to_device == 'gpu'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'

    model.to(device)
    model.eval()
    image = process_image(image_path)
    image = image.to(device)
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        log_ps = model.forward(image)
        top_p, top_class = torch.topk(log_ps, topk)
        probs = top_p.exp()
        class_to_idx = model.class_to_idx
        
    probs = probs.cpu().numpy()
    probs_labels = top_class.cpu().numpy()
    
    class_idx = {model.class_to_idx[i]: i for i in model.class_to_idx}
    class_list = []
    
    for label in probs_labels[0]:
        class_list.append(class_idx[label])
        
    return (probs[0], class_list)

def show_predict(probs, classes, json_names):
    with open(json_names, 'r') as f:
        cat_to_name = json.load(f)
    flower_names = [cat_to_name[i]] for i in classes

    df = pd.DataFrame(
        {'flowers': pd.Series(data= flower_names),
         'probabilities': pd.Series(data=probs, dtype= 'float64')})
    print(df)

def get_in_args():
    parser = argparse.ArgumentParser()
    
    # command line options
    parser.add_argument('--data_dir', type = str, default = 'flowers/', help = 'path to flower images')
    parser.add_argument('--save_dir', type = str, default = 'checkpoints', help = 'path to model checkpoints')
    parser.add_argument('--path_to_image', type = str, default = '/99/image_07833.jpg', help = 'path to an image file')  
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'path to category labels.json')
    parser.add_argument('--to_device', type = str, default = 'gpu', help = 'Run model on CPU or GPU')
    parser.add_argument('--top_k_classes', type = int, default = 3, help = 'Top k')
    return parser.parse_args()

def main():
    in_arg = get_input_args()

    data_dir = in_arg.data_dir
    test_dir = data_dir + 'test'
    image_file = in_arg.path_to_image
    
    print('data_dir: {}'.format(data_dir))
    print('test_dir: {}'.format(test_dir))
    print('save_dir: {}'.format(in_arg.save_dir))
    print('image_file: {}'.format(test_dir+image_file))
    print('category_names: {}'.format(in_arg.category_names))
    print('to_device: {}'.format(in_arg.to_device))
    print('top_k_classes: {}'.format(in_arg.top_k_classes))

    # make sure of checkpoint
    if os.path.exists(in_arg.save_dir+'/trainpy_checkpoint.pth'):
        model, class_to_idx = load_saved_checkpoint(in_arg.save_dir+'/trainpy_checkpoint.pth')
        probs, classes = predict(test_dir+image_file, model, in_arg.to_device, in_arg.top_k_classes)
        show_prediction(probs, classes, in_arg.category_names)
    else:
        print('Checkpoint does not exist! ({})'.format(in_arg.save_dir+'/trainpy_checkpoint.pth'))
    return

if __name__ == '__main__':
    main()