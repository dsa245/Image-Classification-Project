import argparse
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, models, transforms
import os
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(
    description='Flower classification model predicting arguments',
)

parser.add_argument('image_path', type=str, action='store', help='path to the image')
parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pth', help='path to save checkpoint')
parser.add_argument('--gpu', type=bool, default=False, action='store', help='use gpu')
parser.add_argument('--topk', type=int, default=5, action='store', help='top k classes')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Image category file')

args = parser.parse_args()
device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=device)
  
    model = checkpoint['arch']

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classfier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)

    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    img = transform(img)
    return img

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    model = model.to(device)
    class_idx_mapping = model.class_to_idx
    idx_class_mapping = {v: k for k, v in class_idx_mapping.items()}

    with torch.no_grad():
        inputs = process_image(image_path)
        inputs.unsqueeze_(0)
        inputs = inputs.to(device)
        outputs = model.forward(inputs)
        ps = torch.exp(outputs)
        probs, indices = ps.topk(topk, dim = 1)
        probs = [prob.item() for prob in probs[0]]
        classes = [idx_class_mapping[index.item()] for index in indices[0]]
      
    return probs, classes


def main():
    import json
    with open(args.category_names, 'r') as f:
      cat_to_name = json.load(f)

	


    model = load_checkpoint(args.checkpoint_path)


    probs, classes = predict(args.image_path, model, args.topk)
    class_names = [cat_to_name[c] for c in classes]
    print(probs)
    print(classes)
    print(class_names)

if __name__ == "__main__":
    main()


