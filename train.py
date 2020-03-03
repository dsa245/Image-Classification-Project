import argparse
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, models, transforms
import os
import numpy as np



parser = argparse.ArgumentParser(
    description='Flower classification model training arguments',
)

parser.add_argument('data_dir', type=str, action='store', help='path to dataset')
parser.add_argument('--save_dir', type=str, default=os.getcwd(), action='store', help='path to save checkpoint')
parser.add_argument('--gpu', type = bool, default=False, action='store', help='use gpu')
parser.add_argument('--epochs', type=int, default=5, action='store', help='epochs')
parser.add_argument('--arch', type=str, default='vgg19', action='store', help='model architecture')
parser.add_argument('--lr', type=float, default=0.0003, action='store', help='learning rate')
parser.add_argument('--hidden_unit', type=int, default=4096, action='store',help='hidden unit size')
parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='checkpoint file')
parser.add_argument('--print_every', type=int, default=20, help='print frequency')

args = parser.parse_args()


def main():
  train_dir = args.data_dir + '/train'
  valid_dir = args.data_dir + '/valid'

  train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
  valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

  train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
  valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)

  trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
  validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64, shuffle = True)



  model = models.__dict__[args.arch](pretrained=True)
  in_features = model.classifier[0].in_features

  for param in model.parameters():
  	param.requires_grad = False
  	classifier = nn.Sequential(nn.Linear(in_features, args.hidden_unit),
                           nn.ReLU(),
                           nn.Linear(args.hidden_unit, 102),
                           nn.LogSoftmax(dim = 1)
                           )
  model.classifier = classifier
  device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

  criterion = nn.NLLLoss()
  optimizer = optim.Adam(model.classifier.parameters(), lr = args.lr)
  epochs = args.epochs
  print_every = args.print_every
  steps = 0
  running_loss = 0
  model.train()
  for e in range(epochs):
    for inputs, labels in trainloader:
      steps += 1
      model.to(device)
      inputs, labels = inputs.to(device), labels.to(device)
      optimizer.zero_grad()

      logps = model.forward(inputs)
      loss = criterion(logps, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()    		
      if steps % print_every == 0:
        model.eval()
        with torch.no_grad():			
          valid_loss = 0
          accuracy = 0

          for inputs, labels in validloader:
            model.to(device)
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim = 1)
            equals = top_class == labels.view(*top_class.shape)

            valid_loss += loss.item()
            accuracy += torch.mean(equals.type(torch.FloatTensor))

        print('epoch: {}/{}.. '.format(e + 1, epochs),
        			'training_loss: {:.3f}.. '.format(running_loss/print_every),
            	'valid_loss: {:.3f}.. '.format(valid_loss/len(validloader)),
            	'accuracy: {:.3f}.. '.format(accuracy/len(validloader)))

        running_loss = 0
        model.train()



  model.class_to_idx = train_data.class_to_idx
  checkpoint = {'classfier': model.classifier,
                'epochs': epochs,
                'arch': models.__dict__[args.arch](pretrained = True),
                'class_to_idx': model.class_to_idx,
                'optim_state_dict': optimizer.state_dict(),
                'model_state_dict': model.state_dict()}
  torch.save(args.save_dir, args.checkpoint)



if __name__ == "__main__":
    main()




