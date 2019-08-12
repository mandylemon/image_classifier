from functions import *

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable

from workspace_utils import active_session

import argparse

parser = argparse.ArgumentParser(description='Neural Networks Training')
set_train_parser(parser)
args = parser.parse_args()

data_dir = args.test_dir

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_datasets = load_train_data(train_dir)
valid_datasets = load_test_data(valid_dir)
test_datasets = load_test_data(test_dir)

trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size = 30, shuffle = True)
validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size = 30)
testloaders = torch.utils.data.DataLoader(valid_datasets, batch_size = 30)

# Load pre-trained module
model, input_features = load_model(args.arch)

# Update model to match datasets
model = update_model(model, args.arch, input_features, len(train_datasets.class_to_idx), args.hidden_units)

# Define device
if args.gpu == True :
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else :
    dev = "cpu"

# Train the networks
with active_session():
    model = train_model(model, input_features, len(train_datasets.class_to_idx), trainloaders, train_datasets,
                        validloaders, args.epochs, args.learning_rate, args.arch, args.hidden_units,
                        args.save_dir, dev)

#model = load_checkpoint(model, 'image_classifier_checkpoint.pth')

# test the model
validate_model(model, testloaders, dev)