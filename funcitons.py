# TODO1: move imports to a dictionary
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def set_train_parser(parser):
    parser.add_argument("test_dir", help="data directory that contains data for training, testing and validation")

    parser.add_argument('--gpu', action="store_true",
                        dest="gpu", default=False,
                        help='Enable GPU mode', required = False)

    parser.add_argument('--learning_rate', action="store",
                        dest="learning_rate", type=float, default = 0.0001,
                        help='Set the learning rate', required = False)

    parser.add_argument('--hidden_units', action="store",
                        dest="hidden_units", type=int, default = 512,
                        help='Set the number of hidden units in range of (512, 2048)', required = False)

    parser.add_argument('--epochs', action="store",
                        dest="epochs", type=int, default = 3,
                        help='Set the value of epochs', required = False)

    parser.add_argument('--save_dir', action="store",
                        dest='save_dir', default = 'flower_classifier_checkpoint',
                        help='Save checkpoint to directory', required = False)

    parser.add_argument('--arch', action="store",
                        dest="arch", default="vgg11",
                        help='Choose an architecture', required = False)
    return

def load_train_data(train_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(35),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.456],
                                                            [0.229, 0.224, 0.225])])

    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)

    return train_datasets

def load_test_data(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(225),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.456],
                                                           [0.229, 0.224, 0.225])])

    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)

    return test_datasets

def load_model(arch):
    if arch == 'vgg11' :
        model = models.vgg11(pretrained=True)
        feature_size = 25088
    elif arch == 'vgg13' :
        model = models.vgg13(pretrained=True)
        feature_size = 25088
    elif arch == 'vgg16' :
        model = models.vgg16(pretrained=True)
        feature_size = 25088
    elif arch == 'vgg19' :
        model = models.vgg19(pretrained=True)
        feature_size = 25088
    elif arch == 'resnet18' :
        model = models.resnet18(pretrained=True)
        feature_size = 512
    elif arch == 'resnet34' :
        model = models.resnet34(pretrained=True)
        feature_size = 512
    elif arch == 'resnet50' :
        model = models.resnet50(pretrained=True)
        feature_size = 2048
    elif arch == 'resnet101' :
        model = models.resnet101(pretrained=True)
        feature_size = 2048
    elif arch == 'resnet152' :
        model = models.resnet152(pretrained=True)
        feature_size = 2048
    elif arch == 'densenet121' :
        model = models.densenet121(pretrained=True)
        feature_size = 1024
    elif arch == 'densenet169' :
        model = models.densenet169(pretrained=True)
        feature_size = 1664
    elif arch == 'densenet161' :
        model = models.densenet161(pretrained=True)
        feature_size = 2208
    elif arch == 'densenet201' :
        model = models.densenet201(pretrained=True)
        feature_size = 1920
    else :
        print('error: no a valid neural networks architecture')

    return model, feature_size

# Update: Add 'feature_size' and 'target_size' arguments to make training model more general
def update_model(model, arch, feature_size, target_size, hidden_units):
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Build Flower Classifier
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(feature_size, 5000)), # input feature must match pre-trained model
                                        ('relu', nn.ReLU()),
                                        ('dropout', nn.Dropout(p=0.2)),
                                        ('fc2', nn.Linear(5000,2048)),
                                        ('relu', nn.ReLU()),
                                        ('dropout', nn.Dropout(p=0.2)),
                                        ('fc3', nn.Linear(2048,hidden_units)),
                                        ('relu', nn.ReLU()),
                                        ('dropout', nn.Dropout(p=0.2)),
                                        ('fc4', nn.Linear(hidden_units,target_size)),
                                        ('output', nn.LogSoftmax(dim=1))
                                        ]))

    # Update: add options if pre-trained model doesn't have classifier attribute
    if 'resnet' in arch :
        model.fc = classifier
    else :
        model.classifier = classifier

    return model

def validate_model(model, validloaders, dev):

    running_loss = 0
    accuracy = 0
    criterion = nn.NLLLoss()

    model.eval()
    model.to(dev)

    with torch.no_grad():
        for images, labels in validloaders:
            images, labels = images.to(dev), labels.to(dev)

            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)

            # Update 'running_loss'
            running_loss += loss.item()

            ps = torch.exp(log_ps)
            top_p, top_c = ps.topk(1, dim=1)
            equal = top_c == labels.view(*top_c.shape)
            accuracy += torch.mean(equal.type(torch.FloatTensor)).item()

        # TODO: study the printout more
        print(f"Test loss: {running_loss/len(validloaders)}")
        print("Accuracy: {:.3f}".format(accuracy/len(validloaders)))

    model.train()
    return

def train_model(model, feature_size, target_size, trainloaders, train_datasets, validloaders, epochs, learning_rate, arch, hidden_layers, save_dir, dev):
    steps = 0
    running_loss = 0

    # Move model to cuda if available
    model.to(dev);

    criterion = nn.NLLLoss()

    if 'resnet' in arch:
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    for e in range(epochs):
        for images, labels in trainloaders:

            # Move Images and Labels tensors to GPU
            images, labels = images.to(dev), labels.to(dev)

            # Increment steps
            steps += 1

            # Train the networks
            optimizer.zero_grad()

            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)

            loss.backward()
            optimizer.step()

            # Update 'running_loss'
            running_loss += loss.item()

            # Print running_loss for debug purpose
            if(steps%30 == 0):
                #print(f"Loss Item: {loss.item()}")
                print('epochs: {}, steps: {}'.format(e, steps))
                print(f"Training loss: {running_loss/steps}")
                # Validate Model
                validate_model(model, validloaders, dev)

    # Update (Done): add number of hidden layers and architecture name
    checkpoint = {'feature_size':feature_size,
                  'target_size': target_size,
                  'epochs': epochs,
                  'arch': arch,
                  'hidden_layers': hidden_layers,
                  'class_to_idx': train_datasets.class_to_idx,
                  'state_dict': model.state_dict()}

    save_path = save_dir+'.pth'
    torch.save(checkpoint, save_path)

    return model

def load_checkpoint(path):
    checkpoint = torch.load(path)

    # Update (Done): load model with name and input_size
    model, feature_size = load_model(checkpoint['arch'])

    # Update (Done):  Update model with feature_size, target_size and number of hidden_layers
    model = update_model(model, checkpoint['feature_size'],
                         checkpoint['target_size'],
                         checkpoint['hidden_layers'])

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def set_predict_parser(parser):
    parser.add_argument("path_to_image", help="path to the image")
    parser.add_argument("checkpoint", help="checkpoint to load for the trained neural networks")

    parser.add_argument('--gpu', action="store_true",
                        dest="gpu", default=False,
                        help='Enable GPU mode', required = False)

    parser.add_argument('--category_names', action="store",
                        dest="category_names", default='cat_to_name.json',
                        help='json file to use for category name lookup', required = False)

    parser.add_argument('--top_k', action="store",
                        dest="top_k", type = int, default = 5,
                        help='number of predictions to display', required = False)
    return

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    h = image.height
    w = image.width

    # Resize the PIL image with the shortest edge to 256
    if (h<w):
        pil_image = image.resize((int(w/h)*255,255))
    else:
        pil_image = image.resize((255, int(h/w)*255))

    # Crop picture to 224x224
    w, h = pil_image.size
    left = (w - 224) / 2
    top = (h - 224) / 2
    right = (w + 224) / 2
    bottom = (h + 224) / 2

    pil_image = pil_image.crop((left, top, right, bottom))

    # Convert to floats in range (0,1)
    np_image = np.array(pil_image)/255

    # Normalize values
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std

    # Swap first and third dimension
    np_image = np_image.transpose((2,0,1))

    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    #image = image.numpy().transpose((1, 2, 0))
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def predict(image_path, model, topk, dev):

    # Process Image
    img = process_image(Image.open(image_path))

    # Display Image
    #imshow(img)

    # Convert format of img, move to cuda
    img = torch.FloatTensor(img)
    img = img.to(dev)

    # Update model to eval mode
    model.eval()
    model.to(dev)

    # Predict image
    output = model.forward(img.unsqueeze_(0))
    ps = torch.exp(output)

    # Find out top predictions of the image
    top_pbs, top_classes = ps.topk(topk)

    # Move data back to CPU for future calculation
    top_pbs = top_pbs.cpu()
    top_classes = top_classes.cpu()

    # Return as numpy format
    return (top_pbs.data.numpy()[0], top_classes.data.numpy()[0])

def organize_data(classes, probs, class_to_idx, cat_to_name):

    idx_to_class = {v:k for k,v in class_to_idx.items()}
    class_labels = [idx_to_class[idx] for idx in classes]

    # Generate a new flower name list from class list
    names = list()

    for i in range(len(class_labels)):
        names.append(cat_to_name[str(class_labels[i])])

    plt_data = dict()

    # Generate plt_data numpy Series from 'names' and 'probs' list
    plt_data = dict(zip(names, probs))
    plt_data = {'names': names, 'probs': probs}

    return plt_data
