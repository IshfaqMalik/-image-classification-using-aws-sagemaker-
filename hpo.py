#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os

import argparse

from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def test(model, test_loader, criterion):
    '''
    TODO: Complete this function that can take a model and a 
    testing data loader and will get the test accuray/loss of the model
    Remember to include any debugging/profiling hooks that you might need
    '''
    test_loss = 0.0
    correct = 0
    total = 0

    model.to("cpu")
    model.eval()

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to("cpu"), target.to("cpu")
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

    avg_loss = test_loss / total
    accuracy = 100. * correct / total
    print(f"average test loss = {avg_loss}") 
    print(f"test accuracy = {accuracy}") 
    #print(f"Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

def train(model, train_loader, criterion, optimizer, epochs, valid_loader, device):
    '''
    TODO: Complete this function that can take a model and
    data loaders for training and will get train the model
    Remember to include any debugging/profiling hooks that you might need
    '''
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        train_loss /= len(train_loader.dataset)

        valid_loss = 0.0
        model.eval()

        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item() * data.size(0)

        valid_loss /= len(valid_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | Training Loss: {train_loss:.4f} | Validation Loss: {valid_loss:.4f}")

    return model

def net(num_classes, device):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.densenet161(pretrained = True) 
    for param in model.parameters():
        param.requires_grad = False 
    num_features = model.classifier.in_features 
    model.classifier = nn.Sequential(
                 nn.Linear(num_features, num_classes))
    model = model.to(device)

    return model

def create_data_loaders(data, batch_size, shuffle=True):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    dataloader = torch.utils.data.DataLoader(dataset=data,batch_size= batch_size, shuffle = shuffle)

    return dataloader

def create_transformer(split, image_size):
    '''
    Returns torchvision transforms with augmentations for training,
    and standard transforms for validation/testing.
    '''
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == "train":
        transform = transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),              
            transforms.RandomHorizontalFlip(),                     
            transforms.RandomRotation(degrees=30),                
            transforms.ColorJitter(brightness=0.2, contrast=0.2), 
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
             ])
    elif split in["valid", "test"]:
        transform = transforms.Compose([
            transforms.Resize(image_size + 32),  
            transforms.CenterCrop(image_size), 
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
            ])
    else: 
        raise ValueError("split must be 'train', 'valid', or 'test'")
    
    return transform 

def main(args):
    """
    Main execution function: prepares data, trains model, evaluates, and saves it.
    """
    # Build full paths to train, valid, test
    train_dir = os.path.join(args.data_dir, 'train')
    valid_dir = os.path.join(args.data_dir, 'valid')
    test_dir = os.path.join(args.data_dir, 'test')

    # ✅ Validate directory structure before loading
    for name, path in [("train", train_dir), ("valid", valid_dir), ("test", test_dir)]:
        if not os.path.exists(path):
            raise ValueError(f"{name.upper()} directory does not exist: {path}")
        if len(os.listdir(path)) == 0:
            raise ValueError(f"{name.upper()} directory is empty: {path}")
        print(f"{name.upper()} directory found: {path} with {len(os.listdir(path))} class folders")

    # Create transforms
    train_transform = create_transformer("train", args.image_size)
    valid_transform = create_transformer("valid", args.image_size)
    test_transform = create_transformer("test", args.image_size)

    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    # ✅ Print dataset sizes
    print(f"Loaded {len(train_dataset)} training samples.")
    print(f"Loaded {len(valid_dataset)} validation samples.")
    print(f"Loaded {len(test_dataset)} test samples.")

    # ✅ Raise error early if any dataset is empty
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Check class folders and images.")
    if len(train_dataset.classes) < 2:
        raise ValueError("Less than 2 classes detected. 'ImageFolder' expects class subfolders.")

    # Create data loaders
    train_loader = create_data_loaders(train_dataset, args.batch_size, shuffle=True)
    valid_loader = create_data_loaders(valid_dataset, args.batch_size, shuffle=False)
    test_loader = create_data_loaders(test_dataset, args.batch_size, shuffle=False)

    # Initialize model
    model = net(args.num_classes, args.device)

    # Loss and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    learning_rate = args.lr if args.lr else args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model with extra safety logging
    try:
        model = train(model, train_loader, loss_criterion, optimizer, args.epochs, valid_loader, args.device)
    except Exception as e:
        print(f"Error during training loop: {str(e)}")
        raise

    # Test and log final metrics
    test_loss, test_acc = test(model, test_loader, loss_criterion)

    # ✅ Log metric for SageMaker tuner
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_acc}")

    # Save model
    save_file = os.path.join(args.save_path, "model.pth")
    torch.save(model.cpu().state_dict(), save_file)
    print(f"Model saved to {save_file}")



    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    

    parser.add_argument("--data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "data"))
    parser.add_argument("--save_path", type=str, default=os.environ.get("SM_MODEL_DIR", "model"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=133)
    parser.add_argument("--image_size", type=int, default=256)

    
    parser.add_argument("--lr", type=float, default=0.0001)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    main(args)
 

