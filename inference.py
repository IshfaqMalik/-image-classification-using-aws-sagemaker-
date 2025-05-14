import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
import io
import os

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_classes = 133
    model = models.densenet161(pretrained=False)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Linear
                                     (num_features, 512),
                                     nn.ReLU(),
                                     nn.Dropout(0.3),
                                     nn.Linear(512, num_classes)
                                    )
    
    model_path = os.path.join(model_dir, "model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

