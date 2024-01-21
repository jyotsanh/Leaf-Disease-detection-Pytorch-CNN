import pandas as pd
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch import nn
from sklearn.metrics import accuracy_score

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((32,32))
    ]
)

train_data = ImageFolder(
    root='G:/Github/Leaf-Disease-detection-Pytorch-CNN/data/train',
    transform=transform
    )

valid_data = ImageFolder(
    root='G:/Github/Leaf-Disease-detection-Pytorch-CNN/data/valid',
    transform=transform
    )

train_loader = DataLoader(
    dataset=train_data,
    shuffle=True,
    batch_size = 32
    )
test_loader = DataLoader(
    dataset=valid_data,
    shuffle=True,
    batch_size = 32
    )



class CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
        nn.Conv2d(
            in_channels=3,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
            ),
        nn.LeakyReLU(inplace=True),
        nn.BatchNorm2d(num_features=128),
        nn.MaxPool2d(kernel_size=2)

    )
        self.block2 = nn.Sequential(
        nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1
            ),
        nn.LeakyReLU(inplace=True),
        nn.BatchNorm2d(num_features=256),
        nn.MaxPool2d(kernel_size=2)

    )
        self.block3 = nn.Sequential(
        nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1
            ),
        nn.LeakyReLU(inplace=True),
        nn.BatchNorm2d(num_features=512),
        nn.MaxPool2d(kernel_size=2)

    )
        self.classify = nn.Sequential(
        nn.Dropout(0.2),
        nn.Flatten(),
        nn.Linear(
            in_features = 4*4*512,
            out_features = 512
                  ),
        nn.LeakyReLU(inplace=True),
        nn.Linear(
            in_features=512,
            out_features=38
                  )

    )
    def forward(self,x):
        x1 = self.block1(x) 
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.classify(x3)
        
        return x4
    
    
model = CNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(),lr=0.0001)
epochs = 20

for i in range(epochs):
    for batch,(features,target) in enumerate(train_loader):
        feature = features.to(device)
        targets = target.to(device)
        
        model.train()
        y_pred = model(feature)
        loss = loss_fn(y_pred.squeeze(),targets)
        optim.zero_grad()
        loss.backward()
        optim.step()
    pred = []
    target = []
    model.eval()
    with torch.inference_mode():
        for i,(valid_feature,valid_target) in enumerate(test_loader):
            
            valid_feature = valid_feature.to(device)
            valid_target = valid_target.to(device)
            
            valid_y_pred = model(valid_feature)
            valid_loss = loss_fn(valid_y_pred,valid_target)
            pred.extend(torch.argmax(torch.softmax(valid_y_pred,dim=1),dim=1).cpu().numpy())
            target.extend(valid_target.cpu().numpy())  
    accuracy = accuracy_score(pred,target)
    print(f"epoch : {i+1}  | Training Loss : {loss.item()} | Validation Loss : {valid_loss.item()} |")  
    