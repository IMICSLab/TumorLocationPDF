# -*- coding: utf-8 -*-
"""
@author: Ernest (Khashayar) Namdar
"""
import torch.nn as nn
import torch
import copy

class CNN(nn.Module):
    """CREATE MODEL CLASS"""
    def __init__(self):
        super(CNN, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv3d(in_channels=1,
                              out_channels=3,
                              kernel_size=3,
                              stride=2,
                              padding=0)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv3d(in_channels=3,
                              out_channels=9,
                              kernel_size=3,
                              stride=2,
                              padding=0)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)

        # Convolution 3
        self.cnn3 = nn.Conv3d(in_channels=9,
                              out_channels=27,
                              kernel_size=3,
                              stride=2,
                              padding=0)
        self.relu3 = nn.ReLU()

        # Max pool 3
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)
        #Flatten
        self.flatten = nn.Flatten()

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(486, 100)
        self.fc2 = nn.Linear(101, 2)
        #self.fc3 = nn.Linear(3, 2)

    def forward(self, x, loc_pr, out_flag=False):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        if out_flag is True:
            out1 = copy.deepcopy(out.detach())

        # Max pool 1
        out = self.maxpool1(out)
        #print(out.shape)

        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)
        if out_flag is True:
            out2 = copy.deepcopy(out.detach())

        # Max pool 2
        out = self.maxpool2(out)
        #print(out.shape)

        # Convolution 3
        out = self.cnn3(out)
        out = self.relu3(out)
        if out_flag is True:
            out3 = copy.deepcopy(out.detach())

        # Max pool 3
        out = self.maxpool3(out)
        #print(out.shape)

        #out = out.view(out.size(0), -1)
        out = self.flatten(out)

        # FC
        out = self.fc1(out)
        out = torch.cat((out, torch.unsqueeze(loc_pr, dim=-1)), dim=1)
        out = self.fc2(out)
        #print("out.size()", out.size())
        #print("loc_pr.size()", loc_pr.size())
        
        #out = self.fc3(out)

        if out_flag is True:
            return out1, out2, out3, out
        else:
            return out

class CNN2(nn.Module):
    """CREATE MODEL CLASS"""
    def __init__(self):
        super(CNN2, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv3d(in_channels=1,
                              out_channels=3,
                              kernel_size=3,
                              stride=2,
                              padding=0)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv3d(in_channels=3,
                              out_channels=9,
                              kernel_size=3,
                              stride=2,
                              padding=0)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)

        # Convolution 3
        self.cnn3 = nn.Conv3d(in_channels=9,
                              out_channels=27,
                              kernel_size=3,
                              stride=2,
                              padding=0)
        self.relu3 = nn.ReLU()

        # Max pool 3
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)
        #Flatten
        self.flatten = nn.Flatten()

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(486, 100)
        self.fc2 = nn.Linear(100, 2)
        #self.fc3 = nn.Linear(3, 2)

    def forward(self, x, loc_pr, out_flag=False):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        if out_flag is True:
            out1 = copy.deepcopy(out.detach())

        # Max pool 1
        out = self.maxpool1(out)
        #print(out.shape)

        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)
        if out_flag is True:
            out2 = copy.deepcopy(out.detach())

        # Max pool 2
        out = self.maxpool2(out)
        #print(out.shape)

        # Convolution 3
        out = self.cnn3(out)
        out = self.relu3(out)
        if out_flag is True:
            out3 = copy.deepcopy(out.detach())

        # Max pool 3
        out = self.maxpool3(out)
        #print(out.shape)

        #out = out.view(out.size(0), -1)
        out = self.flatten(out)

        # FC
        out = self.fc1(out)
        out = self.fc2(out)
        loc_pr = torch.unsqueeze(loc_pr, dim=-1)
        loc_pr=torch.cat((loc_pr, 1-loc_pr), dim=1)
        #print("loc_pr", loc_pr.size())
        #print("loc_pr", loc_pr)
        #print("out", out.size())
        #print("out", out)
        out = torch.mean(torch.stack([out , loc_pr]), dim=0)
        #print("out.size()", out.size())
        #print("loc_pr.size()", loc_pr.size())
        
        #out = self.fc3(out)

        if out_flag is True:
            return out1, out2, out3, out
        else:
            return out

class CNN3(nn.Module):
    """CREATE MODEL CLASS"""
    def __init__(self):
        super(CNN3, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv3d(in_channels=1,
                              out_channels=3,
                              kernel_size=3,
                              stride=2,
                              padding=0)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv3d(in_channels=3,
                              out_channels=9,
                              kernel_size=3,
                              stride=2,
                              padding=0)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)

        # Convolution 3
        self.cnn3 = nn.Conv3d(in_channels=9,
                              out_channels=27,
                              kernel_size=3,
                              stride=2,
                              padding=0)
        self.relu3 = nn.ReLU()

        # Max pool 3
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)
        #Flatten
        self.flatten = nn.Flatten()

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(486, 100)
        self.fc2 = nn.Linear(100, 2)
        #self.fc3 = nn.Linear(3, 2)

    def forward(self, x, loc_pr, out_flag=False):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        if out_flag is True:
            out1 = copy.deepcopy(out.detach())

        # Max pool 1
        out = self.maxpool1(out)
        #print(out.shape)

        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)
        if out_flag is True:
            out2 = copy.deepcopy(out.detach())

        # Max pool 2
        out = self.maxpool2(out)
        #print(out.shape)

        # Convolution 3
        out = self.cnn3(out)
        out = self.relu3(out)
        if out_flag is True:
            out3 = copy.deepcopy(out.detach())

        # Max pool 3
        out = self.maxpool3(out)
        #print(out.shape)

        #out = out.view(out.size(0), -1)
        out = self.flatten(out)

        # FC
        out = self.fc1(out)
        out = self.fc2(out)
        loc_pr = torch.unsqueeze(loc_pr, dim=-1)
        loc_pr=torch.cat((1-loc_pr, loc_pr), dim=1)
        #print("loc_pr", loc_pr.size())
        #print("loc_pr", loc_pr)
        #print("out", out.size())
        #print("out", out)
        out = torch.mean(torch.stack([out , loc_pr]), dim=0)
        #print("out.size()", out.size())
        #print("loc_pr.size()", loc_pr.size())
        
        #out = self.fc3(out)

        if out_flag is True:
            return out1, out2, out3, out
        else:
            return out

class CNN4(nn.Module):
    """CREATE MODEL CLASS"""
    def __init__(self):
        super(CNN4, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv3d(in_channels=1,
                              out_channels=3,
                              kernel_size=3,
                              stride=2,
                              padding=0)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv3d(in_channels=3,
                              out_channels=9,
                              kernel_size=3,
                              stride=2,
                              padding=0)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)

        # Convolution 3
        self.cnn3 = nn.Conv3d(in_channels=9,
                              out_channels=27,
                              kernel_size=3,
                              stride=2,
                              padding=0)
        self.relu3 = nn.ReLU()

        # Max pool 3
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)
        #Flatten
        self.flatten = nn.Flatten()

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(486, 100)
        self.fc2 = nn.Linear(101, 2)
        #self.fc3 = nn.Linear(3, 2)

    def forward(self, x, loc_pr, out_flag=False):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        if out_flag is True:
            out1 = copy.deepcopy(out.detach())

        # Max pool 1
        out = self.maxpool1(out)
        #print(out.shape)

        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)
        if out_flag is True:
            out2 = copy.deepcopy(out.detach())

        # Max pool 2
        out = self.maxpool2(out)
        #print(out.shape)

        # Convolution 3
        out = self.cnn3(out)
        out = self.relu3(out)
        if out_flag is True:
            out3 = copy.deepcopy(out.detach())

        # Max pool 3
        out = self.maxpool3(out)
        #print(out.shape)

        #out = out.view(out.size(0), -1)
        out = self.flatten(out)

        # FC
        out = self.fc1(out)
        out = torch.cat((out, torch.unsqueeze(loc_pr, dim=-1)), dim=1)
        out = self.fc2(out)
        #out = nn.functional.softmax(out, dim=-1)
        #print("cnn_pr", out)
        loc_pr = torch.unsqueeze(loc_pr, dim=-1)
        loc_pr=torch.cat((1-loc_pr, loc_pr), dim=1)
        #print("loc_pr", loc_pr)
        #print("loc_pr", loc_pr.size())
        #print("loc_pr", loc_pr)
        #print("out", out.size())
        #print("out", out)
        out = torch.mean(torch.stack([out , loc_pr]), dim=0)
        #print("out.size()", out.size())
        #print("loc_pr.size()", loc_pr.size())
        
        #out = self.fc3(out)

        if out_flag is True:
            return out1, out2, out3, out
        else:
            return out

class CNN5(nn.Module):
    """CREATE MODEL CLASS"""
    def __init__(self):
        super(CNN5, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv3d(in_channels=1,
                              out_channels=3,
                              kernel_size=3,
                              stride=2,
                              padding=0)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv3d(in_channels=3,
                              out_channels=9,
                              kernel_size=3,
                              stride=2,
                              padding=0)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)

        # Convolution 3
        self.cnn3 = nn.Conv3d(in_channels=9,
                              out_channels=27,
                              kernel_size=3,
                              stride=2,
                              padding=0)
        self.relu3 = nn.ReLU()

        # Max pool 3
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)
        #Flatten
        self.flatten = nn.Flatten()

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(486, 40)
        self.fc2 = nn.Linear(42, 2)
        #self.fc3 = nn.Linear(3, 2)

    def forward(self, x, loc_pr, out_flag=False):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        if out_flag is True:
            out1 = copy.deepcopy(out.detach())

        # Max pool 1
        out = self.maxpool1(out)
        #print(out.shape)

        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)
        if out_flag is True:
            out2 = copy.deepcopy(out.detach())

        # Max pool 2
        out = self.maxpool2(out)
        #print(out.shape)

        # Convolution 3
        out = self.cnn3(out)
        out = self.relu3(out)
        if out_flag is True:
            out3 = copy.deepcopy(out.detach())

        # Max pool 3
        out = self.maxpool3(out)
        #print(out.shape)

        #out = out.view(out.size(0), -1)
        out = self.flatten(out)

        # FC
        out = self.fc1(out)
        loc_pr = torch.unsqueeze(loc_pr, dim=-1)
        loc_pr=torch.cat((1-loc_pr, loc_pr), dim=1)
        out = torch.cat((out, loc_pr), dim=1)
        out = self.fc2(out)
        #out = nn.functional.softmax(out, dim=-1)
        #print("cnn_pr", out)
        
        #print("loc_pr", loc_pr)
        #print("loc_pr", loc_pr.size())
        #print("loc_pr", loc_pr)
        #print("out", out.size())
        #print("out", out)
        out = torch.mean(torch.stack([out , loc_pr]), dim=0)
        #print("out.size()", out.size())
        #print("loc_pr.size()", loc_pr.size())
        
        #out = self.fc3(out)

        if out_flag is True:
            return out1, out2, out3, out
        else:
            return out