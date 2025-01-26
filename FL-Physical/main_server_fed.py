#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# Python version: 3.6
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import os
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Fed import FedAvg
from models.test import test_img
import torchvision

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torchvision
import time as t
from torch.utils.data import Dataset, DataLoader
from models.server_ssh import Connection_handling

import paramiko
from scp import SCPClient
import socket
import time
import threading

# Custom Dataset for handling MNIST
class CustomDataset(Dataset):
    def __init__(self, data_tensor, target_tensor):
        # Expecting 28x28 images with 1 channel
        self.data = data_tensor.unsqueeze(1)  # Add the channel dimension (1, 28, 28)
        self.targets = target_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Define function to adjust Privacy Budget
def adjustPB(PBList, accList):
    print("Adjusting Privacy Budget...")
    print("Original Privacy Budget List:", PBList)
    print("Accuracy List:", accList)
    
    newAcc = []
    newPB = PBList.copy()
    newAcc = accList.copy()
    newAcc.sort()
    i = len(newAcc)
    indexList = []
    for item in newAcc:
        index = accList.index(item)
        if index in indexList:
            accList[index] = 0
        index = accList.index(item)
        indexList.append(index)
        print(f"Adjusting index: {index}")
        newPB[index] = newPB[index] - (0.1 * i)
        if newPB[index] > 2:
            newPB[index] = 2
        if newPB[index] < 0.7:
            newPB[index] = 0.7
        i -= 1
    print("New Privacy Budget List:", newPB)
    return newPB

if __name__ == '__main__':
    print("Loading config_server.txt...")
    f = open("config_server.txt", "r")
    lineCount = 0
    for line in f:
        currentLine = line.strip('\n').split("=")
        print("Reading line:", currentLine)

        if currentLine[0] == 'NUM_CLIENTS':
            NUM_CLIENTS = int(currentLine[1])

        if currentLine[0] == 'SERVER_PORT':
            PORT = int(currentLine[1])

        if currentLine[0] == 'SERVER_IP':
            SERVER = currentLine[1]

        if currentLine[0] == 'MODELFOLDER':
            MODELFOLDER = currentLine[1] 

        if currentLine[0] == 'NUM_gl_EPOCHS':
            NUM_gl_EPOCHS = int(currentLine[1])       

        lineCount += 1

    f.close()
    print("Config values loaded: NUM_CLIENTS={}, SERVER_PORT={}, SERVER_IP={}".format(NUM_CLIENTS, PORT, SERVER))

    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    args.num_users = NUM_CLIENTS
    args.epochs = NUM_gl_EPOCHS
    args.dataset = 'MNIST'  # Changed dataset to MNIST
    args.model = 'resnet' 
    args.num_channels = 1
    args.bs = 128

    training_accuracy_list = []
    training_loss_list = []

    print("Loading dataset...")
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    print(f"Dataset loaded with {len(dataset)} samples.")
    
    dataset_train = CustomDataset(torch.from_numpy(np.array(dataset.data)).float(), torch.from_numpy(np.array(dataset.targets)).long())        
    
    total_count = len(dataset)
    train_count = int(0.05 * total_count)  # 5%
    test_count = total_count - train_count
    random.seed(42)
    torch.manual_seed(42)
    dataset_train, dataset_test = random_split(dataset, [train_count, test_count])
    img_size = dataset_train[0][0].shape
    
    # build model
    print("Building model...")
    if args.model == 'resnet':
        net_glob = torchvision.models.resnet18()
        net_glob.conv1 = torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
        net_glob.fc = torch.nn.Linear(net_glob.fc.in_features, 10)  # MNIST has 10 classes
        net_glob.to(args.device)
        
        # Check if the model weights file exists
        model_path = 'models/main_server_fed_overall.pt'

        if not os.path.exists(model_path):
            print(f"{model_path} not found. Initializing model and saving it.")
            # If the model doesn't exist, initialize and save it
            torch.save(net_glob.state_dict(), model_path)
        else:
            print(f"Loading model from {model_path}")
            net_glob.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            
    else:
        exit('Error: unrecognized model')

    net_glob.load_state_dict(torch.load("models/main_server_fed_overall.pt", map_location=torch.device('cpu')))
    net_glob.train()

    print("Model loaded and training started...")
    # copy weights
    w_glob = net_glob.state_dict()

    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    clientAddresses = []

    file1 = open("output_FL_Resnet_MNIST.txt", "w")

    epsList = []

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    host = SERVER
    port = PORT

    print(f"Setting up server at {host}:{port}...")
    # set up TCP socket connection for server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(args.num_users) 
    print(f"Server listening on {host}:{port}")

    for iter in range(args.epochs):
        print(f"Epoch {iter} started...")
        for idx in range(0, args.num_users):
            print("Waiting for connection from client {}...".format(idx+1))
            clientsocket, address = server.accept() 
            print(f"Connection from {address[0]} accepted.")
            Connection_handling(clientsocket, address)

        # Wait for all client models to be received
        print("Waiting for all models to be received...")
        modelFolder = MODELFOLDER
        fileCount = 0
        while (fileCount != args.num_users):
            for file in os.scandir(modelFolder):
                if file.is_file():
                    fileCount += 1
        print(f"Received all models. Starting aggregation for epoch {iter}.")

        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        accuracyList = []
        for idx in range(1, args.num_users+1):
            print(f"User {idx} training...")
            
            File_in_use = True
            while File_in_use:
                try:
                    checkpoint = torch.load('Pi_models/main_server_fed_{}.pt'.format(idx), map_location=torch.device('cpu'))
                    File_in_use = False
                except:
                    print(f'Pi_models/main_server_fed_{idx}.pt being written currently')
                    time.sleep(4)
                    File_in_use = True
            net_glob.load_state_dict(checkpoint)
            localModel = net_glob.state_dict()

            if args.all_clients:
                w_locals[idx] = copy.deepcopy(localModel)
            else:
                w_locals.append(copy.deepcopy(localModel))

        # update global weights
        if args.global_aggr == 'FedAvg':
            w_glob = FedAvg(w_locals)
        else:
            print('something wrong')

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # save the model
        print("Saving the global model...")
        torch.save(net_glob.state_dict(), "models/main_server_fed_overall.pt")

        # print loss and accuracy of current model
        net_glob.eval()
        acc_train, l = test_img(net_glob, dataset_train, args)
        training_accuracy_list.append(acc_train)
        training_loss_list.append(l)
        print(f'Accuracy: {acc_train}, Loss: {l}')

        # Remove all previous models for new ones to come in
        print("Removing previous models...")
        for file in os.scandir(modelFolder):
            os.remove(file)

    # Final Connection Handling to terminate clients
    print("Terminating client connections...")
    for idx in range(0, args.num_users):  
        clientsocket, address = server.accept() 
        print(f"Connection from {address[0]} accepted.")
        clientsocket.send(bytes("EXIT()", "utf-8"))    
        msg = clientsocket.recv(64)
        msg_decoded = msg.decode("utf-8")
        print(msg_decoded)

    # testing
    print("Testing final model...")
    net_glob.eval()
    acc_test, loss_test = test_img(net_glob, dataset_train, args)

    # Close the server
    print("Closing the server...")
    server.close()
