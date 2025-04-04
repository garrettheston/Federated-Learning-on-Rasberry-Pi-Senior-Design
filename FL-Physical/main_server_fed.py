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
#from models.Nets import MLP, CNNMnist, CNNCifar, ResNetTest
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
#from opacus.validators import ModuleValidator
from torch.utils.data import Dataset, DataLoader
from models.server_ssh import Connection_handling
import paramiko
from scp import SCPClient
import socket
import time
import threading

class CustomDataset(Dataset):
    def __init__(self, data_tensor):
        #self.data = data_tensor[:, :-1]
        self.data = data_tensor[:, :-1].reshape(-1,1, 9, 100) #[batch_size, channels, height, width]
        self.targets = data_tensor[:, -1]
        print(self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    
def IMU_noniid(dataset, num_users,labels):
    """
    Sample non-I.I.D client data from IMU dataset 
    Altered from Mnist_noniid definition
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 90, 10
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    #labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 30, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        dict_users[i] = [int(x) for x in dict_users[i]]
    return dict_users

#Define function to adjust Privacy Budget
def adjustPB(PBList, accList):
    print(PBList)
    newAcc = []
    newPB = PBList.copy()
    newAcc = accList.copy()
    newAcc.sort()
    i = len(newAcc)
    indexList = []
    for item in newAcc: 
        index = accList.index(item)
        if (index in indexList):
            accList[index] = 0
        index = accList.index(item)
        indexList.append(index)
        print(index)
        newPB[index] = newPB[index] - (0.1*i)
        #Set some bounds
        if (newPB[index] > 2):
            newPB[index] = 2
        if (newPB[index] < 0.7):
            newPB[index] = 0.7
        i -= 1
    print(newPB)
    return newPB
    
if __name__ == '__main__':
    f = open("config_server.txt", "r")
    lineCount = 0
    for line in f:
        currentLine = line.strip('\n').split("=")
        print(currentLine)

        if currentLine[0] == 'NUM_CLIENTS':
            NUM_CLIENTS = currentLine[1]
            NUM_CLIENTS = int(NUM_CLIENTS)
        
        if currentLine[0] == 'SERVER_PORT':
            PORT = currentLine[1]
            PORT = int(PORT)

        if currentLine[0] == 'SERVER_IP':
            SERVER = currentLine[1]

        if currentLine[0] == 'MODELFOLDER':
            MODELFOLDER = currentLine[1] 
        
        if currentLine[0] == 'NUM_gl_EPOCHS':
            NUM_gl_EPOCHS = currentLine[1] 
            NUM_gl_EPOCHS = int(NUM_gl_EPOCHS)      

        
        lineCount += 1

    f.close()
    # parse args
    args = args_parser()
    args.device = torch.device('cpu')
    ################## args def for testing
    args.num_users = NUM_CLIENTS
    args.epochs = NUM_gl_EPOCHS
    args.dataset = 'HAR_LS' 
    args.model = 'resnet' 
    args.num_channels = 1 
    args.bs =128

    ##################
    training_accuracy_list = []
    training_loss_list = []

    # load dataset and split users
    dataset = torch.load('LS_HAR_data.pt', map_location=torch.device('cpu'))
    print(dataset.shape)
    #dataset = dataset.float()
    dataset = CustomDataset(dataset)
    total_count = len(dataset)
    train_count = int(0.05*total_count) # 5%
    test_count = total_count - train_count
    random.seed(42)
    torch.manual_seed(42)
    dataset_train, dataset_test = random_split(dataset, [train_count, test_count])
    img_size = dataset_train[0][0].shape
    
    # build model
    if args.model == 'resnet':
       # net_glob = ResNetTest(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2]).to(args.device)
       net_glob = torchvision.models.resnet18()
      # net_glob.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
       net_glob.conv1 = torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
       net_glob.fc = torch.nn.Linear(net_glob.fc.in_features,5)
       net_glob.to(args.device)
    else:
        exit('Error: unrecognized model')

    net_glob.load_state_dict(torch.load("models/main_server_fed_overall.pt", map_location=torch.device('cpu')))

    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    clientAddresses = []

    file1 = open("output_FL_Resnet_HAR.txt", "w") 

    epsList = [ ]

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    
    #host = "10.4.159.106"   # this the address of server computer (not client!!)
    host = SERVER
    port = PORT

   # set up TCP socket connection for server 
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host,port))
    server.listen(args.num_users) 
        
    for iter in range(args.epochs):

        #Distribute the model to all clients
        for idx in range(0, args.num_users):
            print("In for loop :)")
            clientsocket, address = server.accept() 
            print("connection from " + address[0] + " accepted.")
            Connection_handling(clientsocket, address)

        #Wait for all client models to be received
        modelFolder = MODELFOLDER
        fileCount = 0
        while (fileCount != args.num_users):
            for file in os.scandir(modelFolder):
                if file.is_file():
                    fileCount += 1

        print("Epoch: ", iter)
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        #Comment out when using adaptive dp
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        #print("Idx List: " , idxs_users)
        
        accuracyList = []
        #for idx in idxs_users:
        for idx in range(1, args.num_users+1):
            print(" User: " , idx)
            
           # 'fed_{}_{}_{}_C{}_Non_iid{}_DP_3_clients.png'.format(args.dataset)
            File_in_use = True
            while File_in_use:
                try:
                    checkpoint = torch.load('Pi_models/main_server_fed_{}.pt'.format(idx), map_location=torch.device('cpu'))
                    File_in_use = False
                except:
                    print('Pi_models/main_server_fed_{}.pt being written currently'.format(idx))
                    time.sleep(4)
                    File_in_use = True
            net_glob.load_state_dict(checkpoint)

           # net_glob.load_state_dict(checkpoint['model_state_dict'])
            localModel = net_glob.state_dict()

            if args.all_clients:
                w_locals[idx] = copy.deepcopy(localModel)
            else:
                w_locals.append(copy.deepcopy(localModel))
            #loss_locals.append(copy.deepcopy(loss))

        #print("Epsilon List: ", epsList)

        # update global weights
        if args.global_aggr == 'FedAvg':
            w_glob = FedAvg(w_locals)
            #print('this actually runs')
        else:
            print('something wrong')
            
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        
        #save the model
        torch.save(net_glob.state_dict(), "models/main_server_fed_overall.pt")

        # print loss and accuracy of current model
        net_glob.eval()
        acc_train, l = test_img(net_glob,dataset_train, args)
        training_accuracy_list.append(acc_train)
        training_loss_list.append(l)
        print('Accuracy: ', acc_train)
        print('Loss: ', l)
        print(training_accuracy_list)
        #clearprint(training_loss_list)
        
        #Remove all previous models for new ones to come in
        for file in os.scandir(modelFolder):
            os.remove(file)
            
    # Final Connection Handling to terminate clients
    for idx in range(0, args.num_users):  
        clientsocket, address = server.accept() 
        print("connection from " + address[0] + " accepted.")
        clientsocket.send(bytes("EXIT()", "utf-8"))    
        msg = clientsocket.recv(64)
        msg_decoded = msg.decode("utf-8")
        print(msg_decoded)

    # testing
    net_glob.eval()

    acc_test, loss_test = test_img(net_glob, dataset_train, args)

    #Close the server
    server.close()
