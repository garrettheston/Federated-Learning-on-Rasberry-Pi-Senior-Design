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
from Connection_Handle import connection_handling
from Crypto_Utils import kyber_key_exchange_server, encrypt_model, decrypt_model, wait_for_complete_file
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

def handle_client(idx, server, net_glob):
    try:

        clientsocket, address = server.accept()  # Accepting the connection to handle the client
        print("Connection from " + address[0] + " accepted.")
        shared_key, client_id = kyber_key_exchange_server(clientsocket)
        
        encrypt_model(shared_key, "models/main_server_fed_overall.pt", "models/main_server_fed_protected.pt")  # Encrypt model and send it
        connection_handling(clientsocket, address, client_id)
        wait_for_complete_file(f"Pi_models/main_server_fed_{idx}.pt")
        decrypt_model(shared_key, f"Pi_models/main_server_fed_{idx}.pt")
        
        checkpoint = torch.load(f'Pi_models/main_server_fed_{idx}.pt', map_location=torch.device('cpu'))

        # Load the received model into the global model (net_glob)
        net_glob.load_state_dict(checkpoint)
        
        # Prepare the model for local training or updates
        localModel = net_glob.state_dict()

        # Append to local weights (w_locals) based on client configuration
        if args.all_clients:
            w_locals[idx] = copy.deepcopy(localModel)
        else:
            w_locals.append(copy.deepcopy(localModel))

    except Exception as e:
        print(f"Error in client {idx}: {e}")

if __name__ == '__main__':
    
    # parse args
    args = args_parser()
    args.device = torch.device('cpu')
    ################## args def for testing

    NUM_CLIENTS = 2
    NUM_gl_EPOCHS = 2
    PORT = 4045
    SERVER = '10.0.0.51'
    MODELFOLDER = r"C:\\Users\\garrettssh\\Downloads\\Federated-Learning-on-Rasberry-Pi-Senior-Design\\FL-Physical\\Pi_models"

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
    
    # This stores the fixed number of clients shared keys
    shared_keys = ["" for _ in range(NUM_CLIENTS)] # Save this for later because it's not working :(
    # Maybe multithreading is an alternative to using a list?

    threads = []

    for iter in range(args.epochs):
        print("Epoch: ", iter)
        
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        
        m = max(int(args.frac * args.num_users), 1)
        # Comment out when using adaptive dp
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        accuracyList = []
        
        # Handle each client in one loop: distributing and receiving the model
        for idx in range(1, args.num_users + 1):  # Loop trains all the models in the model folder
            
            print("User:", idx)
            thread = threading.Thread(target=handle_client, args=(idx, server, net_glob))
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        # After all models have been processed, update global weights using FedAvg
        if args.global_aggr == 'FedAvg':
            w_glob = FedAvg(w_locals)
        else:
            print('something wrong')
        
        # Copy updated global weights to net_glob
        net_glob.load_state_dict(w_glob)
        
        # Save the model after aggregation
        torch.save(net_glob.state_dict(), "models/main_server_fed_overall.pt")

        # Evaluate the model after training
        net_glob.eval()
        acc_train, l = test_img(net_glob, dataset_train, args)
        training_accuracy_list.append(acc_train)
        training_loss_list.append(l)
        print('Accuracy: ', acc_train)
        print('Loss: ', l)
        print(training_accuracy_list)

        # Remove all previous models for new ones to come in
        modelFolder = MODELFOLDER
        for file in os.scandir(modelFolder):
            os.remove(file)
            
    # Final Connection Handling to terminate clients server exit -> all client
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
