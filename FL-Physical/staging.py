import paramiko
from scp import SCPClient
import socket
import time
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import torchvision
from torch.utils.data import Dataset,DataLoader, random_split
import torch.nn.functional as F
from torch import nn
from kyber_py.ml_kem import ML_KEM_512
import random

def test(net_g, data_loader, args):
    # testing
    net_g.eval()
    loss = 0
    correct = 0
    with torch.no_grad():  # No need to track gradients during inference
        for data, target in data_loader:
            data, target = data.to(args.device), target.to(args.device)
            log_probs = net_g(data)
            loss += F.cross_entropy(log_probs, target.long()).item()
            y_pred = log_probs.argmax(dim=1)
            correct += y_pred.eq(target).sum().item()

    loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset) * 100
    print('Metrics: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        loss, correct, len(data_loader.dataset), accuracy))

    return accuracy, loss
    
def traffic_handling(server_socket, client_id):
   
    # Receive the public key with the label
    public_key_with_label = server_socket.recv(4096)
   
    # Check for the "EXCHANGE:" label
    label = b"EXCHANGE:"  # The expected label from the server
   
    if public_key_with_label.startswith(label):
        # Extract the actual public key (remove the label)
        public_key = public_key_with_label[len(label):]
        print("[CLIENT] Public key received successfully.")
       
        try:
            # Encapsulate shared secret
            shared_secret, ciphertext = ML_KEM_512.encaps(public_key)
            print("[CLIENT] Here is the ciphertext: ", ciphertext)
            print("[CLIENT] Here is the shared secret key derived from the server", shared_secret.hex())

            data = client_id.to_bytes(1,byteorder='big') + ciphertext
            # Send ciphertext
            server_socket.sendall(data)
           
            return shared_secret

        except Exception as e:
            print(f"[CLIENT] Error during key exchange: {e}")
            return

    else:
        # If the label doesn't match, check for an "EXIT()" message
        msg_decoded = public_key_with_label.decode()  # Decode the received message
        if msg_decoded == "EXIT()":
            print("[CLIENT] Received exit message.")
            server_socket.send(bytes("Client terminated", "utf-8"))
            server_socket.close()
            exit()
        else:
            print("[CLIENT] Error: Public key does not start with the expected label or invalid message.")
            server_socket.close()
            exit()

    print("[CLIENT] created shared secret")

    return

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class CustomDataset(Dataset):
    def __init__(self, data_tensor):
        #self.data = data_tensor[:, :-1]
        self.data = data_tensor[:, :-1].reshape(-1,1, 9, 100) #[batch_size, channels, height, width]
        self.targets = data_tensor[:, -1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class LocalUpdate(object):
    def __init__(self, args, dataset_train=None,dataset_test=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(dataset_train, batch_size=self.args.local_bs, shuffle=True) #local_bs =1 for HAR_LS for now
        self.ldr_test = DataLoader(dataset_test, batch_size=args.local_bs, shuffle=False)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                labels = labels.float()
                images = images.float()
                net.zero_grad()
                log_probs = net(images)
                labels = labels.long()
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                #if self.args.verbose and batch_idx % 2 == 0:
                 #   print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  #      iter, batch_idx * len(images), len(self.ldr_train.dataset),
                   #            100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            print('Local Epoch {} Finished'.format(iter))
            train_accuracy, train_loss = test(net, self.ldr_train, self.args)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), epoch_loss

def SendToServer(server, file = "",filepath = "",message = ""):
 #   try:
    with SCPClient(server.get_transport()) as scp_Client:
        scp_Client.put(file, filepath)

    #serverSocket.send(bytes(message, "utf-8"))
#    except:
        #print("SentToServer() Failed.")

Searching_connection = True
PORT = 4045
#SERVER = "10.4.159.106"
#SERVER = "10.4.148.119"

#Create dataset
dataset = torch.load('LS_HAR_data.pt').float()
dataset = CustomDataset(dataset)

#Split data
total_count = len(dataset)
train_count = int(0.01*total_count) # 5%
test_count = total_count - train_count # 
random.seed(42)
torch.manual_seed(42)
dataset_train, dataset_test = random_split(dataset, [train_count, test_count])

#Defining arguments
class Args:
    def __init__(self):
        self.local_bs = 128
        self.lr = 0.01
        self.momentum = 0.9
        self.device = 'cpu'
        self.verbose = True
        self.local_ep = 1
        #self.client_id = 1
        #self.ServerName = "ServerUsername"
        #self.ServerPassword = "ServerPassword"
args = Args()

# Read in the config file
f = open("config.txt", "r")

lineCount = 0
for line in f:
    currentLine = line.strip('\n').split("=")
    print(currentLine)
    
    if currentLine[0] == 'CLIENT_ID':
        CLIENT_ID = currentLine[1]
        #CLIENT_ID = int(CLIENT_ID)
        print(currentLine[1])
    if currentLine[0] == 'SERVER_PORT':
        PORT = currentLine[1]
        PORT = int(PORT)
        print(currentLine[1])
    if currentLine[0] == 'SERVER_IP':
        SERVER = currentLine[1]
        print(SERVER)
    if currentLine[0] == 'SERVER_NAME':
        SERVER_NAME = currentLine[1]
        print(currentLine[1])
    if currentLine[0] == 'SERVER_PASS':
        SERVER_PASS = currentLine[1]
        print(currentLine[1])
    if currentLine[0] == 'SERVER_FILE_LOC':
        SERVER_FILE_LOC = currentLine[1]
        print(currentLine[1])
    lineCount += 1

f.close()

while True:


    test_loader = DataLoader(dataset_test, batch_size=args.local_bs, shuffle=False)
    torch.manual_seed(56)
    
    # Create an instance of LocalUpdate
    local_update = LocalUpdate(args, dataset_train, dataset_test)

    #Create an instance of model architecture
    net_glob = torchvision.models.resnet18()
    net_glob.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) #args.num_channels,num out channels, all others labeled
    net_glob.fc = torch.nn.Linear(net_glob.fc.in_features, 5) #5 = num features for HAR_LS
    net_glob.to(args.device)
##

    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    Searching_connection = True
    while Searching_connection:
        try:
            #connect here ##############
            #PORT = 4045          
            #SERVER = "10.4.130.19"
            serversocket.connect((SERVER, PORT))
            Searching_connection = False

        except:
            print("Searching for server system...")
            Searching_connection = True
            time.sleep(2)
    
    ## Handle Connection
    # here is were we have communication with a socket back and forth
    shared_secret = traffic_handling(serversocket, int(CLIENT_ID))
    
    ## Model Training
    time.sleep(15)
    # Load the model dictionary/parameters
    print("Loading Model Parameters...")
    net_glob.load_state_dict(torch.load('main_server_fed.pt', map_location=torch.device('cpu')))
    # Call training function
    print("\nTraining...")
    state_dict, avg_loss, lossPerEpoch = local_update.train(net_glob)
    print("Training Finished")
    # Save the model dictionary/parameters
    torch.save(state_dict, 'main_server_fed_'+CLIENT_ID+'.pt')

    ## Send Model
    # Here is only sending the model back
    # no sockets
    username = SERVER_NAME  # username of central server
    password = SERVER_PASS  # password of central server
    file_path = SERVER_FILE_LOC
        

    server_SSH = paramiko.client.SSHClient()
    server_SSH.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    server_SSH.connect(SERVER, username=username, password=password)
    SendToServer(server=server_SSH,file="main_server_fed_"+CLIENT_ID+".pt",
                filepath="C:/Users/garrettssh/Downloads/Federated-Learning-on-Rasberry-Pi-Senior-Design/FL-Physical/Pi_models/main_server_fed_"+CLIENT_ID+".pt",
                message="sent file")










