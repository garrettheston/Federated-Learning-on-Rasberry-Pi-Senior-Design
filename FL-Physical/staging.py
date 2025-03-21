#includes
import paramiko
from scp import SCPClient
import socket
import time
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import torchvision
from torch.utils.data import Dataset,DataLoader, random_split
import torch.nn.functional as F
from torch import nn
import random
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Protocol.KDF import HKDF
from Crypto.Hash import SHA256
import hashlib
import os
from kyber_py.ml_kem import ML_KEM_512 # updated kyber import

# In the future this infrastructure could be facilitated via pickles
def kyber_key_exchange_client(server_socket):
    
    # Receive public key
    public_key = b""
    
    public_key = server_socket.recv(4096)
        
    print("[CLIENT] Public key received successfully.")

    try: 
        
        # Encapsulate shared secret
        shared_secret, ciphertext = ML_KEM_512.encaps(public_key)
        print("[CLIENT] Here is the ciphertext: ", ciphertext)
        print("[CLIENT] Here is the shared secret key derived from the server", shared_secret.hex())

        # Send ciphertext
        server_socket.sendall(ciphertext)
        
        return shared_secret

    except Exception:
        msg_decoded = public_key.decode()
        if msg_decoded == "Exit()":
            print("[CLIENT] Received exit message.")
            server_socket.send(bytes("Client terminated", "utf-8"))
            server_socket.close()
            exit()

    print("[CLIENT] created shared secret")

    return

def decrypt_model(shared_secret):
    aes_key = HKDF(master=shared_secret, key_len=32, salt=None, hashmod=SHA256, num_keys=1)

    #wait_for_file("main_server_fed_encrypted.pt")

    print(f"[CLIENT] Model is beginning decryption")

    with open("main_server_fed_encrypted.pt", "rb") as f:
        data = f.read()

    sha256, iv, ciphertext = data[:32], data[32:48], data[48:]  # Extract components and hash

    print(f"[CLIENT] received IV: {iv}")

    cipher = AES.new(aes_key, AES.MODE_OFB, iv=iv)
    plaintext = cipher.decrypt(ciphertext)

    with open("main_server_fedd.pt", "wb") as f:
        f.write(plaintext)

    print("[CLIENT] Model decrypted successfully.")
    
    return sha256

def encrypt_model(shared_secret,input_file,encrypted_file, sha256):
    aes_key = HKDF(master=shared_secret, key_len=32, salt=None, hashmod=SHA256, num_keys=1)

    iv = get_random_bytes(16)

    with open(input_file, "rb") as f:
        plaintext = f.read()
    
    cipher = AES.new(aes_key, AES.MODE_OFB, iv=iv)
    ciphertext = cipher.encrypt(plaintext)

    print(f"[SERVER] iv: {iv}")

    data_to_send = sha256.digest() + iv + ciphertext

    with open(encrypted_file, "wb") as f:
        f.write(data_to_send)

    print("[SERVER] Model encrypted successfully.")

def wait_for_file(filename, timeout=10):
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(filename) and os.access(filename, os.R_OK):
            print("Access is true")
            return True  # File exists and is readable
        print(f"Waiting for {filename} to become accessible...")
        time.sleep(0.5)  # Wait 500ms before checking again
    raise TimeoutError(f"File {filename} is not accessible after {timeout} seconds.")

def hash_file(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(4096):
            sha256.update(chunk)
    return sha256

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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

shared_secret = 0

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

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    Searching_connection = True
    while Searching_connection:
        try:
            #connect here ##############
            #PORT = 4045          
            #SERVER = "10.4.130.19"
            client.connect((SERVER, PORT))
            Searching_connection = False

        except:
            print("Searching for server system...")
            Searching_connection = True
            time.sleep(2)
    
    ## Handle Connection
    # here is were we have communication with a socket back and forth
    # general key exchange handler but also handles messages
    shared_secret = kyber_key_exchange_client(client)

    client.send(bytes("Client recieved file from sever","utf-8"))
    client.close()
    # we close socket here
    
    ## Model Training
    time.sleep(20) # This is necessary because if client opens the file before receiving you have a massive problem
    
    server_sha256 = decrypt_model(shared_secret)    

    time.sleep(5)

    client_sha256 = hash_file("main_server_fedd.pt") # generate hash with decrypted model and then compare

    if server_sha256 == client_sha256.digest():
        print("[CLIENT] passed model update")
    else:
        print("[CLIENT] security incident: hash failed")

    # Load the model dictionary/parameters
    print("Loading Model Parameters...")
    net_glob.load_state_dict(torch.load('main_server_fedd.pt'))
    # Call training function
    print("\nTraining...")
    state_dict, avg_loss, lossPerEpoch = local_update.train(net_glob)
    print("Training Finished")
    # Save the model dictionary/parameters
    torch.save(state_dict, 'main_server_fed_'+CLIENT_ID+'.pt')

    time.sleep(5)

    client_based_sha256 = hash_file("main_server_fed_"+CLIENT_ID+".pt")

    encrypt_model(shared_secret, 'main_server_fed_'+CLIENT_ID+'.pt', 'main_server_fed_'+CLIENT_ID+'_protected.pt', client_based_sha256)

    time.sleep(2)

    # Define your server credentials and file path
    username = SERVER_NAME  # username of central server
    password = SERVER_PASS  # password of central server
    file_path = SERVER_FILE_LOC

    # Create the SSH client and set the policy
    print("Initializing SSH client...")
    server_SSH = paramiko.client.SSHClient()
    server_SSH.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        print(f"Connecting to the server {SERVER}...")
        # Try connecting to the server
        server_SSH.connect(SERVER, username=username, password=password)
        print(f"Successfully connected to {SERVER}.")
        
        # If connected, proceed with sending the file
        print(f"Preparing to send file: main_server_fed_{CLIENT_ID}.pt")
        SendToServer(server=server_SSH,
                    file="main_server_fed_" + CLIENT_ID + "_protected.pt",
                    filepath=file_path + "main_server_fed_" + CLIENT_ID + "_protected.pt",
                    message="sent file")
        print(f"File sent successfully to {SERVER}.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        server_SSH.close()
        print("SSH connection closed.")
