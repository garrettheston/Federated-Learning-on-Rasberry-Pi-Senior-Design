import paramiko
from scp import SCPClient
import socket
import time
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torch import nn
import random

# Custom Dataset for handling MNIST
class CustomDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor[:, :-1].reshape(-1, 1, 28, 28)  # Adjusting for MNIST's 28x28 size and 1 channel
        self.targets = data_tensor[:, -1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Local Update Class
class LocalUpdate(object):
    def __init__(self, args, dataset_train=None, dataset_test=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(dataset_train, batch_size=self.args.local_bs, shuffle=True)
        self.ldr_test = DataLoader(dataset_test, batch_size=args.local_bs, shuffle=False)

    def train(self, net):
        net.train()
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
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            print('Local Epoch {} Finished'.format(iter))
            train_accuracy, train_loss = test(net, self.ldr_train, self.args)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), epoch_loss

# Test function for evaluating the model
def test(net_g, data_loader, args):
    net_g.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
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

# Function to send the model to the server
def SendToServer(server, file="", filepath="", message=""):
    with SCPClient(server.get_transport()) as scp_Client:
        scp_Client.put(file, filepath)

# Arguments for training
class Args:
    def __init__(self):
        self.local_bs = 128
        self.lr = 0.01
        self.momentum = 0.9
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.verbose = True
        self.local_ep = 1
args = Args()

# Read in the config file
f = open("config.txt", "r")
for line in f:
    currentLine = line.strip('\n').split("=")
    if currentLine[0] == 'CLIENT_ID':
        CLIENT_ID = currentLine[1]
    if currentLine[0] == 'SERVER_PORT':
        PORT = int(currentLine[1])
    if currentLine[0] == 'SERVER_IP':
        SERVER = currentLine[1]
    if currentLine[0] == 'SERVER_NAME':
        SERVER_NAME = currentLine[1]
    if currentLine[0] == 'SERVER_PASS':
        SERVER_PASS = currentLine[1]
    if currentLine[0] == 'SERVER_FILE_LOC':
        SERVER_FILE_LOC = currentLine[1]
f.close()

# Create dataset (MNIST)
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

# Split data
total_count = len(dataset)
train_count = int(0.01 * total_count)  # 1% for training
test_count = total_count - train_count
random.seed(42)
torch.manual_seed(42)
dataset_train, dataset_test = random_split(dataset, [train_count, test_count])

# Create an instance of LocalUpdate
test_loader = DataLoader(dataset_test, batch_size=args.local_bs, shuffle=False)
local_update = LocalUpdate(args, dataset_train, dataset_test)

# Define the model architecture (ResNet for MNIST)
net_glob = torchvision.models.resnet18()
net_glob.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # 1 channel for MNIST
net_glob.fc = torch.nn.Linear(net_glob.fc.in_features, 10)  # 10 classes for MNIST
net_glob.to(args.device)

# Start the client connection
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
Searching_connection = True
while Searching_connection:
    try:
        client.connect((SERVER, PORT))
        Searching_connection = False
    except:
        print("Searching for server system...")
        time.sleep(2)

# Handle the connection and receive message
msg = client.recv(1024)
msg_decoded = msg.decode("utf-8")
print(msg_decoded)

if(msg_decoded == "EXIT()"):
    client.send(bytes(f"Client-{CLIENT_ID} terminated", "utf-8"))
    client.close()
    exit()

client.send(bytes("Client received file from server", "utf-8"))
client.close()

# Model Training
time.sleep(5)
# Load model parameters
print("Loading Model Parameters...")
net_glob.load_state_dict(torch.load('main_server_fed.pt'))

# Train the model
print("\nTraining...")
state_dict, avg_loss, lossPerEpoch = local_update.train(net_glob)
print("Training Finished")

# Save the trained model
torch.save(state_dict, f'main_server_fed_{CLIENT_ID}.pt')

# Send model back to the server
server_SSH = paramiko.client.SSHClient()
server_SSH.set_missing_host_key_policy(paramiko.AutoAddPolicy())
server_SSH.connect(SERVER, username=SERVER_NAME, password=SERVER_PASS)
SendToServer(server=server_SSH, file=f"main_server_fed_{CLIENT_ID}.pt", filepath=SERVER_FILE_LOC + f"main_server_fed_{CLIENT_ID}.pt", message="sent file")
