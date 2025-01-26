import os
import socket
import time
import torch
import paramiko
from scp import SCPClient
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Constants and Parameters
MODEL_SAVE_PATH = "./received_model.pt"  # Where the model received from the server will be saved
MODEL_SEND_PATH = "./trained_model.pt"   # Path to save the trained model before sending it back
SERVER_PORT = 4045                       # Example port for socket communication
BUFFER_SIZE = 4096                       # Buffer size for receiving data
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def receive_model(client_socket, save_path=MODEL_SAVE_PATH):
    """Receives the model file from the server."""
    print("Receiving model from server...")
    data = b""
    while True:
        packet = client_socket.recv(BUFFER_SIZE)
        if not packet:  # End of transmission
            break
        data += packet
    
    # Save the received model to the specified path
    with open(save_path, "wb") as model_file:
        model_file.write(data)
    print(f"Model received and saved at {save_path}")

def send_model(server_ip, username, password, local_file, remote_path):
    """Sends the trained model to the server using SCP."""
    print("Sending trained model back to the server...")
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh_client.connect(server_ip, username=username, password=password)
        with SCPClient(ssh_client.get_transport()) as scp:
            scp.put(local_file, remote_path)
        print(f"Model sent to server at {remote_path}")
    except Exception as e:
        print(f"Error sending model to server: {e}")
    finally:
        ssh_client.close()

def train_model(model, train_loader, epochs=1, lr=0.01, momentum=0.9):
    """Trains the model locally."""
    print("Starting local training...")
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    print("Training complete.")
    return model.state_dict()

def main():
    # Connect to the server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            print(f"Connecting to server at {SERVER} on port {PORT}...")
            client_socket.connect((SERVER, PORT))
            print("Connected to server.")
            break
        except Exception as e:
            print("Server not reachable. Retrying in 2 seconds...")
            time.sleep(2)
    
    # Receive the initial model
    receive_model(client_socket)
    client_socket.sendall(b"Model received and saved.")
    client_socket.close()
    
    # Load the received model
    model = torch.load(MODEL_SAVE_PATH)
    model.to(DEVICE)

    # Prepare data (replace with appropriate data preparation logic)
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Train the model
    trained_state_dict = train_model(model, train_loader, epochs=1, lr=0.01, momentum=0.9)
    
    # Save the trained model
    torch.save(trained_state_dict, MODEL_SEND_PATH)
    print(f"Trained model saved at {MODEL_SEND_PATH}")

    # Send the model back to the server
    send_model(SERVER, SERVER_NAME, SERVER_PASS, MODEL_SEND_PATH, SERVER_FILE_LOC + "trained_model.pt")

if __name__ == "__main__":
    # Replace these values with your actual server config
    SERVER = "192.168.1.1"          # Server IP
    PORT = SERVER_PORT              # Server port
    SERVER_NAME = "username"        # SSH username
    SERVER_PASS = "password"        # SSH password
    SERVER_FILE_LOC = "/home/server/models/"  # Path to save the model on the server

    main()
