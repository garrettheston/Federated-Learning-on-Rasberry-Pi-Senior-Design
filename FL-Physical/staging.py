import os
import socket
import time
import torch
import pickle
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Constants and Parameters
MODEL_SAVE_PATH = "./received_model.pkl"  # Where the model received from the server will be saved
MODEL_SEND_PATH = "./trained_model.pkl"   # Path to save the trained model before sending it back
SERVER_PORT = 4045                        # Example port for socket communication
BUFFER_SIZE = 4096                        # Buffer size for receiving data
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def receive_model(client_socket, save_path=MODEL_SAVE_PATH):
    """Receives the pickled model file from the server and unpickles it."""
    print("Receiving model from server...")
    data = b""
    while True:
        packet = client_socket.recv(BUFFER_SIZE)
        if not packet:  # End of transmission
            break
        data += packet
    
    # Unpickle the model
    model = pickle.loads(data)
    # Save the model as a pickle file
    with open(save_path, "wb") as model_file:
        pickle.dump(model, model_file)
    
    print(f"Model received and saved at {save_path}")
    return model

def send_model(client_socket, model, buffer_size=BUFFER_SIZE):
    """Sends the pickled model back to the server over the TCP socket."""
    print("Sending trained model back to the server...")
    
    # Pickle the model
    model_data = pickle.dumps(model)
    
    # Send the pickled model in chunks
    for i in range(0, len(model_data), buffer_size):
        client_socket.send(model_data[i:i+buffer_size])
    
    print(f"Trained model sent back to the server.")

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
    return model

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
    
    # Receive the initial model (as a pickled file)
    model = receive_model(client_socket)
    client_socket.sendall(b"Model received and saved.")
    
    # Load the received model
    model.to(DEVICE)

    # Prepare data (replace with appropriate data preparation logic)
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Train the model
    trained_model = train_model(model, train_loader, epochs=1, lr=0.01, momentum=0.9)
    
    # Send the trained model back to the server via TCP
    send_model(client_socket, trained_model)

    client_socket.close()

if __name__ == "__main__":
    # Replace these values with your actual server config
    SERVER = "192.168.1.1"          # Server IP
    PORT = SERVER_PORT              # Server port

    main()
