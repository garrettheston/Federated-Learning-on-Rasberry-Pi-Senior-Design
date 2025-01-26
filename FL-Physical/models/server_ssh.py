import pickle
import socket
import os
import torch

def SaveModelToFile(model_data, idx):
    try:
        # Deserialize the pickle data
        model = pickle.loads(model_data)

        # Ensure the Pi_models folder exists
        model_folder = 'Pi_models'
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        
        # Save the model to a .pt file
        model_file_path = os.path.join(model_folder, f"main_server_fed_{idx}.pt")
        torch.save(model, model_file_path)
        print(f"Server: Saved model for client {idx} to {model_file_path}")
    except Exception as e:
        print(f"Error in SaveModelToFile(): {e}")

def SendToClient(clientsocket, file_path="", message=""):
    try:
        # Open the file in binary read mode and send it
        with open(file_path, "rb") as file:
            model_data = file.read()
            pickle_data = pickle.dumps(model_data)
            clientsocket.sendall(pickle_data)  # Send serialized data
            print(f"Server: Sent file '{file_path}' to client.")
        
        # Wait for acknowledgment from the client
        msg = clientsocket.recv(64)
        msg_decoded = msg.decode("utf-8")
        print(f"Client acknowledgment: {msg_decoded}")
    except Exception as e:
        print(f"SendToClient() failed: {e}")

def Connection_handling(clientsocket, address):
    try:
        # Configuration for file paths and communication
        file_path = "models/main_server_fed_overall.pt"  # File to send
        message = "Server: Sent file to client"
        
        print(f"Connection established with client at {address}")
        SendToClient(clientsocket, file_path=file_path, message=message)
        print("Model sent to the client successfully.")
    except Exception as e:
        print(f"Error in Connection_handling(): {e}")

# Sample usage of Connection_handling():
# This is just to demonstrate. In actual implementation, the server would handle socket connections dynamically.
if __name__ == "__main__":
    HOST = "10.0.0.51"  # Server IP
    PORT = 4045         # Server Port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"Server listening on {HOST}:{PORT}...")
        
        clientsocket, address = server_socket.accept()
        with clientsocket:
            Connection_handling(clientsocket, address)


'''  
while !flag
    try
        Connectio:
        flag = true
    except:
        flag = false
'''



'''
Server/client port settings for windows
Control Panel
\->System and Security
    \->Windows Defender Firewall
        \->Advanced Settings
            \->Inbound Rules
                \->New Rule...
                    |ruletype == Port
                    |TCP and Specific local ports: 4045 (or any port you want to use over 1000ish and not reserved for any other communication)
                    |Allow the Connection
                    |Domain Private Public
                    |Name = TCP Port 4045 opening
            \->Outbound Rules
                \->New Rule...
                    |ruletype == Port
                    |TCP and Specific local ports: 4045 (or any port you want to use over 1000ish and not reserved for any other communication)
                    |Allow the Connection
                    |Domain Private Public
                    |Name = TCP Port 4045 opening


'''