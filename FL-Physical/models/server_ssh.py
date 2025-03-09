import paramiko
from scp import SCPClient
import socket
import time
import threading
from kyber_py.ml_kem import ML_KEM_512  # Updated Kyber import
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Protocol.KDF import HKDF
from Crypto.Hash import SHA256
import hashlib

import socket

def kyber_key_exchange_server(client_socket):
    
    public_key, secret_key = ML_KEM_512.keygen()
    
    client_socket.sendall(public_key)
    print("Sent public key")
    
    ciphertext = client_socket.recv(4096)

    print("[SERVER] Here is the ciphertext that is sent to client: ", ciphertext)

    shared_secret = ML_KEM_512.decaps(secret_key, ciphertext)

    print("[SERVER] Established shared secret with client:", shared_secret.hex())
    
    return shared_secret

def encrypt_model(shared_secret, input_file, encrypted_file):
    aes_key = HKDF(master=shared_secret, key_len=32, salt=None, hashmod=SHA256, num_keys=1)

    iv = get_random_bytes(12)  # Generate IV (Nonce)

    with open(input_file, "rb") as f:
        plaintext = f.read()

    cipher = AES.new(aes_key, AES.MODE_GCM, nonce=iv)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)

    print(f"[SERVER] iv: {iv}")
    print(f"[SERVER] tag: {tag}")

    # Save IV + Tag + Ciphertext in one file
    with open(encrypted_file, "wb") as f:
        f.write(iv + tag + ciphertext)

    print("[SERVER] Model encrypted successfully.")

def SendToClient(client, clientsocket, file="", filepath="", message=""):
    try:
        print(f"Starting SendToClient: Sending file {file} to {filepath}")
        
        # Check if file exists
        if not file:
            print("Error: No file specified to send.")
            return
        
        # SCP File Sending
        with SCPClient(client.get_transport()) as scp_Client:
            print(f"Attempting to send file {file} to {filepath} using SCP...")
            scp_Client.put(file, filepath)
            print(f"File {file} sent to {filepath} successfully.")
        
        # Sending a message to the client
        print(f"Sending message to client: {message}")
        clientsocket.send(bytes(message, "utf-8"))
        
        # Receiving response from client
        msg = clientsocket.recv(64)
        msg_decoded = msg.decode("utf-8")
        print(f"Received from client: {msg_decoded}")
    
    except Exception as e:
        print(f"Error in SendToClient: {e}")
        # Uncomment this to raise the error if you want it to propagate
        # raise

def Connection_handling(clientsocket, address):
    print(f"Connection handling started for {address}")
    
    try:
        # Read client credentials from config file
        print("Opening config_server.txt to retrieve credentials...")
        f = open("config_server.txt", "r")
        lineCount = 0
        for line in f:
            currentLine = line.strip('\n').split("=")
            print(f"Processing config line {lineCount}: {currentLine}")
            
            if currentLine[0] == 'CLIENT_USRNM':
                CLIENT_USRNM = currentLine[1]
                print(f"CLIENT_USRNM: {CLIENT_USRNM}")

            if currentLine[0] == 'CLIENT_PSWD':
                CLIENT_PSWD = currentLine[1]
                print(f"CLIENT_PSWD: {CLIENT_PSWD}")

            lineCount += 1
        f.close()
        
        # Set up SSH client for SCP transfer
        print(f"Setting up SSH client to connect to {address[0]} with username {CLIENT_USRNM}")
        SSH_client = paramiko.client.SSHClient()
        SSH_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        SSH_client.connect(address[0], username=CLIENT_USRNM, password=CLIENT_PSWD)
        print(f"SSH client connected to {address[0]}.")

        # This shares the secret key
        #shared_secret = kyber_key_exchange_server(clientsocket)
        #encrypt_model(shared_secret, "models/main_server_fed_overall.pt", "models/main_server_fed_encrypted.pt")

        # Call SendToClient to transfer the model file
        SendToClient(client=SSH_client, clientsocket=clientsocket, 
                     file="models/main_server_fed_overall.pt", #Changed from main_server_fed_encrypted to default 
                     filepath="C:/Users/garrettssh2/Federated-Learning-on-Rasberry-Pi-Senior-Design/FL-Physical/main_server_fedd.pt",
                     message="Server: Sent file to client")

        # Closing SSH client
        print("Closing SSH client connection.")
        SSH_client.close()
    
    except Exception as e:
        print(f"Error in Connection_handling: {e}")
        # Uncomment this to raise the error if you want it to propagate
        # raise

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