import paramiko
from scp import SCPClient
import socket
import time
import threading

def SendToClient(client,clientsocket, file = "",filepath = "",message = ""):
    #try:
    with SCPClient(client.get_transport()) as scp_Client:
        scp_Client.put(file, filepath)

    clientsocket.send(bytes(message, "utf-8"))
    msg = clientsocket.recv(64)
    msg_decoded = msg.decode("utf-8")
    print(msg_decoded)
    #except:
    #    print("SentToClient() Failed.")

def Connection_handling(clientsocket, address):
    #time.sleep(5)
    f = open("config_server.txt", "r")
    lineCount = 0
    for line in f:
        currentLine = line.strip('\n').split("=")

        if currentLine[0] == 'CLIENT_USRNM':
            CLIENT_USRNM = currentLine[1]       

        if currentLine[0] == 'CLIENT_PSWD':
            CLIENT_PSWD = currentLine[1]  
        
        lineCount += 1

    f.close()
    username = CLIENT_USRNM   # username of raspberry pi 4
    password = CLIENT_PSWD   # pasword of raspberry pi 4

    # set up paramiko ssh client for scp file sending
    SSH_client = paramiko.client.SSHClient()
    SSH_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    SSH_client.connect(address[0], username=username, password=password)

    # Sending main_server_fed_overall.pt to the server and it is receiving main_server_fed.pt which is the equivalent model.
    SendToClient(client=SSH_client,clientsocket=clientsocket,file="models/main_server_fed_overall.pt", 
                 filepath="C:/Users/garrettssh2/Federated-Learning-on-Rasberry-Pi-Senior-Design/FL-Physical/main_server_fed.pt",
                 message="Server:Sent file to client")
    #time.sleep(3)
    
    SSH_client.close()