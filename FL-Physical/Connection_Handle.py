import paramiko
from scp import SCPClient

def SendToModelClient(client,clientsocket, file = "",filepath = "",message = ""):
    
    # Kyber key exchange needs to occur individually with all of the clients
    #public_key, secret_key = ML_KEM_512.keygen()
    #ciphertext = client_socket.recv(4096)

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
    
    #username = 'garrettssh2'   # username of raspberry pi 4
    username = 'garrettssh'
    password = 'password2'   # pasword of raspberry pi 4

    # set up paramiko ssh client for scp file sending
    SSH_client = paramiko.client.SSHClient()
    SSH_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    SSH_client.connect(address[0], username=username, password=password)

    # Sending main_server_fed_overall.pt to the server and it is receiving main_server_fed.pt which is the equivalent model.
    SendToModelClient(client=SSH_client,clientsocket=clientsocket,file="models/main_server_fed_overall.pt", 
                 filepath="/home/garrettssh/Federated-Learning-on-Rasberry-Pi-Senior-Design/FL-Physical/main_server_fed.pt",
                 #filepath="C:/Users/garrettssh2/Federated-Learning-on-Rasberry-Pi-Senior-Design/FL-Physical/main_server_fed.pt",
                 message="Server:Sent file to client")
    #time.sleep(3)
    
    SSH_client.close()