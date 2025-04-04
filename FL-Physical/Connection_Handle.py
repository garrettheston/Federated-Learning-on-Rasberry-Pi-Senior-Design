import paramiko
from scp import SCPClient

def SendToModelClient(client,clientsocket, file = "",filepath = "",message = ""):

    #try:
    with SCPClient(client.get_transport()) as scp_Client:
        scp_Client.put(file, filepath)

def connection_handling(clientsocket, address):
    
    #username = 'garrettssh2'   # username of raspberry pi 4
    username = 'garrettssh2'
    password = 'password2'   # pasword of raspberry pi 4

    # set up paramiko ssh client for scp file sending
    SSH_client = paramiko.client.SSHClient()
    SSH_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    SSH_client.connect(address[0], username=username, password=password)

    # Sending main_server_fed_overall.pt to the server and it is receiving main_server_fed.pt which is the equivalent model.
    SendToModelClient(client=SSH_client,clientsocket=clientsocket,file="models/main_server_fed_protected.pt", 
                 #filepath="/home/garrettssh/Federated-Learning-on-Rasberry-Pi-Senior-Design/FL-Physical/main_server_fed.pt", # rasp pi location
                 filepath="C:/Users/garrettssh2/Federated-Learning-on-Rasberry-Pi-Senior-Design/FL-Physical/main_server_fed.pt", # Windows loc
                 message="Server:Sent file to client")
    #time.sleep(3)
    
    SSH_client.close()