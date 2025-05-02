import paramiko
import time
from scp import SCPClient

# chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys -- for appropriate permissions

def SendToModelClient(client,clientsocket, file = "",filepath = "",message = ""):

    #try:
    with SCPClient(client.get_transport()) as scp_Client:
        scp_Client.put(file, filepath)

def connection_handling(clientsocket, address, client_id):

    # Load the private key
    private_key_path = r"C:\\Users\\garrettssh2\\.ssh\\id_rsa"
    private_key = paramiko.RSAKey.from_private_key_file(private_key_path)

    if client_id == 1:

        username = 'garrettssh' # Username of windows pc
        password = 'password2'   # password of windows pc

        # set up paramiko ssh client for scp file sending
        SSH_client = paramiko.client.SSHClient()
        SSH_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        SSH_client.connect(address[0], username=username, password=password)
    
        start_time = time.time()
        # Sending main_server_fed_overall.pt to the server and it is receiving main_server_fed.pt which is the equivalent model.
        SendToModelClient(client=SSH_client,clientsocket=clientsocket,file="models/main_server_fed_protected.pt", 
                    #filepath="/home/garrettssh/Federated-Learning-on-Rasberry-Pi-Senior-Design/FL-Physical/main_server_fed.pt", # rasp pi location
                    filepath="C:/Users/garrettssh/Federated-Learning-on-Rasberry-Pi-Senior-Design/FL-Physical/main_server_fed.pt", # Windows loc
                    message="Server:Sent file to client")
        end_time = time.time()
        print(f"Server -> client transmission of model: {(end_time-start_time)*1000}")

    else:
        username = 'garrettssh' # Username for rasp pis
        password = 'password2'   # pasword of raspberry pi 4

        # set up paramiko ssh client for scp file sending
        SSH_client = paramiko.client.SSHClient()
        SSH_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        SSH_client.connect(address[0], username=username, pkey=private_key)

        # Sending main_server_fed_overall.pt to the server and it is receiving main_server_fed.pt which is the equivalent model.
        SendToModelClient(client=SSH_client,clientsocket=clientsocket,file="models/main_server_fed_protected.pt", 
                    filepath="/home/garrettssh/Federated-Learning-on-Rasberry-Pi-Senior-Design/FL-Physical/main_server_fed.pt", # rasp pi location
                    #filepath="C:/Users/garrettssh2/Federated-Learning-on-Rasberry-Pi-Senior-Design/FL-Physical/main_server_fed.pt", # Windows loc
                    message="Server:Sent file to client")
        #time.sleep(3)
    
    SSH_client.close()
