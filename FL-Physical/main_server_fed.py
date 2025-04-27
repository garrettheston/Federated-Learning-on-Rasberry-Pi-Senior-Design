import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import socket
import time
import os
import random
import copy
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib
import time
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Fed import FedAvg, calculate_l2_norm
from models.test import test_img
from Connection_Handle import connection_handling
from Crypto_Utils import kyber_key_exchange_server, encrypt_model, decrypt_model, wait_until_file_is_complete
from scp import SCPClient

# Server -> client send a machine learning model
    # This is perfect because I encrypt the model and then it's decrypted when the client receives
    # There is a chance that someone has gained control of one of the many different clients connected to the server.
        # Therefore, after the client receives and decrypts the transmission, there is a decrypted file that exists on the client

        # Introduced ephemeral storage of the model
            # The decryption of the (at the time) encrypted model decrypts into a io buffer which exists only in memory
            # So instead of existing in a .pt format, it exists in ASLR which protects it entirely
            # Python garbage collection deallocates it after it goes out of scope

class FederatedLearningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Federated Learning Server")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Global variables
        self.training_accuracy_list = []
        self.training_loss_list = []
        self.lock = threading.Lock()
        self.server = None
        self.is_running = False
        self.is_paused = False
        self.current_epoch = 0
        self.threads = []
        self.connected_clients = 0
        self.w_locals = []
        self.net_glob = None
        self.dataset_train = None
        self.dataset_test = None
        self.args = None
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create top frame for controls
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Server settings frame
        settings_frame = ttk.Frame(control_frame)
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Server IP and Port
        ttk.Label(settings_frame, text="Server IP:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.server_ip = tk.StringVar(value="10.0.0.51")
        ttk.Entry(settings_frame, textvariable=self.server_ip, width=15).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(settings_frame, text="Port:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.port = tk.IntVar(value=4045)
        ttk.Entry(settings_frame, textvariable=self.port, width=6).grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(settings_frame, text="Num Clients:").grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        self.num_clients = tk.IntVar(value=1)
        ttk.Entry(settings_frame, textvariable=self.num_clients, width=3).grid(row=0, column=5, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(settings_frame, text="Global Epochs:").grid(row=0, column=6, padx=5, pady=5, sticky=tk.W)
        self.num_epochs = tk.IntVar(value=10)
        ttk.Entry(settings_frame, textvariable=self.num_epochs, width=3).grid(row=0, column=7, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(settings_frame, text="Model Folder:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.model_folder = tk.StringVar(value=r"C:\\Users\\garrettssh\\Downloads\\Federated-Learning-on-Rasberry-Pi-Senior-Design\\FL-Physical\\Pi_models")
        ttk.Entry(settings_frame, textvariable=self.model_folder, width=70).grid(row=1, column=1, columnspan=7, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_button = ttk.Button(button_frame, text="Start", command=self.start_server)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.pause_button = ttk.Button(button_frame, text="Pause", command=self.pause_server, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_server, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(main_frame, text="Training Progress")
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_label = ttk.Label(progress_frame, text="Overall Progress: 0%")
        self.progress_label.pack(fill=tk.X, padx=5, pady=2)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        self.epoch_label = ttk.Label(progress_frame, text="Current Epoch: 0/0")
        self.epoch_label.pack(fill=tk.X, padx=5, pady=2)
        
        # Lower section - split for clients and graphs
        lower_frame = ttk.Frame(main_frame)
        lower_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a PanedWindow for resizable sections
        paned = ttk.PanedWindow(lower_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left section - Clients
        client_frame = ttk.LabelFrame(paned, text="Connected Clients")
        paned.add(client_frame, weight=30)
        
        # Clients list
        self.client_list_frame = ttk.Frame(client_frame)
        self.client_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollable client list
        self.client_list = scrolledtext.ScrolledText(self.client_list_frame, wrap=tk.WORD, height=10)
        self.client_list.pack(fill=tk.BOTH, expand=True)
        
        # Right section - Graphs
        graph_frame = ttk.LabelFrame(paned, text="Training Metrics")
        paned.add(graph_frame, weight=70)
        
        # Set up the figures for plotting
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial plot setup
        self.ax1.set_title('Training Accuracy')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Accuracy (%)')
        self.ax1.grid(True)
        
        self.ax2.set_title('Training Loss')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Loss')
        self.ax2.grid(True)
        
        # Log frame
        log_frame = ttk.LabelFrame(main_frame, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Log area
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Update UI
        self.update_ui()
    
    def log(self, message):
        """Add message to log with timestamp"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}\n"
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)
        print(message)  # Also print to console
    
    def update_ui(self):
        """Update UI elements periodically"""
        if self.is_running:
            # Update progress bar
            if self.num_epochs.get() > 0:
                progress = (self.current_epoch / self.num_epochs.get()) * 100
                self.progress_var.set(progress)
                self.progress_label.config(text=f"Overall Progress: {progress:.1f}%")
            
            # Update epoch label
            self.epoch_label.config(text=f"Current Epoch: {self.current_epoch}/{self.num_epochs.get()}")
            
            # Update plots if we have data
            if self.training_accuracy_list:
                self.update_plots()
        
        # Schedule the next update
        self.root.after(1000, self.update_ui)
    
    def update_plots(self):
        """Update the matplotlib plots with current data"""
        self.ax1.clear()
        self.ax2.clear()
        
        epochs = list(range(1, len(self.training_accuracy_list) + 1))
        
        self.ax1.plot(epochs, self.training_accuracy_list, 'b-o')
        self.ax1.set_title('Training Accuracy')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Accuracy (%)')
        self.ax1.grid(True)
        
        self.ax2.plot(epochs, self.training_loss_list, 'r-o')
        self.ax2.set_title('Training Loss')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Loss')
        self.ax2.grid(True)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def add_client(self, address):
        """Add a client to the connected clients list"""
        timestamp = time.strftime('%H:%M:%S')
        client_info = f"[{timestamp}] Client connected: {address[0]}\n"
        
        self.client_list.configure(state=tk.NORMAL)
        self.client_list.insert(tk.END, client_info)
        self.client_list.see(tk.END)
        self.client_list.configure(state=tk.DISABLED)
    
    def start_server(self):
        """Start the federated learning server"""
        if self.is_running:
            return
        
        # Initialize parameters
        self.args = self.initialize_args()
        
        # Start in a separate thread to keep UI responsive
        threading.Thread(target=self.run_server, daemon=True).start()
        
        # Update UI
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL)
        
        self.log("Server starting...")
    
    def initialize_args(self):
        """Initialize arguments similar to your original script"""
        args = args_parser()
        args.device = torch.device('cpu')
        args.num_users = self.num_clients.get()
        args.epochs = self.num_epochs.get()
        args.dataset = 'HAR_LS'
        args.model = 'resnet'
        args.num_channels = 1
        args.bs = 128
        args.all_clients = True  # Based on your original code
        args.frac = 1.0  # Use all clients by default
        args.global_aggr = 'FedAvg'
        return args
    
    def load_dataset(self):
        """Load and prepare the dataset"""
        self.log("Loading dataset...")
        try:
            dataset = torch.load('LS_HAR_data.pt', map_location=torch.device('cpu'), weights_only=False)
            self.log(f"Dataset shape: {dataset.shape}")
            
            # Create custom dataset
            dataset = CustomDataset(dataset)
            total_count = len(dataset)
            print(f"Total count of the dataset: {total_count}")
            train_count = int(0.05*total_count)  # 5%
            test_count = total_count - train_count
            
            # Set random seed for reproducibility
            random.seed(42)
            torch.manual_seed(42)
            
            # Split dataset
            self.dataset_train, self.dataset_test = random_split(dataset, [train_count, test_count])
            self.log(f"Dataset loaded: {train_count} training samples, {test_count} test samples")
            
            return dataset, self.dataset_train, self.dataset_test
        except Exception as e:
            self.log(f"Error loading dataset: {e}")
            return None, None, None
    
    def initialize_model(self):
        """Initialize the global model"""
        self.log("Initializing model...")
        try:
            if self.args.model == 'resnet':
                net_glob = torchvision.models.resnet18()
                net_glob.conv1 = torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
                net_glob.fc = torch.nn.Linear(net_glob.fc.in_features, 5)
                net_glob.to(self.args.device)
            else:
                self.log('Error: unrecognized model')
                return None
            
            # Load pretrained weights if available
            try:
                net_glob.load_state_dict(torch.load("models/main_server_fed_overall.pt", 
                                               map_location=torch.device('cpu'), 
                                               weights_only=False))
                self.log("Loaded existing model weights")
            except:
                self.log("No existing model found, using initialized weights")
            
            net_glob.train()
            return net_glob
        except Exception as e:
            self.log(f"Error initializing model: {e}")
            return None
    
    def handle_client(self, idx, clientsocket, address, net_glob, w_locals):
        """Handle a client connection (similar to your original code)"""
        try:
            self.add_client(address)
            self.log(f"Connection from {address[0]} accepted.")
            
            shared_key, client_id = kyber_key_exchange_server(clientsocket)
            file_path = f"Pi_models/main_server_fed_{client_id}.pt"
                    
            with self.lock:
                encrypt_model(shared_key, "models/main_server_fed_overall.pt", "models/main_server_fed_protected.pt")
                connection_handling(clientsocket, address, client_id)
                
            wait_until_file_is_complete(file_path)
                
            self.log(f"{file_path} is now available and readable!")
            ephemeral_model = decrypt_model(shared_key, file_path)
            self.log(f"Model decrypted successfully!")

            checkpoint = torch.load(ephemeral_model, map_location=torch.device('cpu'))
            
            ephemeral_model = None # Free up the memory space to to prevent scraping and save memory space

            # Load the received model into the global model (net_glob)
            net_glob.load_state_dict(checkpoint)
            self.log(f"Model loaded into global model")
            # Prepare the model for local training or updates
            localModel = net_glob.state_dict()

            # Append to local weights (w_locals) based on client configuration
            if self.args.all_clients:
                w_locals[idx] = copy.deepcopy(localModel)
            else:
                w_locals.append(copy.deepcopy(localModel))
                
            self.log(f"Processed model from client {client_id}")

        except Exception as e:
            self.log(f"Error in client {idx}: {e}")
    
    def run_server(self):
        """Main server function"""
        try:
            # Load dataset
            dataset, self.dataset_train, self.dataset_test = self.load_dataset()
            if dataset is None:
                self.log("Failed to load dataset. Aborting.")
                self.stop_server()
                return
            
            # Initialize model
            self.net_glob = self.initialize_model()
            if self.net_glob is None:
                self.log("Failed to initialize model. Aborting.")
                self.stop_server()
                return
            
            # Copy initial weights
            w_glob = self.net_glob.state_dict()
            
            # Initialize training metrics lists
            self.training_accuracy_list = []
            self.training_loss_list = []
            
            # Initialize w_locals based on all_clients setting
            if self.args.all_clients:
                self.log("Aggregation over all clients")
                self.w_locals = [w_glob for i in range(self.args.num_users)]
            else:
                self.w_locals = []
            
            # Set up server socket
            host = self.server_ip.get()
            port = self.port.get()
            
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.bind((host, port))
            self.server.settimeout(1.0)  # Set a timeout for accepting connections
            self.server.listen(self.args.num_users)
            
            self.log(f"Server started on {host}:{port}")
            self.log(f"Waiting for {self.args.num_users} clients to connect...")
            
            # Main loop for epochs
            for iter in range(self.args.epochs):
                if not self.is_running:
                    break
                    
                self.current_epoch = iter + 1
                self.log(f"\n===== Starting Epoch: {self.current_epoch}/{self.args.epochs} =====")
                
                # Skip loop iterations while paused
                while self.is_paused and self.is_running:
                    time.sleep(0.5)
                
                if not self.args.all_clients:
                    self.w_locals = []
                
                m = max(int(self.args.frac * self.args.num_users), 1)
                # Comment out when using adaptive dp
                idxs_users = np.random.choice(range(self.args.num_users), m, replace=False)
                
                self.connected_clients = 0
                self.threads = []

                # Accept clients for this epoch
                while self.connected_clients < self.args.num_users and self.is_running:
                    try:
                        clientsocket, address = self.server.accept()
                        
                        # Handle client in a separate thread
                        thread = threading.Thread(target=self.handle_client, 
                                                 args=(self.connected_clients, clientsocket, address, 
                                                       self.net_glob, self.w_locals))
                        self.threads.append(thread)
                        thread.start()
                        
                        self.connected_clients += 1
                        self.log(f"Client {self.connected_clients}/{self.args.num_users} connected")
                        
                    except socket.timeout:
                        # Just continue on timeout
                        continue
                    except Exception as e:
                        self.log(f"Error accepting client: {e}")
                
                # Wait for all client threads to finish
                for thread in self.threads:
                    thread.join()
                            
                # Calculate L2 norms for the updates
                l2_norms = calculate_l2_norm(self.w_locals)
                
                # Compute the mean and std of L2 norms
                mean_l2_norm = np.mean(l2_norms)
                std_l2_norm = np.std(l2_norms)
                
                # Set the threshold (2 standard deviations)
                threshold = mean_l2_norm + 2 * std_l2_norm
                
                print(f"Threshold for anomaly detection: {threshold}")
                
                # Check for anomalous updates (those that exceed the threshold)
                for idx, l2_norm in enumerate(l2_norms):
                    if l2_norm > threshold:
                        print(f"Anomalous model update detected from client {idx+1} with L2 norm: {l2_norm}")
                    else:
                        print(f"Model update from client {idx+1} is normal with L2 norm: {l2_norm}")

                # Federated weight aggregation
                if self.is_running:
                    if self.args.global_aggr == 'FedAvg':
                        self.log("Aggregating models using FedAvg...")
                        w_glob = FedAvg(self.w_locals)
                    else:
                        self.log('Unrecognized aggregation method')
                    
                    # Update global model
                    self.net_glob.load_state_dict(w_glob)
                    
                    # Save the aggregated model
                    torch.save(self.net_glob.state_dict(), "models/main_server_fed_overall.pt")
                    self.log("Saved aggregated model")
                    
                    # Evaluate model
                    self.net_glob.eval()
                    acc_train, l = test_img(self.net_glob, self.dataset_train, self.args)
                    self.training_accuracy_list.append(acc_train)
                    self.training_loss_list.append(l)
                    self.log(f'Training Accuracy: {acc_train:.2f}%')
                    self.log(f'Training Loss: {l:.4f}')
                    
                    # Clean up model folder for next epoch
                    model_folder = self.model_folder.get()
                    for file in os.scandir(model_folder):
                        os.remove(file.path)
                    self.log(f"Cleared model folder for next epoch")
            
            # Final evaluation
            if self.is_running:
                self.log("\n===== Training Complete =====")
                self.net_glob.eval()
                acc_test, loss_test = test_img(self.net_glob, self.dataset_test, self.args)
                self.log(f"Final Test Accuracy: {acc_test:.2f}%")
                self.log(f"Final Test Loss: {loss_test:.4f}")
                
                # Send exit signal to all clients
                self.log("Sending exit signal to all clients...")
                for idx in range(self.args.num_users):
                    try:
                        clientsocket, address = self.server.accept() 
                        self.log(f"Sending exit signal to {address[0]}")
                        clientsocket.send(bytes("EXIT()", "utf-8"))    
                        msg = clientsocket.recv(64)
                        msg_decoded = msg.decode("utf-8")
                        self.log(msg_decoded)
                    except Exception as e:
                        self.log(f"Error sending exit signal: {e}")
            
            # Close server
            if self.server:
                self.server.close()
                self.server = None
            
            self.log("Server closed")
            
        except Exception as e:
            self.log(f"Error in server execution: {e}")
        finally:
            # Reset UI state
            self.is_running = False
            self.root.after(0, self.reset_ui)
    
    def pause_server(self):
        """Pause the server"""
        if not self.is_running:
            return
        
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_button.config(text="Resume")
            self.log("Server paused")
        else:
            self.pause_button.config(text="Pause")
            self.log("Server resumed")
    
    def stop_server(self):
        """Stop the server"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.is_paused = False
        self.log("Stopping server...")
        
        # Close socket
        if self.server:
            try:
                self.server.close()
                self.server = None
            except:
                pass
        
        # Reset UI
        self.reset_ui()
    
    def reset_ui(self):
        """Reset UI elements to default state"""
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED, text="Pause")
        self.stop_button.config(state=tk.DISABLED)
        
        self.progress_var.set(0)
        self.progress_label.config(text="Overall Progress: 0%")
        self.epoch_label.config(text=f"Current Epoch: 0/0")
        
        # Clear client list
        self.client_list.configure(state=tk.NORMAL)
        self.client_list.delete(1.0, tk.END)
        self.client_list.configure(state=tk.DISABLED)
    
    def on_closing(self):
        """Handle window close event"""
        if self.is_running:
            if messagebox.askokcancel("Quit", "Server is running. Do you want to stop it and quit?"):
                self.stop_server()
                self.root.destroy()
        else:
            self.root.destroy()


# CustomDataset class from your original code
class CustomDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor[:, :-1].reshape(-1, 1, 9, 100)  # [batch_size, channels, height, width]
        self.targets = data_tensor[:, -1]
        print(self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


if __name__ == "__main__":
    root = tk.Tk()
    app = FederatedLearningGUI(root)
    root.mainloop()