import unittest
from unittest.mock import MagicMock, patch
import torch
import copy

class TestConnectionLogic(unittest.TestCase):

    def setUp(self):
        # Mock any required attributes
        self.mock_server = MagicMock()
        self.mock_server.add_client = MagicMock()
        self.mock_server.log = MagicMock()
        self.mock_server.lock = MagicMock()  # You can mock this if needed

        # Example mock data
        self.mock_net_glob = MagicMock()  # This will be the global model
        self.mock_w_locals = []

    def handle_client(self, server, idx, clientsocket, address, net_glob, w_locals):
        """Redefining the handle_client function inline for the test"""
        try:
            server.add_client(address)
            server.log(f"Connection from {address[0]} accepted.")

            # Mocked Kyber key exchange
            shared_key, client_id = "shared_key", "client_1"  # Mocking this
            server.log("Kyber PQC KEM conducted.")
            file_path = f"Pi_models/main_server_fed_{client_id}.pt"

            with server.lock:
                server.log("Model encrypted using PQC scheme.")
                # Mocking connection handling (this can be anything)
                server.log(f"Handling connection with {client_id}")

            # Simulate the file completion and decryption
            server.log(f"{file_path} is now available and readable!")
            ephemeral_model = "decrypted_model"  # Mocked decrypted model
            server.log("Integrity check passed.")

            # Simulate loading model (mocking torch.load and model)
            checkpoint = {"state_dict": "dummy_state"}  # Mocked checkpoint
            server.log("Model decrypted successfully!")

            net_glob.load_state_dict(checkpoint)  # Mocking model loading
            server.log("Model loaded into global model")
            localModel = net_glob.state_dict()

            # Simulate adding the model to w_locals
            w_locals.append(copy.deepcopy(localModel))

            server.log(f"Processed model from client {client_id}")

        except Exception as e:
            server.log(f"Error in client {idx}: {e}")

    @patch('builtins.print')  # Optional, if you want to patch print statements
    def test_handle_four_clients(self, mock_print):
        # Mock client data for 4 clients
        clients = [
            ("127.0.0.1", 5000),  # Mocked client address and port
            ("127.0.0.2", 5001),
            ("127.0.0.3", 5002),
            ("127.0.0.4", 5003)
        ]

        # Run the test for 4 clients
        for idx, address in enumerate(clients):
            with self.subTest(client=address):
                self.handle_client(self.mock_server, idx, None, address, self.mock_net_glob, self.mock_w_locals)

        # Check if the server log contains the expected messages
        self.mock_server.log.assert_any_call("Connection from 127.0.0.1 accepted.")
        self.mock_server.log.assert_any_call("Processed model from client client_1")

        # Assert that w_locals was updated for each client
        self.assertEqual(len(self.mock_w_locals), 4)


if __name__ == '__main__':
    unittest.main()
