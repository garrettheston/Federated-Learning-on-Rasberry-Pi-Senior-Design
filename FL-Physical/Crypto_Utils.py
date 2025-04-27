from kyber_py.ml_kem import ML_KEM_512  # Updated Kyber import
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Protocol.KDF import HKDF
from Crypto.Hash import SHA256
import hashlib
import time
import os
import io

# After I receive the machine learning model, it exists in a file with the extension .pt after it
    # Representing the machine learning model as a file.
# If someone where to penetrate my machine, they could steal, exfihltrate, or manipulate the model.
# Therefore, I implemented something known as ephemeral storage
    # Basically, the model after it's decrypted it's written exclusively to memory (RAM)
    # And python uses ASLR which randomizes memory locations
    # So basically, the model doesn't exist plainly to been because it's now in a memory buffer that you can't access

def kyber_key_exchange_server(clientsocket):

    public_key, secret_key = ML_KEM_512.keygen()
    
    # Prepend "EXCHANGE:" label to the public key
    label = b"EXCHANGE:"  # Make sure this is in bytes
    public_key_with_label = label + public_key
    # Send the public key with the label to the client
    start_time = time.time()
    clientsocket.sendall(public_key_with_label)
    end_time = time.time()
    print(f"Transmission of public key to client: {(end_time-start_time) * 1000}")
    data = clientsocket.recv(4096)
    client_id = data[0]
    ciphertext = data[1:]
    shared_secret = ML_KEM_512.decaps(secret_key, ciphertext)
    print("[SERVER] Established shared secret with client:", shared_secret.hex())
    
    return shared_secret, client_id

def encrypt_model(shared_key, input_file, output_file):

    aes_key = HKDF(master=shared_key, key_len=32, salt=None, hashmod=SHA256, num_keys=1)
    print("[SERVER] Encryption begins for model")
    iv = get_random_bytes(16)  # Generate IV (Nonce)
    with open(input_file, "rb") as f:
        plaintext = f.read()
    cipher = AES.new(aes_key, AES.MODE_OFB, iv=iv)
    ciphertext = cipher.encrypt(plaintext)
    print(f"[SERVER] iv: {iv}")
    data_to_send = iv + ciphertext
    # Save IV + Tag + Ciphertext in one file
    with open(output_file, "wb") as f:
        f.write(data_to_send)
    print("[SERVER] Model encrypted successfully.")

def decrypt_model(secret,provided):
    aes_key = HKDF(master = secret, key_len=32, salt=None, hashmod=SHA256, num_keys=1)

    print(f"Currently decrypting {provided}")

    with open(provided, 'rb') as f:
        data = f.read()
    iv, ciphertext = data[:16], data[16:]
    print(f"[SERVER] received IV for decryption: {iv}")
    cipher = AES.new(aes_key, AES.MODE_OFB, iv=iv)
    plaintext = cipher.decrypt(ciphertext)
    print(f"[SERVER] Model decryption successful. Finished writing this to an ephemeral buffer for security measures.")
    
    return io.BytesIO(plaintext)
    

# STANDARD UTIL
def wait_until_file_is_complete(file_path, stable_time=2.0, check_interval=0.5):
    last_size = -1
    same_size_count = 0

    try:
        while True:
            if os.path.exists(file_path) and os.access(file_path, os.R_OK):
                current_size = os.path.getsize(file_path)
                if current_size == last_size:
                    same_size_count += check_interval
                    if same_size_count >= stable_time:
                        print(f"File {file_path} is now stable and readable.")
                        break
                else:
                    same_size_count = 0
                    last_size = current_size
            else:
                same_size_count = 0  # Reset if file disappears or unreadable

            print(f"Waiting for file {file_path} to stabilize...")
            time.sleep(check_interval)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting wait loop.")