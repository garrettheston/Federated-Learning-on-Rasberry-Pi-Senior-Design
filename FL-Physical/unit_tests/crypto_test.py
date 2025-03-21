
 # The NIST Module-Lattice-Based Key-Encapsulation Mechanism Standard ML-KEM 
from kyber_py.ml_kem import ML_KEM_512
 
# The AES algorithm for the keys actually used after the KEM
import unittest
from kyber_py.ml_kem import ML_KEM_512  # NIST Module-Lattice-Based KEM
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
 
# Import pickle for data serialization
import pickle
 
class TestCryptosystem(unittest.TestCase):
 
     def setUp(self):
         """Set up keys and data for the cryptosystem tests."""
         self.ek, self.dk = ML_KEM_512.keygen()  # Generate keys
         self.sks, self.ct = ML_KEM_512.encaps(self.ek)  # Encapsulate session key
         self.skr = ML_KEM_512.decaps(self.dk, self.ct)  # Decapsulate session key
         
         self.iv = get_random_bytes(16)  # Initialization vector for AES
         self.sample_data = "Sample model data provided for federated learning."
 
     def test_key_equality(self):
         """Test that encapsulated and decapsulated keys are identical."""
         self.assertEqual(self.sks, self.skr, "Cryptosystem key mismatch: Encapsulated and decapsulated keys do not match.")
 
     def test_encryption_decryption(self):
         """Test that serialized data matches after encryption and decryption."""
         # Encrypt data
         cipher_enc = AES.new(self.sks[:32], AES.MODE_OFB, self.iv)
         serialized_data = pickle.dumps(self.sample_data)
         ciphertext = cipher_enc.encrypt(serialized_data)
 
         # Decrypt data
         cipher_dec = AES.new(self.skr[:32], AES.MODE_OFB, self.iv)
         decrypted_data = cipher_dec.decrypt(ciphertext)
 
         # Check if serialized data matches
         self.assertEqual(serialized_data, decrypted_data, "Decryption failed: Serialized data does not match.")
 
     def test_serialization_deserialization(self):
         """Test that original data matches deserialized data after encryption and decryption."""
         # Encrypt data
         cipher_enc = AES.new(self.sks[:32], AES.MODE_OFB, self.iv)
         serialized_data = pickle.dumps(self.sample_data)
         ciphertext = cipher_enc.encrypt(serialized_data)
 
         # Decrypt data
         cipher_dec = AES.new(self.skr[:32], AES.MODE_OFB, self.iv)
         decrypted_data = cipher_dec.decrypt(ciphertext)
 
         # Deserialize data
         deserialized_data = pickle.loads(decrypted_data)
 
         # Check if original data matches
         self.assertEqual(self.sample_data, deserialized_data, "Deserialization error: Original data does not match deserialized data.")
 