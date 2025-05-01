import unittest
import hashlib
import os

class TestHashlibIntegration(unittest.TestCase):
    def setUp(self):
        """Create a temporary file with example data for hashing."""
        self.test_filename = "test_file.txt"
        self.test_data = b"example model data"
       
        with open(self.test_filename, "wb") as f:
            f.write(self.test_data)
       
        self.expected_hash = hashlib.sha256(self.test_data).hexdigest()

    def tearDown(self):
        """Remove the temporary test file after the test completes."""
        if os.path.exists(self.test_filename):
            os.remove(self.test_filename)

    def test_hash_file_content(self):
        """Verify the hash of a file's content is consistent."""
        with open(self.test_filename, "rb") as f:
            file_data = f.read()
       
        digest = hashlib.sha256(file_data).hexdigest()
        self.assertEqual(digest, self.expected_hash, "File hash mismatch")

    def test_hash_pipeline_storage(self):
        """Simulate a system that hashes and stores data, then verifies integrity."""
        # Simulate data transmission
        transmitted_data = self.test_data

        # Hash before storage
        stored_hash = hashlib.sha256(transmitted_data).hexdigest()

        # Retrieve and verify
        retrieved_data = transmitted_data  # Simulating retrieval
        retrieved_hash = hashlib.sha256(retrieved_data).hexdigest()

        self.assertEqual(stored_hash, retrieved_hash, "Hash mismatch in pipeline storage")

    def test_hash_verification_after_storage(self):
        """Ensure stored hash matches recomputed hash after retrieval."""
        # Simulating a hash stored in a database
        stored_hash = self.expected_hash

        # Read file again and recompute hash
        with open(self.test_filename, "rb") as f:
            file_data = f.read()
       
        computed_hash = hashlib.sha256(file_data).hexdigest()
       
        self.assertEqual(stored_hash, computed_hash, "Stored hash does not match computed hash")

if __name__ == "__main__":
    unittest.main()