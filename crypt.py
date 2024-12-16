
from cryptography.fernet import Fernet

# Generate and print the encryption key
encryption_key = Fernet.generate_key()
print(encryption_key.decode())