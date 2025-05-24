from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import base64

# AES configuration
SECRET_KEY = b'AquaCipherKey123'  # 16 bytes
IV = b'InitializationVe'  # 16 bytes

def encrypt_aes(plaintext):
    try:
        cipher = AES.new(SECRET_KEY, AES.MODE_CBC, IV)
        padded_text = pad(plaintext.encode('utf-8'), AES.block_size)
        encrypted_bytes = cipher.encrypt(padded_text)
        return base64.b64encode(encrypted_bytes).decode('utf-8')
    except Exception as e:
        print(f"Encryption error: {e}")
        return plaintext  # Fallback to unencrypted

def decrypt_aes(ciphertext):
    try:
        cipher = AES.new(SECRET_KEY, AES.MODE_CBC, IV)
        encrypted_bytes = base64.b64decode(ciphertext.encode('utf-8'))
        decrypted_bytes = cipher.decrypt(encrypted_bytes)
        unpadded_text = unpad(decrypted_bytes, AES.block_size)
        return unpadded_text.decode('utf-8')
    except Exception as e:
        print(f"Decryption error: {e}")
        return ciphertext  # Return as-is if decryption fails