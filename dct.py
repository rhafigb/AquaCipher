import cv2
import numpy as np

def embed_dct(host_image, watermark_data):
    if len(host_image.shape) == 3:
        ycc_image = cv2.cvtColor(host_image, cv2.COLOR_BGR2YCrCb)
        y_channel = ycc_image[:,:,0]
    else:
        y_channel = host_image
    
    if isinstance(watermark_data, str):
        watermark_binary = ''.join(format(ord(c), '08b') for c in watermark_data)
    else:
        watermark_binary = ''.join(format(p, '08b') for p in watermark_data.flatten())
    
    watermarked = y_channel.copy().astype(np.float32)
    block_size = 8
    watermark_index = 0
    
    for i in range(0, y_channel.shape[0], block_size):
        for j in range(0, y_channel.shape[1], block_size):
            if watermark_index >= len(watermark_binary):
                break
            
            block = y_channel[i:i+block_size, j:j+block_size]
            dct_block = cv2.dct(block.astype(np.float32))
            
            if watermark_index < len(watermark_binary):
                bit = int(watermark_binary[watermark_index])
                dct_block[4,4] = dct_block[4,4] * (1 + 0.01 * bit)
                watermark_index += 1
            
            watermarked[i:i+block_size, j:j+block_size] = cv2.idct(dct_block)
    
    if len(host_image.shape) == 3:
        ycc_image[:,:,0] = np.clip(watermarked, 0, 255).astype(np.uint8)
        watermarked_image = cv2.cvtColor(ycc_image, cv2.COLOR_YCrCb2BGR)
    else:
        watermarked_image = np.clip(watermarked, 0, 255).astype(np.uint8)
    
    return watermarked_image

def extract_dct(watermarked_image):
    if len(watermarked_image.shape) == 3:
        ycc_image = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2YCrCb)
        y_channel = ycc_image[:,:,0]
    else:
        y_channel = watermarked_image
    
    block_size = 8
    watermark_bits = []
    max_bits = 10000  # Limit to prevent excessive extraction
    
    for i in range(0, y_channel.shape[0], block_size):
        for j in range(0, y_channel.shape[1], block_size):
            if len(watermark_bits) >= max_bits:
                break
            
            block = y_channel[i:i+block_size, j:j+block_size]
            dct_block = cv2.dct(block.astype(np.float32))
            
            # Check if coefficient was modified
            if dct_block[4,4] > 1.005 * dct_block[3,3]:  # Threshold
                bit = 1
            else:
                bit = 0
                
            watermark_bits.append(str(bit))
    
    # Convert binary to data
    watermark_bytes = []
    for i in range(0, len(watermark_bits), 8):
        byte = watermark_bits[i:i+8]
        if len(byte) < 8:
            break
        watermark_bytes.append(int(''.join(byte), 2))
    
    try:
        # Try to return as text
        watermark_text = bytes(watermark_bytes).decode('utf-8', errors='ignore').strip('\x00')
        return watermark_text
    except:
        # Return as numpy array if it looks like image data
        if len(watermark_bytes) > 100:  # Arbitrary threshold for image data
            size = int(np.sqrt(len(watermark_bytes)//3)*3)  # Approximate square image
            if size > 0:
                arr = np.array(watermark_bytes[:size*size]).reshape((size, size))
                return arr
        return bytes(watermark_bytes)