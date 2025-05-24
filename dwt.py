import cv2
import numpy as np
import pywt

def embed_dwt(host_image, watermark_data):
    if len(host_image.shape) == 3:
        gray_image = cv2.cvtColor(host_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = host_image
    
    if isinstance(watermark_data, str):
        watermark_binary = ''.join(format(ord(c), '08b') for c in watermark_data)
    else:
        watermark_binary = ''.join(format(p, '08b') for p in watermark_data.flatten())
    
    coeffs = pywt.wavedec2(gray_image, 'haar', level=2)
    cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
    
    watermark_index = 0
    for i in range(cH2.shape[0]):
        for j in range(cH2.shape[1]):
            if watermark_index >= len(watermark_binary):
                break
            
            bit = int(watermark_binary[watermark_index])
            
            # Modify coefficients
            cH2[i,j] = cH2[i,j] * (1 + 0.03 * bit)
            cV2[i,j] = cV2[i,j] * (1 + 0.03 * bit)
            
            watermark_index += 1
    
    new_coeffs = (cA2, (cH2, cV2, cD2), (cH1, cV1, cD1))
    watermarked_gray = pywt.waverec2(new_coeffs, 'haar')
    watermarked_gray = np.clip(watermarked_gray, 0, 255).astype(np.uint8)
    
    if len(host_image.shape) == 3:
        watermarked_image = cv2.cvtColor(watermarked_gray, cv2.COLOR_GRAY2BGR)
    else:
        watermarked_image = watermarked_gray
    
    return watermarked_image

def extract_dwt(watermarked_image):
    if len(watermarked_image.shape) == 3:
        gray_image = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = watermarked_image
    
    coeffs = pywt.wavedec2(gray_image, 'haar', level=2)
    cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
    
    watermark_bits = []
    max_bits = 10000  # Limit to prevent excessive extraction
    
    for i in range(cH2.shape[0]):
        for j in range(cH2.shape[1]):
            if len(watermark_bits) >= max_bits:
                break
            
            # Extract bit from coefficients
            h_bit = 1 if cH2[i,j] > 1.015 * cH1[min(i, cH1.shape[0]-1), min(j, cH1.shape[1]-1)] else 0
            v_bit = 1 if cV2[i,j] > 1.015 * cV1[min(i, cV1.shape[0]-1), min(j, cV1.shape[1]-1)] else 0
            
            bit = 1 if (h_bit + v_bit) >= 1 else 0
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