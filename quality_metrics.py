import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(original, watermarked):
    if original.shape != watermarked.shape:
        watermarked = cv2.resize(watermarked, (original.shape[1], original.shape[0]))
    mse = np.mean((original - watermarked) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def calculate_ssim(original, watermarked):
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        watermarked_gray = cv2.cvtColor(watermarked, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        watermarked_gray = watermarked
    
    if original_gray.shape != watermarked_gray.shape:
        watermarked_gray = cv2.resize(watermarked_gray, (original_gray.shape[1], original_gray.shape[0]))
    
    return ssim(original_gray, watermarked_gray, 
               data_range=watermarked_gray.max() - watermarked_gray.min(),
               win_size=3)

def calculate_nc(original_wm, extracted_wm):
    if isinstance(original_wm, str) and isinstance(extracted_wm, str):
        original_bits = ''.join(format(ord(c), '08b') for c in original_wm)
        extracted_bits = ''.join(format(ord(c), '08b') for c in extracted_wm)
        min_len = min(len(original_bits), len(extracted_bits))
        if min_len == 0:
            return 0.0
        matches = sum(1 for a, b in zip(original_bits[:min_len], extracted_bits[:min_len]) if a == b)
        return matches / min_len
    else:
        if isinstance(original_wm, bytes):
            original_wm = np.frombuffer(original_wm, dtype=np.uint8)
        if isinstance(extracted_wm, bytes):
            extracted_wm = np.frombuffer(extracted_wm, dtype=np.uint8)
        
        min_len = min(len(original_wm), len(extracted_wm))
        if min_len == 0:
            return 0.0
            
        original_wm = original_wm[:min_len]
        extracted_wm = extracted_wm[:min_len]
        
        cross_corr = np.correlate(original_wm, extracted_wm)
        auto_corr = np.correlate(original_wm, original_wm)
        
        return (cross_corr[0] / auto_corr[0]) if auto_corr[0] != 0 else 0.0