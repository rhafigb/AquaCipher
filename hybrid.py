import cv2
import numpy as np
from dct import embed_dct, extract_dct
from dwt import embed_dwt, extract_dwt

def embed_hybrid(host_image, watermark_data):
    # First apply DWT
    dwt_watermarked = embed_dwt(host_image, watermark_data)
    
    # Then apply DCT on the LL subband of DWT
    if len(dwt_watermarked.shape) == 3:
        ycc_image = cv2.cvtColor(dwt_watermarked, cv2.COLOR_BGR2YCrCb)
        y_channel = ycc_image[:,:,0]
    else:
        y_channel = dwt_watermarked
    
    # Perform DCT on Y channel
    watermarked_y = embed_dct(y_channel, watermark_data)
    
    if len(dwt_watermarked.shape) == 3:
        ycc_image[:,:,0] = watermarked_y
        hybrid_watermarked = cv2.cvtColor(ycc_image, cv2.COLOR_YCrCb2BGR)
    else:
        hybrid_watermarked = watermarked_y
    
    return hybrid_watermarked

def extract_hybrid(watermarked_image):
    # Extract from DCT first
    if len(watermarked_image.shape) == 3:
        ycc_image = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2YCrCb)
        y_channel = ycc_image[:,:,0]
    else:
        y_channel = watermarked_image
    
    dct_extracted = extract_dct(y_channel)
    
    # Then extract from DWT
    dwt_extracted = extract_dwt(watermarked_image)
    
    # Combine results (prioritize text from DCT and image from DWT)
    if isinstance(dct_extracted, str) and isinstance(dwt_extracted, str):
        # If both are text, return the longer one
        return dct_extracted if len(dct_extracted) > len(dwt_extracted) else dwt_extracted
    elif isinstance(dct_extracted, str):
        return dct_extracted
    elif isinstance(dwt_extracted, (np.ndarray, bytes)):
        return dwt_extracted
    else:
        return dct_extracted or dwt_extracted or "No watermark detected"