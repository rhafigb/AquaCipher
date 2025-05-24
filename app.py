import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import io
import base64
from dct import embed_dct, extract_dct
from dwt import embed_dwt, extract_dwt
from hybrid import embed_hybrid, extract_hybrid
from aes import encrypt_aes, decrypt_aes
from quality_metrics import calculate_psnr, calculate_ssim, calculate_nc

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html', page='home')

@app.route('/watermarking')
def watermarking():
    return render_template('index.html', page='watermarking')

@app.route('/about')
def about():
    return render_template('index.html', page='about')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        img = Image.open(filepath)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({
            'filename': filename,
            'image': f"data:image/png;base64,{img_str}"
        })

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    method = data['method']
    watermark = data.get('watermark', '')
    filename = data['filename']
    action = data['action']
    use_aes = data.get('use_aes', False)
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    
    if action == 'embed':
        if len(watermark) < 100:  # Assuming text watermark
            encrypted_watermark = encrypt_aes(watermark) if use_aes else watermark
        else:
            encrypted_watermark = watermark  # Assuming image watermark
            
        if method == 'dct':
            watermarked_img = embed_dct(img, encrypted_watermark)
        elif method == 'dwt':
            watermarked_img = embed_dwt(img, encrypted_watermark)
        else:  # hybrid
            watermarked_img = embed_hybrid(img, encrypted_watermark)
            
        watermarked_filename = f"watermarked_{filename}"
        watermarked_path = os.path.join(app.config['UPLOAD_FOLDER'], watermarked_filename)
        cv2.imwrite(watermarked_path, watermarked_img)
        
        original_img = cv2.imread(filepath)
        psnr = calculate_psnr(original_img, watermarked_img)
        ssim = calculate_ssim(original_img, watermarked_img)
        
        _, buffer = cv2.imencode('.png', watermarked_img)
        watermarked_img_str = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'watermarked_image': f"data:image/png;base64,{watermarked_img_str}",
            'watermarked_filename': watermarked_filename,
            'metrics': {
                'psnr': f"{psnr:.2f} dB",
                'ssim': f"{ssim:.4f}"
            }
        })
        
    elif action == 'extract':
        if method == 'dct':
            extracted_data = extract_dct(img)
        elif method == 'dwt':
            extracted_data = extract_dwt(img)
        else:  # hybrid
            extracted_data = extract_hybrid(img)
            
        # Try to decrypt if AES was used
        if use_aes and isinstance(extracted_data, str):
            try:
                decrypted_data = decrypt_aes(extracted_data)
                if len(decrypted_data) < 100:  # Assuming text watermark
                    extracted_data = decrypted_data
            except:
                extracted_data = "Decryption failed - possibly wrong key or corrupted data"
        
        # If it's an image, convert to base64
        if isinstance(extracted_data, np.ndarray):
            _, buffer = cv2.imencode('.png', extracted_data)
            extracted_img_str = base64.b64encode(buffer).decode('utf-8')
            extracted_data = f"data:image/png;base64,{extracted_img_str}"
            
        return jsonify({
            'status': 'success',
            'extracted_data': extracted_data,
            'is_image': isinstance(extracted_data, str) and extracted_data.startswith('data:image')
        })

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)