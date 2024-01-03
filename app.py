from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import os
import numpy as np
import PIL.Image
from sklearn.decomposition import PCA
from werkzeug.utils import secure_filename
import joblib

app = Flask(__name__, template_folder='templates')
app.secret_key = 'supersecretkey'  # Change this to a more secure key
app.config['UPLOAD_FOLDER'] = 'uploads'

class PCAImageCompressor:
    def __init__(self):
        self.image_path = ''
        self.result_filename = ''
        self.restored_filename = ''
        self.reconstruction_error = 0
        self.reconstructed_percentage = 0
        self.pca_model = self.load_model()  # Load the PCA model during initialization

    def load_model(self, model_filename='pca_model.joblib'):
        if os.path.exists(model_filename):
            return joblib.load(model_filename)
        else:
            return None

    def calculate_reconstruction_error(self, original, reconstructed):
        return np.sqrt(((original - reconstructed) ** 2).mean())

    def compress_image(self, image_path, num_components):
        img = PIL.Image.open(image_path)
        img = img.convert("L")
        img_array = np.array(img, dtype=np.float32)

        if self.pca_model is None:
            self.pca_model = PCA(n_components=num_components)
        compressed_img_array = self.pca_model.fit_transform(img_array)
        reconstructed_img_array = self.pca_model.inverse_transform(compressed_img_array)

        reconstructed_img_array = np.clip(reconstructed_img_array, 0, 255)
        reconstructed_img_array = reconstructed_img_array.astype(np.uint8)

        compressed_img = PIL.Image.fromarray(reconstructed_img_array)

        filename, extension = os.path.splitext(os.path.basename(image_path))
        result_filename = f"{filename}_compressed_pca_{num_components}.jpg"
        restored_filename = f"{filename}_restored_pca_{num_components}.jpg"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        restored_path = os.path.join(app.config['UPLOAD_FOLDER'], restored_filename)

        compressed_img.save(result_path)

        original_size = os.path.getsize(image_path)
        compressed_size = os.path.getsize(result_path)

        self.reconstruction_error = (compressed_size / original_size) * 100
        self.reconstructed_percentage = 100 - self.reconstruction_error

        # Save the restored image
        restored_img = PIL.Image.fromarray(reconstructed_img_array)
        restored_img.save(restored_path)
        self.restored_filename = restored_filename

        return result_path, restored_path, compressed_size

    def reset(self):
        self.image_path = ''
        self.result_filename = ''
        self.restored_filename = ''
        self.reconstruction_error = 0
        self.reconstructed_percentage = 0
        self.pca_model = self.load_model()  # Reload the PCA model

obj = PCAImageCompressor()

@app.route('/')
def index():
    return render_template('index.html', result='', restored_result='', compressed_img_size='', restoration_img_size='', reconstruction_error='', reconstructed_percentage='')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        flash('No image file provided.')
        return redirect(url_for('index'))

    image = request.files['image']
    if image.filename == '':
        flash('No selected image file.')
        return redirect(url_for('index'))

    num_components = int(request.form['components'])
    obj.reset()

    try:
        filename = secure_filename(image.filename)
        obj.image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(obj.image_path)

        result_path, restored_path, compressed_img_size = obj.compress_image(obj.image_path, num_components)

        return render_template('index.html', result=result_path, restored_result=restored_path,
                               compressed_img_size=compressed_img_size / 1024,
                               restoration_img_size=os.path.getsize(restored_path) / 1024,
                               reconstruction_error=obj.reconstruction_error,
                               reconstructed_percentage=obj.reconstructed_percentage)

    except Exception as e:
        flash(f'Error: {e}')
        return redirect(url_for('index'))

@app.route('/download/<filename>')
def download(filename):
    return send_file(filename, as_attachment=True)

@app.route('/download_restored')
def download_restored():
    restored_path = os.path.join(app.config['UPLOAD_FOLDER'], obj.restored_filename)
    return send_file(restored_path, as_attachment=True)

#if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    #app.run(debug=True)
