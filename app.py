import streamlit as st
import os
import numpy as np
import PIL.Image
from sklearn.decomposition import PCA
import joblib

# Set up Streamlit app
st.title('PCA Image Compressor')

# Create a class for PCA Image Compressor
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

    def compress_image(self, image_content, num_components):
        img = PIL.Image.open(image_content).convert("L")
        img_array = np.array(img, dtype=np.float32)

        if self.pca_model is None:
            self.pca_model = PCA(n_components=num_components)
        compressed_img_array = self.pca_model.fit_transform(img_array)
        reconstructed_img_array = self.pca_model.inverse_transform(compressed_img_array)

        reconstructed_img_array = np.clip(reconstructed_img_array, 0, 255)
        reconstructed_img_array = reconstructed_img_array.astype(np.uint8)

        compressed_img = PIL.Image.fromarray(reconstructed_img_array)

        filename, extension = os.path.splitext('uploaded_image.jpg')
        result_filename = f"{filename}_compressed_pca_{num_components}.jpg"
        restored_filename = f"{filename}_restored_pca_{num_components}.jpg"
        result_path = os.path.join('uploads', result_filename.replace(" ", "_"))
        restored_path = os.path.join('uploads', restored_filename.replace(" ", "_"))

        compressed_img.save(result_path)

        original_size = len(image_content.getvalue())
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

# Streamlit app layout
st.sidebar.header('Upload Image')
uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    num_components = st.sidebar.slider('Select Number of Components:', 1, 100, 10)

    if st.sidebar.button('Compress Image'):
        try:
            obj.reset()

            result_path, restored_path, compressed_img_size = obj.compress_image(uploaded_file, num_components)

            st.image(result_path, caption='Compressed Image.', use_column_width=True)
            st.image(restored_path, caption='Restored Image.', use_column_width=True)
            
            # Download links
            st.sidebar.markdown(f"**Download Results:**")
            st.sidebar.markdown(f"- [Compressed Image]({result_path})")
            st.sidebar.markdown(f"- [Restored Image]({restored_path})")
            
        except Exception as e:
            st.error(f'Error: {e}')
else:
    st.warning('Please upload an image.')
