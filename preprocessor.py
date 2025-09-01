import numpy as np
import SimpleITK as sitk
import pydicom
import io
import tempfile
import os
import cv2
from PIL import Image

def load_medical_image(file_obj):
    file_extension = file_obj.name.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
        tmp_file.write(file_obj.getvalue())
        tmp_file_path = tmp_file.name
    try:
        if file_extension == 'dcm':
            dicom_data = pydicom.dcmread(tmp_file_path)
            image = dicom_data.pixel_array
            if image.dtype != np.uint8:
                image = (image - image.min()) / (image.max() - image.min()) * 255
                image = image.astype(np.uint8)
            original_image = image.copy()
        elif file_extension in ['nii', 'nii.gz']:
            image_sitk = sitk.ReadImage(tmp_file_path)
            image = sitk.GetArrayFromImage(image_sitk)
            if len(image.shape) == 3:
                middle_slice = image.shape[0] // 2
                image = image[middle_slice, :, :]
            if image.dtype != np.uint8:
                image = (image - image.min()) / (image.max() - image.min()) * 255
                image = image.astype(np.uint8)
            original_image = image.copy()
        else:
            image = np.array(Image.open(tmp_file_path).convert('RGB'))
            original_image = image.copy()
        os.unlink(tmp_file_path)
        return image, original_image
    except Exception as e:
        os.unlink(tmp_file_path)
        raise Exception(f"Error loading medical image: {str(e)}")

def normalize_image(image):
    if image.dtype != np.uint8:
        normalized = (image - image.min()) / (image.max() - image.min()) * 255
        return normalized.astype(np.uint8)
    return image

def enhance_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray.astype(np.uint8))
    return enhanced

def apply_windowing(image, window_center, window_width):
    min_value = window_center - window_width // 2
    max_value = window_center + window_width // 2
    windowed = np.clip(image, min_value, max_value)
    windowed = ((windowed - min_value) / (max_value - min_value)) * 255
    return windowed.astype(np.uint8)
