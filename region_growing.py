import numpy as np
from skimage import filters
import cv2
from queue import Queue

def region_growing_segmentation(image, seed_point, threshold=0.1, connectivity=8, max_iterations=100):
    if len(image.shape) == 3 and image.shape[2] > 1:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image.copy()
    height, width = gray_image.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    if (seed_point[0] < 0 or seed_point[0] >= height or 
        seed_point[1] < 0 or seed_point[1] >= width):
        raise ValueError(f"Seed point {seed_point} is outside image bounds ({height}x{width})")
    seed_value = float(gray_image[seed_point])
    normalized_image = gray_image.astype(np.float32) / 255.0
    seed_value_normalized = seed_value / 255.0
    if connectivity == 4:
        neighbor_offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    else:
        neighbor_offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                            (1, 0), (1, -1), (0, -1), (-1, -1)]
    queue = Queue()
    queue.put(seed_point)
    mask[seed_point] = 1
    iterations = 0
    while not queue.empty() and iterations < max_iterations:
        current_point = queue.get()
        current_row, current_col = current_point
        for offset_row, offset_col in neighbor_offsets:
            neighbor_row = current_row + offset_row
            neighbor_col = current_col + offset_col
            if (neighbor_row < 0 or neighbor_row >= height or 
                neighbor_col < 0 or neighbor_col >= width):
                continue
            if mask[neighbor_row, neighbor_col] == 1:
                continue
            neighbor_value = normalized_image[neighbor_row, neighbor_col]
            if abs(neighbor_value - seed_value_normalized) <= threshold:
                mask[neighbor_row, neighbor_col] = 1
                queue.put((neighbor_row, neighbor_col))
        iterations += 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask

def region_growing_adaptive(image, seed_point, initial_threshold=0.1, max_threshold=0.3, connectivity=8, max_iterations=100):
    min_region_size = image.shape[0] * image.shape[1] * 0.005
    max_region_size = image.shape[0] * image.shape[1] * 0.3
    current_threshold = initial_threshold
    mask = region_growing_segmentation(image, seed_point, current_threshold, connectivity, max_iterations)
    region_size = np.sum(mask)
    while region_size < min_region_size and current_threshold < max_threshold:
        current_threshold += 0.05
        mask = region_growing_segmentation(image, seed_point, current_threshold, connectivity, max_iterations)
        region_size = np.sum(mask)
    if region_size > max_region_size:
        current_threshold = initial_threshold
        while region_size > max_region_size and current_threshold > 0.01:
            current_threshold -= 0.01
            mask = region_growing_segmentation(image, seed_point, current_threshold, connectivity, max_iterations)
            region_size = np.sum(mask)
    return mask
