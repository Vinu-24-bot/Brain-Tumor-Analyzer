import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage
from scipy import signal
import matplotlib.pyplot as plt

def calculate_ssim(image1, image2):
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    if image1.dtype != np.uint8:
        image1 = (image1 / image1.max() * 255).astype(np.uint8)
    if image2.dtype != np.uint8:
        image2 = (image2 / image2.max() * 255).astype(np.uint8)
    ssim_value, _ = ssim(image1, image2, full=True)
    return ssim_value

def butterworth_filter(image, cutoff_frequency=30, order=2, high_pass=True):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    u, v = np.meshgrid(u, v, indexing='ij')
    u = u - crow
    v = v - ccol
    D = np.sqrt(u**2 + v**2)
    if high_pass:
        H = 1 / (1 + (cutoff_frequency / (D + 1e-8))**(2 * order))
    else:
        H = 1 / (1 + (D / (cutoff_frequency + 1e-8))**(2 * order))
    F = np.fft.fft2(image)
    F_shifted = np.fft.fftshift(F)
    F_filtered = F_shifted * H
    F_filtered_shifted = np.fft.ifftshift(F_filtered)
    filtered_image = np.abs(np.fft.ifft2(F_filtered_shifted))
    filtered_image = (filtered_image - filtered_image.min()) / (filtered_image.max() - filtered_image.min()) * 255
    return filtered_image.astype(np.uint8)

def calculate_bf_score(segmented_image, ground_truth=None, cutoff_frequency=30, order=2):
    if segmented_image.dtype != np.uint8:
        segmented_image = (segmented_image > 0).astype(np.uint8) * 255
    edges_x = cv2.Sobel(segmented_image, cv2.CV_64F, 1, 0, ksize=3)
    edges_y = cv2.Sobel(segmented_image, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(edges_x**2 + edges_y**2)
    edges = (edges / edges.max() * 255).astype(np.uint8)
    filtered_edges = butterworth_filter(edges, cutoff_frequency, order, high_pass=True)
    if ground_truth is not None:
        if ground_truth.dtype != np.uint8:
            ground_truth = (ground_truth > 0).astype(np.uint8) * 255
        gt_edges_x = cv2.Sobel(ground_truth, cv2.CV_64F, 1, 0, ksize=3)
        gt_edges_y = cv2.Sobel(ground_truth, cv2.CV_64F, 0, 1, ksize=3)
        gt_edges = np.sqrt(gt_edges_x**2 + gt_edges_y**2)
        gt_edges = (gt_edges / gt_edges.max() * 255).astype(np.uint8)
        filtered_gt_edges = butterworth_filter(gt_edges, cutoff_frequency, order, high_pass=True)
        bf_score = calculate_ssim(filtered_edges, filtered_gt_edges)
    else:
        bf_score = np.sum(filtered_edges) / (filtered_edges.size * 255)
    return bf_score

def calculate_jaccard_index(segmented_image, ground_truth):
    pred = segmented_image > 0
    gt = ground_truth > 0
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union > 0:
        jaccard = intersection / union
    else:
        jaccard = 1.0 if np.sum(pred) == 0 and np.sum(gt) == 0 else 0.0
    return jaccard

def generate_dummy_ground_truth(segmented_image, variation_factor=0.1):
    dummy_gt = segmented_image.copy().astype(np.uint8)
    kernel_size = max(3, int(min(segmented_image.shape) * variation_factor * 0.1))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if np.random.rand() > 0.5:
        dummy_gt = cv2.dilate(dummy_gt, kernel, iterations=1)
    else:
        dummy_gt = cv2.erode(dummy_gt, kernel, iterations=1)
    contours, _ = cv2.findContours(dummy_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(dummy_gt)
    for contour in contours:
        for i in range(len(contour)):
            if np.random.rand() > 0.7:
                dx = np.random.randint(-3, 4)
                dy = np.random.randint(-3, 4)
                contour[i][0][0] += dx
                contour[i][0][1] += dy
        cv2.drawContours(mask, [contour], 0, 1, -1)
    return mask

def generate_metric_plot(segmented_image, metrics):
    radar_metrics = [
        ('SSIM', metrics.get('ssim', 0)),
        ('BF Score', metrics.get('bf_score', 0)),
        ('Jaccard', metrics.get('jaccard', 0)),
        ('Circularity', metrics.get('circularity', 0))
    ]
    labels = [m[0] for m in radar_metrics]
    values = [m[1] for m in radar_metrics]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    labels += labels[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles), labels)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.set_title("Segmentation Quality Metrics", size=15)
    return fig
