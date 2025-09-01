import numpy as np
import cv2
from skimage import measure

def calculate_metrics(segmentation_mask):
    metrics = {}
    metrics['area'] = np.sum(segmentation_mask)
    contours, _ = cv2.findContours(segmentation_mask.astype(np.uint8), 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    perimeter = 0
    for contour in contours:
        perimeter += cv2.arcLength(contour, True)
    metrics['perimeter'] = perimeter
    if perimeter > 0:
        metrics['circularity'] = 4 * np.pi * metrics['area'] / (perimeter**2)
    else:
        metrics['circularity'] = 0
    if np.sum(segmentation_mask) > 0:
        props = measure.regionprops(segmentation_mask.astype(np.uint8))
        if len(props) > 0:
            metrics['eccentricity'] = props[0].eccentricity
            metrics['major_axis_length'] = props[0].major_axis_length
            metrics['minor_axis_length'] = props[0].minor_axis_length
            metrics['solidity'] = props[0].solidity
    else:
        metrics['eccentricity'] = 0
        metrics['major_axis_length'] = 0
        metrics['minor_axis_length'] = 0
        metrics['solidity'] = 0
    return metrics

def calculate_dice_coefficient(predicted_mask, ground_truth_mask):
    pred = predicted_mask > 0
    gt = ground_truth_mask > 0
    intersection = np.logical_and(pred, gt).sum()
    sum_pred = pred.sum()
    sum_gt = gt.sum()
    if sum_pred + sum_gt > 0:
        dice = 2.0 * intersection / (sum_pred + sum_gt)
    else:
        dice = 1.0
    return dice

def calculate_jaccard_index(predicted_mask, ground_truth_mask):
    pred = predicted_mask > 0
    gt = ground_truth_mask > 0
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union > 0:
        jaccard = intersection / union
    else:
        jaccard = 1.0
    return jaccard

def calculate_sensitivity_specificity(predicted_mask, ground_truth_mask):
    pred = predicted_mask > 0
    gt = ground_truth_mask > 0
    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sensitivity, specificity
