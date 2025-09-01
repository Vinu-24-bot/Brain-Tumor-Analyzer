import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from PIL import Image
import io
import cv2
from region_growing import region_growing_segmentation, region_growing_adaptive
from utils import preprocess_image, overlay_segmentation, convert_to_grayscale, calculate_tumor_probability
from evaluation import calculate_metrics, calculate_dice_coefficient, calculate_jaccard_index as calc_jaccard, calculate_sensitivity_specificity
from preprocessor import load_medical_image, normalize_image, enhance_contrast
from advanced_metrics import (
    calculate_ssim, 
    calculate_bf_score, 
    calculate_jaccard_index,
    generate_dummy_ground_truth,
    generate_metric_plot
)

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
)

def main():
    st.title("Brain Tumor Detection with Region Growing")
    st.write("""
    This application uses region growing algorithm to segment brain tumors from MRI images.
    Upload your images and adjust parameters to get the optimal segmentation results.
    The app provides advanced evaluation metrics including SSIM, BF score, and Jaccard Index.
    """)

    st.sidebar.title("Algorithm Parameters")
    
    seed_point_method = st.sidebar.radio(
        "Seed Point Selection",
        ["Automatic", "Manual"]
    )
    
    threshold = st.sidebar.slider(
        "Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.1, 
        step=0.01
    )
    
    connectivity = st.sidebar.radio(
        "Connectivity",
        [4, 8]
    )
    
    iterations = st.sidebar.slider(
        "Max Iterations",
        min_value=10,
        max_value=1000,
        value=100,
        step=10
    )
    
    smoothing = st.sidebar.slider(
        "Smoothing Factor",
        min_value=0.0,
        max_value=5.0,
        value=0.5,
        step=0.1
    )

    st.sidebar.title("Advanced Metrics Parameters")
    
    bf_cutoff = st.sidebar.slider(
        "BF Cutoff Frequency",
        min_value=10,
        max_value=100,
        value=30,
        step=5
    )
    
    bf_order = st.sidebar.slider(
        "BF Filter Order",
        min_value=1,
        max_value=6,
        value=2,
        step=1
    )
    
    gt_options = st.sidebar.radio(
        "Ground Truth",
        ["None", "Upload", "Auto-generate"]
    )

    uploaded_file = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png", "tif", "bmp", "dcm", "nii", "nii.gz"])
    
    uploaded_gt = None
    if gt_options == "Upload":
        uploaded_gt = st.file_uploader("Upload ground truth segmentation", type=["jpg", "jpeg", "png", "tif", "bmp"])

    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension in ['dcm', 'nii', 'nii.gz']:
                image, original_image = load_medical_image(uploaded_file)
            else:
                image = np.array(Image.open(uploaded_file).convert('RGB'))
                original_image = image.copy()
                if len(image.shape) == 3 and image.shape[2] > 1:
                    image = convert_to_grayscale(image)
            
            st.subheader("Original Image")
            fig_original, ax_original = plt.subplots(figsize=(10, 8))
            ax_original.imshow(original_image, cmap='gray' if len(original_image.shape) == 2 else None)
            ax_original.axis('off')
            st.pyplot(fig_original)
            
            ground_truth = None
            if gt_options == "Upload" and uploaded_gt is not None:
                ground_truth = np.array(Image.open(uploaded_gt).convert('L'))
                ground_truth = (ground_truth > 127).astype(np.uint8)
                
                st.subheader("Ground Truth")
                fig_gt, ax_gt = plt.subplots(figsize=(10, 8))
                ax_gt.imshow(ground_truth, cmap='gray')
                ax_gt.axis('off')
                st.pyplot(fig_gt)
            
            preprocessed_image = preprocess_image(image, smoothing)
            
            if seed_point_method == "Automatic":
                st.info("Automatic seed point selection is being used.")
                if len(preprocessed_image.shape) == 2:
                    seed_point = np.unravel_index(np.argmax(preprocessed_image), preprocessed_image.shape)
                else:
                    gray = cv2.cvtColor(preprocessed_image, cv2.COLOR_RGB2GRAY)
                    seed_point = np.unravel_index(np.argmax(gray), gray.shape)
            else:
                st.info("Please click on the suspected tumor area.")
                fig_seed, ax_seed = plt.subplots(figsize=(10, 8))
                ax_seed.imshow(preprocessed_image, cmap='gray' if len(preprocessed_image.shape) == 2 else None)
                ax_seed.axis('off')
                st.pyplot(fig_seed)
                
                seed_coords = st.text_input("Enter seed coordinates (row, column):", value="100, 100")
                try:
                    seed_point = tuple(map(int, seed_coords.split(',')))
                except:
                    st.error("Invalid coordinates. Using default (100, 100).")
                    seed_point = (100, 100)

            if st.button("Run Segmentation"):
                with st.spinner("Processing..."):
                    segmented_image = region_growing_segmentation(
                        preprocessed_image, 
                        seed_point,
                        threshold,
                        connectivity,
                        iterations
                    )
                    
                    if gt_options == "Auto-generate" and ground_truth is None:
                        ground_truth = generate_dummy_ground_truth(segmented_image)
                        
                        st.subheader("Auto-Generated Ground Truth (Simulated)")
                        fig_auto_gt, ax_auto_gt = plt.subplots(figsize=(10, 8))
                        ax_auto_gt.imshow(ground_truth, cmap='gray')
                        ax_auto_gt.axis('off')
                        st.pyplot(fig_auto_gt)
                    
                    st.subheader("Segmentation Result")
                    
                    if ground_truth is not None:
                        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 8))
                    else:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
                    
                    ax1.imshow(original_image, cmap='gray' if len(original_image.shape) == 2 else None)
                    ax1.set_title("Original Image")
                    ax1.axis('off')
                    
                    overlay = overlay_segmentation(original_image, segmented_image)
                    ax2.imshow(overlay)
                    ax2.set_title("Segmentation Overlay")
                    ax2.axis('off')
                    
                    if ground_truth is not None:
                        comparison = np.zeros_like(segmented_image, dtype=np.uint8)
                        comparison_rgb = np.zeros((comparison.shape[0], comparison.shape[1], 3), dtype=np.uint8)
                        comparison_rgb[np.logical_and(segmented_image > 0, ground_truth == 0)] = [255, 0, 0]
                        comparison_rgb[np.logical_and(segmented_image == 0, ground_truth > 0)] = [0, 255, 0]
                        comparison_rgb[np.logical_and(segmented_image > 0, ground_truth > 0)] = [255, 255, 255]
                        ax3.imshow(comparison_rgb)
                        ax3.set_title("Comparison with Ground Truth")
                        ax3.axis('off')
                    
                    st.pyplot(fig)
                    
                    st.subheader("Evaluation Metrics")
                    
                    metrics = calculate_metrics(segmented_image)
                    
                    if ground_truth is not None:
                        dice_coef = calculate_dice_coefficient(segmented_image, ground_truth)
                        metrics['dice'] = dice_coef
                        sensitivity, specificity = calculate_sensitivity_specificity(segmented_image, ground_truth)
                        metrics['sensitivity'] = sensitivity
                        metrics['specificity'] = specificity
                        jaccard = calculate_jaccard_index(segmented_image, ground_truth)
                        metrics['jaccard'] = jaccard
                        gray_image = convert_to_grayscale(original_image)
                        segmentation_vis = np.zeros_like(gray_image)
                        segmentation_vis[segmented_image > 0] = gray_image[segmented_image > 0]
                        ground_truth_vis = np.zeros_like(gray_image)
                        ground_truth_vis[ground_truth > 0] = gray_image[ground_truth > 0]
                        ssim_value = calculate_ssim(segmentation_vis, ground_truth_vis)
                        metrics['ssim'] = ssim_value
                        bf_score = calculate_bf_score(segmented_image, ground_truth, bf_cutoff, bf_order)
                        metrics['bf_score'] = bf_score
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Segmented Area (pixels)", f"{metrics['area']}")
                        col1.metric("Perimeter (pixels)", f"{metrics['perimeter']:.2f}")
                        col1.metric("Circularity", f"{metrics['circularity']:.4f}")
                        
                        col2.metric("Dice Coefficient", f"{metrics['dice']:.4f}")
                        col2.metric("Jaccard Index", f"{metrics['jaccard']:.4f}")
                        col2.metric("SSIM", f"{metrics['ssim']:.4f}")
                        
                        col3.metric("BF Score", f"{metrics['bf_score']:.4f}")
                        col3.metric("Sensitivity", f"{metrics['sensitivity']:.4f}")
                        col3.metric("Specificity", f"{metrics['specificity']:.4f}")
                        
                        radar_fig = generate_metric_plot(segmented_image, metrics)
                        st.pyplot(radar_fig)
                        
                        st.subheader("Advanced Metrics Explanation")
                        st.write("""
                        **Structural Similarity Index Measure (SSIM):** Measures the perceived similarity between two images.
                        **Butterworth Filter (BF) Score:** Evaluates edge alignment quality.
                        **Jaccard Index:** Measures overlap between segmentation and ground truth.
                        """)
                        
                    else:
                        bf_score = calculate_bf_score(segmented_image, None, bf_cutoff, bf_order)
                        metrics['bf_score'] = bf_score
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Segmented Area (pixels)", f"{metrics['area']}")
                        col2.metric("Perimeter (pixels)", f"{metrics['perimeter']:.2f}")
                        col3.metric("Circularity", f"{metrics['circularity']:.4f}")
                        
                        st.info("For advanced metrics, a ground truth is required.")
                        st.metric("BF Score (Edge Quality)", f"{metrics['bf_score']:.4f}")
                        st.write("""
                        **Butterworth Filter (BF) Score:** Measures edge strength in the segmentation mask.
                        """)
                    
                    st.subheader("Intensity Distribution")
                    fig_hist, (ax_hist1, ax_hist2) = plt.subplots(1, 2, figsize=(15, 5))
                    
                    if len(image.shape) == 2:
                        ax_hist1.hist(image.flatten(), bins=50, color='blue', alpha=0.7)
                    else:
                        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                        ax_hist1.hist(gray.flatten(), bins=50, color='blue', alpha=0.7)
                    ax_hist1.set_title("Original Image Histogram")
                    ax_hist1.set_xlabel("Pixel Intensity")
                    ax_hist1.set_ylabel("Frequency")
                    
                    mask = segmented_image > 0
                    if len(image.shape) == 2:
                        segmented_values = image[mask]
                    else:
                        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                        segmented_values = gray[mask]
                        
                    if len(segmented_values) > 0:
                        ax_hist2.hist(segmented_values, bins=50, color='red', alpha=0.7)
                        ax_hist2.set_title("Segmented Region Histogram")
                        ax_hist2.set_xlabel("Pixel Intensity")
                        ax_hist2.set_ylabel("Frequency")
                    else:
                        ax_hist2.set_title("No pixels in segmented region")
                    
                    st.pyplot(fig_hist)
                    
                    st.subheader("Save Results")
                    
                    buffer = io.BytesIO()
                    plt.figure(figsize=(10, 8))
                    plt.imshow(overlay)
                    plt.axis('off')
                    plt.tight_layout(pad=0)
                    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
                    buffer.seek(0)
                    
                    st.download_button(
                        label="Download Segmentation Result",
                        data=buffer,
                        file_name="brain_tumor_segmentation.png",
                        mime="image/png"
                    )
                    
                    st.subheader("Interpretation")
                    st.write("""
                    The red overlay indicates the detected tumor region.
                    **Area**: Segmented region size in pixels.
                    **Perimeter**: Length of boundary.
                    **Circularity**: Shape similarity to a circle.
                    """)
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    st.sidebar.header("About")
    st.sidebar.info("""
    This application uses region growing segmentation for brain tumors in MRI images.
    **Region Growing**: Expands regions based on intensity similarity.
    **Advanced Metrics**: SSIM, BF Score, Jaccard Index.
    """)

if __name__ == "__main__":
    main()
