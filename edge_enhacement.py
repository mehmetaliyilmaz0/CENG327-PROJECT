"""
CENG327 - PROJECT
@TOPİC: Gradient-Based Edge Enhancement in Images
@AUTHOR: MEHMET ALİ YILMAZ 21050111057
"""


"""
Gradient-Based Edge Enhancement Application
----------------------------------------
This application performs edge detection and enhancement
using gradient operators on images.

Features:
- Sobel Gradient operator 
- Automatic threshold determination (Otsu)
- Gradient magnitude and direction analysis
"""


import numpy as np #for numerical operations and handling arrays.
from PIL import Image #for image loading and manipulation
import cv2 #for image processing, including filtering and edge detection
import matplotlib.pyplot as plt #for visualizing
from math import log10, sqrt
from scipy.ndimage import binary_dilation


def load_grayscale_image(image_path):
    """Loads an image from the specified path and converts it to grayscale.

    Handles both color and grayscale images.  Color images are converted to
    grayscale by averaging the RGB channels.  Includes error handling for
    file loading issues.

    Args:
        image_path (str): The path to the image file.

    Returns:
        np.ndarray: The grayscale image as a NumPy array, or None if loading fails.
    """
    try:
        input_image = Image.open(image_path)
        image_array = np.array(input_image)
        if len(image_array.shape) == 3:  # Color image (3 channels)
            return np.mean(image_array, axis=2).astype(np.uint8)
        return image_array.astype(np.uint8)  # Grayscale image (already 1 channel)
    except Exception as e:
        print(f"Error: Failed to load image: {e}")
        return None


def smooth_image_gaussian(input_image, smoothing_kernel_size=5, smoothing_sigma=1.0):
    """Applies Gaussian smoothing to the input image to reduce noise.

    Ensures the kernel size is odd for proper Gaussian blurring.

    Args:
        input_image (np.ndarray): The input image.
        smoothing_kernel_size (int): The size of the Gaussian kernel.
        smoothing_sigma (float): The standard deviation of the Gaussian kernel.

    Returns:
        np.ndarray: The smoothed image.
    """
    
    kernel_size = max(3, smoothing_kernel_size + (smoothing_kernel_size % 2 == 0))
    return cv2.GaussianBlur(input_image, (kernel_size, kernel_size), smoothing_sigma)


def non_maximum_suppression(gradient_magnitude, gradient_direction):
    """Applies non-maximum suppression using vectorized operations.

    This eliminates the nested loops for improved performance.

    Args:
        gradient_magnitude (np.ndarray): The magnitude of the image gradients.
        gradient_direction (np.ndarray): The direction of the image gradients (in radians).

    Returns:
        np.ndarray: The gradient magnitude image after non-maximum suppression.
    """
    
    rows, cols = gradient_magnitude.shape
    suppressed_edges = np.zeros_like(gradient_magnitude)

    # Pad the magnitude image
    padded_magnitude = np.pad(gradient_magnitude, 1, mode='constant')

    # Quantize the gradient direction into four main directions (0, 45, 90, 135 degrees)
    angle = gradient_direction * (180 / np.pi)
    angle[angle < 0] += 180
    angle_q = np.round(angle / 45) % 4  # 0: horizontal, 1: diagonal, 2: vertical, 3: other diagonal


    # Create masks for each direction
    horizontal_mask = angle_q == 0
    diag1_mask = angle_q == 1
    vertical_mask = angle_q == 2
    diag2_mask = angle_q == 3

    # Compare with neighbors in each direction using vectorized operations
    suppressed_edges[horizontal_mask] = np.where(
        padded_magnitude[1:-1, 1:-1][horizontal_mask] >= np.maximum(padded_magnitude[1:-1, :-2][horizontal_mask], padded_magnitude[1:-1, 2:][horizontal_mask]),
        gradient_magnitude[horizontal_mask], 0
    )

    suppressed_edges[diag1_mask] = np.where(
        padded_magnitude[1:-1, 1:-1][diag1_mask] >= np.maximum(padded_magnitude[:-2, :-2][diag1_mask], padded_magnitude[2:, 2:][diag1_mask]),
        gradient_magnitude[diag1_mask], 0
    )

    suppressed_edges[vertical_mask] = np.where(
        padded_magnitude[1:-1, 1:-1][vertical_mask] >= np.maximum(padded_magnitude[:-2, 1:-1][vertical_mask], padded_magnitude[2:, 1:-1][vertical_mask]),
        gradient_magnitude[vertical_mask], 0
    )

    suppressed_edges[diag2_mask] = np.where(
        padded_magnitude[1:-1, 1:-1][diag2_mask] >= np.maximum(padded_magnitude[:-2, 2:][diag2_mask], padded_magnitude[2:, :-2][diag2_mask]),
        gradient_magnitude[diag2_mask], 0
    )


    return suppressed_edges



def compute_image_gradients(input_image):
    """
    Computes the image gradients using the Sobel operator and applies non-maximum suppression.

    Args:
        input_image (np.ndarray): The input image.

    Returns:
        tuple: A tuple containing the normalized gradient magnitude, gradient direction,
               gradient in x direction, and gradient in y direction.
    """
    # Calculate gradients in x and y directions using Sobel operator
    gradient_x = cv2.Sobel(input_image, cv2.CV_64F, 1, 0, ksize=3)  # ksize=3 for a 3x3 Sobel kernel
    gradient_y = cv2.Sobel(input_image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate gradient magnitude and direction
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)  # Radians

    # Apply Non-Maximum Suppression (NMS) to thin the edges
    suppressed_magnitude = non_maximum_suppression(gradient_magnitude, gradient_direction)

    # Normalize the gradient magnitude to the range 0-255
    normalized_magnitude = cv2.normalize(suppressed_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return normalized_magnitude, gradient_direction, gradient_x, gradient_y




def enhance_edge_features(gradient_magnitude):
    """Enhances edge features using Otsu's thresholding.

    Otsu's method automatically determines an optimal threshold value by minimizing
    the intra-class variance between the background and foreground pixels.  It's
    particularly effective for bimodal histograms, which are common in edge
    detection scenarios. This function applies Otsu's thresholding to the gradient magnitude.   

    Args:
        gradient_magnitude (np.ndarray): The gradient magnitude image.

    Returns:
        np.ndarray: The thresholded edge image.
    """
    # Otsu's thresholding (using cv2.THRESH_OTSU flag)
    _, thresholded_edges = cv2.threshold(gradient_magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded_edges


def calculate_psnr(original_image, enhanced_image):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.  

    PSNR is a common metric for evaluating image quality, especially after
    processing like edge detection.  Higher PSNR values generally indicate
    better image quality.   The formula is defined as:  PSNR = 20 * log10(MAX / sqrt(MSE))  (dB)  


    Args:
        original_image (np.ndarray): The original image.
        enhanced_image (np.ndarray): The processed image.

    Returns:
        float: The PSNR value in decibels (dB).
    """
    mse = np.mean((original_image - enhanced_image) ** 2)
    if mse == 0:  # Handle perfect reconstruction (MSE = 0)
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def visualize_results(original_image, gradient_magnitude, gradient_direction, enhanced_edges, psnr_value):
    """Visualizes the results of the edge detection process.

    Displays the original image, gradient magnitude, gradient direction, and
    enhanced edges in a 2x2 grid using Matplotlib.  Includes titles and labels
    for clarity.  Shows the calculated PSNR value in the figure title.

    Args:
        original_image (np.ndarray): The original grayscale image.
        gradient_magnitude (np.ndarray): The magnitude of the image gradients.
        gradient_direction (np.ndarray): The direction of the image gradients.
        enhanced_edges (np.ndarray): The enhanced edges after hysteresis thresholding.
        low_threshold (float): The low threshold used for hysteresis.
        high_threshold (float): The high threshold used for hysteresis.
        psnr_value (float): The calculated PSNR value.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(original_image, cmap='gray')
    axes[0, 0].set_title('1. Original Image\n(Gray Scaled)', fontsize=10, pad=10)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gradient_magnitude, cmap='gray')
    axes[0, 1].set_title(f'2. Gradient Magnitude\n (Sobel Operator)', fontsize=10, pad=10)
    axes[0, 1].axis('off')

    gradient_direction_display = axes[1, 0].imshow(gradient_direction, cmap='hsv')
    axes[1, 0].set_title('3. Gradient Direction\n(Color Map)', fontsize=10, pad=10)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(enhanced_edges, cmap='gray')
    axes[1, 1].set_title(f'4. Enhanced Edges\n(Otsu\'s Thresholding)', fontsize=10, pad=10)
    axes[1, 1].axis('off')

    fig.suptitle(f"Edge Detection Results (PSNR: {psnr_value:.2f} dB)", fontsize=14)
    plt.tight_layout()
    plt.show()


def process_edge_detection(image_path, smoothing_kernel_size=5, smoothing_sigma=1.0, low_threshold_ratio=0.05, high_threshold_ratio=0.15):
    """
    Executes the complete Canny edge detection pipeline.

    This function orchestrates the entire edge detection process, from loading
    and preprocessing the image to calculating gradients, applying non-maximum
    suppression, hysteresis thresholding, and finally visualizing the results.

    Args:
        image_path (str): The path to the input image.
        smoothing_kernel_size (int): Size of the Gaussian smoothing kernel.
        smoothing_sigma (float): Standard deviation of the Gaussian kernel.
        low_threshold_ratio (float): Ratio for the low hysteresis threshold.
        high_threshold_ratio (float): Ratio for the high hysteresis threshold.
    """
    grayscale_image = load_grayscale_image(image_path)
    if grayscale_image is None:  # Exit if image loading fails
        return

    smoothed_image = smooth_image_gaussian(grayscale_image, smoothing_kernel_size, smoothing_sigma)
    
    gradient_magnitude, gradient_direction, gradient_x, gradient_y = compute_image_gradients(smoothed_image)
    
    enhanced_edges = enhance_edge_features(gradient_magnitude)
    
    psnr_value = calculate_psnr(grayscale_image, enhanced_edges)
    print(f"PSNR: {psnr_value:.2f} dB")
    
    visualize_results(grayscale_image, gradient_magnitude, gradient_direction, enhanced_edges,psnr_value)



def main():
    
    INPUT_IMAGE_PATH = "img/image01.jpeg"
    #INPUT_IMAGE_PATH = "img/image02.jpg"
    #INPUT_IMAGE_PATH = "img/image03.jpg"
    #INPUT_IMAGE_PATH = "img/image04.jpg"
    #INPUT_IMAGE_PATH = "img/image05.jpg"
    #INPUT_IMAGE_PATH = "img/image06.jpg"  
    
    SMOOTHING_KERNEL_SIZE = 5  # Size of the Gaussian kernel for smoothing
    SMOOTHING_SIGMA = 1.0  # Standard deviation for Gaussian smoothing
    
    process_edge_detection(
        image_path=INPUT_IMAGE_PATH,
        smoothing_kernel_size=SMOOTHING_KERNEL_SIZE,
        smoothing_sigma=SMOOTHING_SIGMA,
    )

if __name__ == "__main__":
    main()