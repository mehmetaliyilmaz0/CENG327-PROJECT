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
- Gradient operators (Sobel, Prewitt)
- Automatic threshold determination (Otsu and adaptive)
- Gradient magnitude and direction analysis
"""

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def load_grayscale_image(image_path):
    """
    Loads an image and converts it to grayscale.

    Args:
        image_path (str): Path to the input image

    Returns:
        numpy.ndarray: Grayscale image array
    """
    try:
        input_image = Image.open(image_path)
        image_array = np.array(input_image)
        if len(image_array.shape) == 3:
            return np.mean(image_array, axis=2).astype(np.uint8)
        return image_array.astype(np.uint8)
    except Exception as e:
        print(f"Error: Failed to load image: {e}")
        return None

def smooth_image_gaussian(input_image, smoothing_kernel_size=5, smoothing_sigma=1.0):
    """
    Applies Gaussian smoothing to reduce noise.

    Args:
        input_image (numpy.ndarray): Input image array
        smoothing_kernel_size (int): Size of Gaussian kernel
        smoothing_sigma (float): Standard deviation for Gaussian

    Returns:
        numpy.ndarray: Smoothed image
    """
    kernel_size = max(3, smoothing_kernel_size + (smoothing_kernel_size % 2 == 0))
    return cv2.GaussianBlur(input_image, (kernel_size, kernel_size), smoothing_sigma)

def compute_image_gradients(input_image, gradient_operator='sobel'):
    """
    Computes image gradients using specified operator.

    Args:
        input_image (numpy.ndarray): Input image array
        gradient_operator (str): Type of gradient operator

    Returns:
        tuple: (gradient_magnitude, gradient_direction, gradient_x, gradient_y)
    """
    if gradient_operator == 'sobel':
        gradient_x = cv2.Sobel(input_image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(input_image, cv2.CV_64F, 0, 1, ksize=3)

    elif gradient_operator == 'prewitt':
        prewitt_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        prewitt_kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        gradient_x = cv2.filter2D(input_image, cv2.CV_64F, prewitt_kernel_x)
        gradient_y = cv2.filter2D(input_image, cv2.CV_64F, prewitt_kernel_y)

    else:
        raise ValueError("Invalid gradient operator. Use 'sobel' or 'prewitt'")

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)
    normalized_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return normalized_magnitude, gradient_direction, gradient_x, gradient_y

def calculate_adaptive_threshold(gradient_magnitude, threshold_method='otsu'):
    """
    Calculates threshold value using specified method.

    Args:
        gradient_magnitude (numpy.ndarray): Gradient magnitude image
        threshold_method (str): Method for threshold calculation

    Returns:
        float: Calculated threshold value
    """
    if threshold_method == 'otsu':
        computed_threshold = cv2.threshold(gradient_magnitude, 0, 255, cv2.THRESH_OTSU)[0]

    elif threshold_method == 'adaptive':
        magnitude_mean = np.mean(gradient_magnitude)
        magnitude_std = np.std(gradient_magnitude)
        computed_threshold = magnitude_mean + 0.5 * magnitude_std

    else:
        computed_threshold = 30

    return computed_threshold

def enhance_edge_features(gradient_magnitude, gradient_x, gradient_y, threshold_method='otsu'):
    """
    Enhances edge features using adaptive thresholding.

    Args:
        gradient_magnitude (numpy.ndarray): Gradient magnitude
        gradient_x (numpy.ndarray): X-direction gradient
        gradient_y (numpy.ndarray): Y-direction gradient
        threshold_method (str): Method for threshold calculation

    Returns:
        tuple: (enhanced_edges, threshold_value)
    """
    threshold_value = calculate_adaptive_threshold(gradient_magnitude, threshold_method)

    enhanced_edges = gradient_magnitude.copy()
    enhanced_edges[gradient_magnitude < threshold_value] = 0

    normalized_edges = cv2.normalize(enhanced_edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return normalized_edges, threshold_value

def visualize_results(original_image, gradient_magnitude, gradient_direction, 
                     enhanced_edges, operator_name, threshold_value):
    """
    Visualizes the edge detection results.

    Args:
        original_image (numpy.ndarray): Original input image
        gradient_magnitude (numpy.ndarray): Computed gradient magnitude
        gradient_direction (numpy.ndarray): Computed gradient direction
        enhanced_edges (numpy.ndarray): Enhanced edge image
        operator_name (str): Name of gradient operator used
        threshold_value (float): Applied threshold value
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # 1. Original Image
    axes[0, 0].imshow(original_image, cmap='gray')
    axes[0, 0].set_title('1. Original Image\n(Gray Scaled)', 
                        fontsize=10, pad=10)
    axes[0, 0].axis('off')

    # 2. Gradient Magnitude
    axes[0, 1].imshow(gradient_magnitude, cmap='gray')
    axes[0, 1].set_title(f'2. Gradient Magnitude\n({operator_name.capitalize()} Operator)', 
                        fontsize=10, pad=10)
    axes[0, 1].axis('off')

    # 3. Gradient Direction
    gradient_direction_display = axes[1, 0].imshow(gradient_direction, cmap='hsv')
    axes[1, 0].set_title('3. Gradient Direction\n(Color Map)', 
                        fontsize=10, pad=10)
    axes[1, 0].axis('off')


    # 4. Enhanced Edges
    axes[1, 1].imshow(enhanced_edges, cmap='gray')
    axes[1, 1].set_title(f'4. Enhanced Edges\nTreshold Value: {threshold_value:.2f}', 
                        fontsize=10, pad=10)
    axes[1, 1].axis('off')
    

    plt.tight_layout()
    plt.show()

def process_edge_detection(image_path, gradient_operator='sobel', 
                         smoothing_kernel_size=5, smoothing_sigma=1.0, 
                         threshold_method='otsu'):
    """
    Executes complete edge detection pipeline.

    Args:
        image_path (str): Path to input image
        gradient_operator (str): Type of gradient operator
        smoothing_kernel_size (int): Size of smoothing kernel
        smoothing_sigma (float): Sigma for Gaussian smoothing
        threshold_method (str): Method for threshold calculation
    """
    # Load and convert image
    grayscale_image = load_grayscale_image(image_path)
    if grayscale_image is None:
        return

    # Apply noise reduction
    smoothed_image = smooth_image_gaussian(
        grayscale_image, 
        smoothing_kernel_size, 
        smoothing_sigma
    )

    # Compute gradients
    gradient_magnitude, gradient_direction, gradient_x, gradient_y = compute_image_gradients(
        smoothed_image, 
        gradient_operator
    )

    # Enhance edges
    enhanced_edges, threshold_value = enhance_edge_features(
        gradient_magnitude, 
        gradient_x, 
        gradient_y, 
        threshold_method
    )

    # Display results
    visualize_results(
        grayscale_image,
        gradient_magnitude,
        gradient_direction,
        enhanced_edges,
        gradient_operator,
        threshold_value
    )

def main():
    
    # Configuration parameters
    
    #INPUT_IMAGE_PATH = "img/image01.jpeg"
    #INPUT_IMAGE_PATH = "img/image02.jpg"
    #INPUT_IMAGE_PATH = "img/image03.jpg"
    #INPUT_IMAGE_PATH = "img/image04.jpg"
    #INPUT_IMAGE_PATH = "img/image05.jpg"
    INPUT_IMAGE_PATH = "img/image06.jpg"
    
    GRADIENT_OPERATOR = 'sobel'
    #GRADIENT_OPERATOR = 'prewitt'

    SMOOTHING_KERNEL_SIZE = 5
    SMOOTHING_SIGMA = 1.0
    
    #THRESHOLD_METHOD = 'otsu'
    THRESHOLD_METHOD = 'adaptive'

    # Execute edge detection
    process_edge_detection(
        image_path=INPUT_IMAGE_PATH,
        gradient_operator=GRADIENT_OPERATOR,
        smoothing_kernel_size=SMOOTHING_KERNEL_SIZE,
        smoothing_sigma=SMOOTHING_SIGMA,
        threshold_method=THRESHOLD_METHOD
    )

if __name__ == "__main__":
    main()