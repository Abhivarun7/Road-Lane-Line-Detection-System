import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Function to open a file dialog and select an image
def select_image():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    image_path = filedialog.askopenfilename(title="Select an Image", 
                                             filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    return image_path

def grayscale(img):
    """Applies the Grayscale transform."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform."""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel."""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """Applies an image mask to keep only the region defined by the polygon formed from vertices."""
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """Draws lines on the image. Separates lines into left and right based on slope."""
    if lines is None:
        return

    slopes = []
    new_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else float('inf')

        if abs(slope) > 0.5:  # Only consider significant slopes
            slopes.append(slope)
            new_lines.append(line)

    lines = new_lines
    
    right_lines = []
    left_lines = []
    img_x_center = img.shape[1] / 2

    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        if slopes[i] > 0 and x1 > img_x_center and x2 > img_x_center:
            right_lines.append(line)
        elif slopes[i] < 0 and x1 < img_x_center and x2 < img_x_center:
            left_lines.append(line)

    # Draw right lines
    if right_lines:
        right_x = np.array([x1 for line in right_lines for x1 in line[0][::2]])
        right_y = np.array([y1 for line in right_lines for y1 in line[0][1::2]])
        right_m, right_b = np.polyfit(right_x, right_y, 1) if len(right_x) > 0 else (0, 0)
        
        y1, y2 = img.shape[0], int(img.shape[0] * 0.6)
        right_x1 = int((y1 - right_b) / right_m)
        right_x2 = int((y2 - right_b) / right_m)
        cv2.line(img, (right_x1, y1), (right_x2, y2), color, thickness)

    # Draw left lines
    if left_lines:
        left_x = np.array([x1 for line in left_lines for x1 in line[0][::2]])
        left_y = np.array([y1 for line in left_lines for y1 in line[0][1::2]])
        left_m, left_b = np.polyfit(left_x, left_y, 1) if len(left_x) > 0 else (0, 0)

        y1, y2 = img.shape[0], int(img.shape[0] * 0.6)
        left_x1 = int((y1 - left_b) / left_m)
        left_x2 = int((y2 - left_b) / left_m)
        cv2.line(img, (left_x1, y1), (left_x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """Detects lines in the image using the Hough Line Transform."""
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """Combines two images using weighted summation."""
    return cv2.addWeighted(initial_img, α, img, β, λ)

# Main code
if __name__ == "__main__":
    # Select image from files
    image_path = select_image()
    
    if not image_path:
        print("No image selected, exiting.")
        exit()

    # Load image
    image = cv2.imread(image_path)

    # Apply image processing
    gray_image = grayscale(image)
    blurred_image = gaussian_blur(gray_image, 5)
    edges = canny(blurred_image, 50, 150)
    roi_vertices = np.array([[(0, image.shape[0]), (450, 320), (490, 320), (image.shape[1], image.shape[0])]], dtype=np.int32)
    roi_image = region_of_interest(edges, roi_vertices)
    lines_image = hough_lines(roi_image, 1, np.pi/180, 15, 40, 20)

    # Combine the results
    result = weighted_img(lines_image, image)

    # Display the result
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
    plt.axis('off')
    plt.show()
