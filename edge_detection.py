import argparse
import cv2
import numpy as np


def smooth(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gaussian_kernel = np.array([
        [1,  4,  7,  4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1,  4,  7,  4, 1]
    ], dtype=np.float32)

    gaussian_kernel /= np.sum(gaussian_kernel)

    blurred_img = conv2d(img, gaussian_kernel)
    
    return blurred_img

def conv2d(image, kernel):
    
    k = kernel.shape[0]
    
    r, c = image.shape
    
    out_r, out_c = r - k + 1, c - k + 1
    out = np.zeros((out_r, out_c))
    
    for i in range(out_r):
        for j in range(out_c):
            out[i, j] = np.sum(image[i : i + k, j : j + k] * kernel)
    
    return out


def compute_gradients(img, out_name):

    sobel_x_kernel = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]]

    sobel_x_kernel = np.array(sobel_x_kernel)

    sobel_y_kernel = [
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ]
    sobel_y_kernel = np.array(sobel_y_kernel)

    img_Gx = conv2d(img,sobel_x_kernel)

    img_Gy = conv2d(img,sobel_y_kernel)

    img_gradient_magnitude = np.sqrt(img_Gx ** 2 + img_Gy ** 2)

    img_orientation = np.arctan2(img_Gy, img_Gx)

    cv2.imwrite('%s_gx.png' % out_name, img_Gx)
    cv2.imwrite('%s_gy.png' % out_name, img_Gy)
    cv2.imwrite('%s_grad.png' % out_name, img_gradient_magnitude)
    cv2.imwrite('%s_orit.png' % out_name, img_orientation)


    return img_gradient_magnitude, img_orientation

def NMS(image, angles):
    size = image.shape
    suppressed = np.zeros(size)
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                value_to_compare = max(image[i, j - 1], image[i, j + 1])
            elif (22.5 <= angles[i, j] < 67.5):
                value_to_compare = max(image[i - 1, j - 1], image[i + 1, j + 1])
            elif (67.5 <= angles[i, j] < 112.5):
                value_to_compare = max(image[i - 1, j], image[i + 1, j])
            else:
                value_to_compare = max(image[i + 1, j - 1], image[i - 1, j + 1])
            
            if image[i, j] >= value_to_compare:
                suppressed[i, j] = image[i, j]
    suppressed = np.multiply(suppressed, 255.0 / suppressed.max())
    return suppressed

def thresholding(image, low, high,out_name):
    weak = 50
    strong = 255
    result = np.zeros_like(image)
    
    weak_x, weak_y = np.where((image > low) & (image <= high))
    strong_x, strong_y = np.where(image >= high)
    
    result[strong_x, strong_y] = strong
    result[weak_x, weak_y] = weak


    cv2.imwrite('%s_low.png' % out_name, result * (result == weak))
    cv2.imwrite('%s_high.png' % out_name, result * (result == strong))

    return result, strong_x, strong_y

def tracking(result, strong_x, strong_y,out_name):
    weak = 50
    strong = 255
    
    dx = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    dy = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    size = result.shape
    
    while len(strong_x):
        x = strong_x[0]
        y = strong_y[0]
        strong_x = np.delete(strong_x, 0)
        strong_y = np.delete(strong_y, 0)
        
        for direction in range(8):
            new_x = x + dx[direction]
            new_y = y + dy[direction]
            
            if 0 <= new_x < size[0] and 0 <= new_y < size[1] and result[new_x, new_y] == weak:
                result[new_x, new_y] = strong
                strong_x = np.append(strong_x, new_x)
                strong_y = np.append(strong_y, new_y)
    
    result[result != strong] = 0
    cv2.imwrite('%s_track.png' % out_name, result)
    return result

def edge_detection(img, low_thresh, high_thresh, out_name):
    smoothed_img = smooth(img)

    gradient_magnitude, gradient_orientation = compute_gradients(smoothed_img, out_name)

    suppressed_img = NMS(gradient_magnitude, gradient_orientation)

    thresholded_img, strong_x, strong_y = thresholding(suppressed_img, low_thresh, high_thresh,out_name)

    final_edges = tracking(thresholded_img, strong_x, strong_y,out_name)

    cv2.imwrite('%s_edges.png' % out_name, final_edges)
    return final_edges
    
def __main__():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run image edge Detection.")

    # Required argument for the image filename
    parser.add_argument('img_name', type=str, help="Path to the input image")
    # Required argument for the output filename
    parser.add_argument('out_name', type=str, help="Path to the output image")

    # Canny edge detection threshold values (default values are provided)
    parser.add_argument('--low_thresh', type=int, default=50, help="Low threshold for Canny edge detection")
    parser.add_argument('--high_thresh', type=int, default=150, help="High threshold for Canny edge detection")

    args = parser.parse_args()

    # Load the image
    img = cv2.imread(args.img_name)
    edges = edge_detection(img, args.low_thresh, args.high_thresh, args.out_name)
    print("Completed!")


if __name__ == "__main__":
    __main__()
