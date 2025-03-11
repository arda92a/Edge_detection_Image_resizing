import argparse
import cv2
import numpy as np
import math
from edge_detection import conv2d

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

def nn_interpolate(im, c, h, w):

    old_h, old_w, _ = im.shape 
    new_h, new_w = im.shape[:2]  

    x = round(h * (old_h / new_h))
    y = round(w * (old_w / new_w))

    x = min(old_h - 1, max(0, x))
    y = min(old_w - 1, max(0, y))

    return im[x, y, c]



def nn_resize(im, h, w, out_name):
    old_h, old_w, _ = im.shape 

    resized_nn = np.zeros((h, w, im.shape[2]), dtype=np.uint8)

    h_scale = old_h / h
    w_scale = old_w / w

    for i in range(h):
        for j in range(w):
            for c in range(im.shape[2]):  
                resized_nn[i, j, c] = nn_interpolate(im, c, i * h_scale, j * w_scale)

    cv2.imwrite(f"{out_name}_nn.jpg", resized_nn)
    return resized_nn


def bilinear_interpolate(img, c, h, w):
    org_h, org_w, _ = img.shape 

    x_floor, x_ceil = math.floor(h), min(org_h - 1, math.ceil(h))
    y_floor, y_ceil = math.floor(w), min(org_w - 1, math.ceil(w))

    if x_ceil == x_floor and y_ceil == y_floor:
        return img[x_floor, y_floor, c]
    elif x_ceil == x_floor: 
        return img[x_floor, y_floor, c] * (y_ceil - w) + img[x_floor, y_ceil, c] * (w - y_floor)
    elif y_ceil == y_floor:  
        return img[x_floor, y_floor, c] * (x_ceil - h) + img[x_ceil, y_floor, c] * (h - x_floor)
    else:  
        v1 = img[x_floor, y_floor, c]
        v2 = img[x_ceil, y_floor, c]
        v3 = img[x_floor, y_ceil, c]
        v4 = img[x_ceil, y_ceil, c]

        q1 = v1 * (x_ceil - h) + v2 * (h - x_floor)
        q2 = v3 * (x_ceil - h) + v4 * (h - x_floor)
        
        return q1 * (y_ceil - w) + q2 * (w - y_floor)


def bilinear_resize(img, new_h, new_w, out_name):
    old_h, old_w, _ = img.shape  

    resized = np.zeros((new_h, new_w, img.shape[2]), dtype=np.uint8)

    h_scale = old_h / new_h
    w_scale = old_w / new_w

    for i in range(new_h):
        for j in range(new_w):
            for c in range(img.shape[2]):  
                resized[i, j, c] = bilinear_interpolate(img, c, i * h_scale, j * w_scale)

    cv2.imwrite(f"{out_name}_bilinear.jpg", resized)
    return resized

    
    
def __main__():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run image resizing.")

    # Required argument for the image filename
    parser.add_argument('img_name', type=str, help="Path to the input image")
    # Required argument for the output filename
    parser.add_argument('out_name', type=str, help="Path to the output image")

    # Optional arguments for resizing dimensions
    parser.add_argument('--width', type=int, default=None, help="Width of the resized image")
    parser.add_argument('--height', type=int, default=None, help="Height of the resized image")

    # Choose between Nearest Neighbor (nn) and Bilinear (bilinear) resizing
    parser.add_argument('--resize_method', type=str, choices=['nn', 'bilinear'], default='nn', help="Resizing method to use")

    args = parser.parse_args()

    # Load the image
    img = cv2.imread(args.img_name)
    
    if args.width and args.height:
        if args.resize_method == "nn":
            resized_img = nn_resize(img, args.height, args.width, args.out_name)
            print("Resized image using Nearest-Neighbor interpolation.")
        elif args.resize_method == "bilinear":
            resized_img = bilinear_resize(img, args.height, args.width, args.out_name)
            print("Resized image using Bilinear interpolation.")
        
if __name__ == "__main__":
    __main__()