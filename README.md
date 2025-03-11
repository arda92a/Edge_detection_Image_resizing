Image Resizing Tool
This Python tool allows you to resize images using two different interpolation methods: Nearest Neighbor (NN) and Bilinear interpolation. You can provide custom dimensions for the output image and choose between the two methods for resizing. This tool leverages OpenCV and NumPy for efficient image processing.

Features
Resize images using Nearest Neighbor interpolation.
Resize images using Bilinear interpolation.
Option to customize output image dimensions (height and width).
Simple command-line interface with support for input and output filenames.
Requirements
Before running the script, make sure you have the following libraries installed:

argparse: For parsing command-line arguments.
cv2 (OpenCV): For image loading, processing, and saving.
numpy: For numerical operations such as creating arrays and handling image data.
math: For mathematical operations like floor, ceil, and rounding.
You can install the required libraries using pip:

bash
Kopyala
Düzenle
pip install opencv-python numpy
Files
main.py: The main Python script that contains the resizing functions and logic.
edge_detection.py: Contains a helper function (conv2d) for image processing (used for smoothing).
Example Images:
input_image.jpg: Example of an original image.
output_image_nn.jpg: Example of the image resized using Nearest Neighbor interpolation.
output_image_bilinear.jpg: Example of the image resized using Bilinear interpolation.
How to Use
Running the Script
To run the script, use the command-line interface. The script requires two mandatory arguments for the input and output image paths, and it has additional optional arguments for specifying the dimensions of the resized image and the interpolation method.

Command-Line Arguments:
img_name (required): Path to the input image (e.g., input_image.jpg).
out_name (required): Path to save the output image (e.g., output_image).
--width (optional): Width of the resized image (integer value).
--height (optional): Height of the resized image (integer value).
--resize_method (optional): Method to use for resizing. Available options are:
nn: Nearest Neighbor interpolation (default).
bilinear: Bilinear interpolation.
Example Commands
Nearest Neighbor Resize: Resize an image to 500x500 pixels using Nearest Neighbor interpolation:

bash
Kopyala
Düzenle
python main.py input_image.jpg output_image --width 500 --height 500 --resize_method nn
Bilinear Resize: Resize an image to 500x500 pixels using Bilinear interpolation:

bash
Kopyala
Düzenle
python main.py input_image.jpg output_image --width 500 --height 500 --resize_method bilinear
Output Files
The resized image will be saved in the same directory with a suffix based on the chosen interpolation method:
output_image_nn.jpg for Nearest Neighbor resizing.
output_image_bilinear.jpg for Bilinear resizing.
Example Images
Here are examples of the images used and their resized outputs:

Original Image (input_image.jpg):


Resized Image using Nearest Neighbor (output_image_nn.jpg):


Resized Image using Bilinear (output_image_bilinear.jpg):


Detailed Explanation of Interpolation Methods
Nearest Neighbor Interpolation
Nearest Neighbor is the simplest interpolation method. When resizing, it finds the nearest pixel to the target location and uses that value. While fast, this method can lead to blocky images, especially when scaling up.

Advantages:

Fast and easy to implement.
Works well when performance is the priority over image quality.
Disadvantages:

Results in pixelated or blocky images, especially with large scaling factors.
Bilinear Interpolation
Bilinear Interpolation considers the nearest 2x2 neighborhood of known pixel values surrounding the target pixel. The pixel is then calculated using a weighted average of these 4 pixels. This method provides smoother transitions and better quality than Nearest Neighbor, particularly for scaling up images.

Advantages:

Produces smoother and higher-quality resized images.
Less blocky compared to Nearest Neighbor interpolation.
Disadvantages:

Slightly slower than Nearest Neighbor interpolation due to more computations.
Code Breakdown
smooth(img)
This function converts the input image to grayscale and applies a Gaussian blur to smooth the image. The smoothing is done using a 5x5 Gaussian kernel, which helps reduce noise and detail in the image.

nn_interpolate(im, c, h, w)
This function performs Nearest Neighbor interpolation for a given pixel (h, w) and color channel c in the image im. It simply finds the nearest pixel and returns its value.

nn_resize(im, h, w, out_name)
This function resizes the input image im to new dimensions (h, w) using Nearest Neighbor interpolation. It loops through each pixel in the output image and finds the nearest pixel from the input image.

bilinear_interpolate(img, c, h, w)
This function performs Bilinear interpolation for a given pixel (h, w) and color channel c in the image img. It calculates the weighted average of the 4 nearest neighboring pixels for smoother transitions.

bilinear_resize(img, new_h, new_w, out_name)
This function resizes the input image img to new dimensions (new_h, new_w) using Bilinear interpolation. It iterates over the pixels in the output image and applies the bilinear interpolation method to each pixel.

Additional Notes
The script supports both color (RGB) and grayscale images.
The resizing methods work for any image type supported by OpenCV.
The script saves resized images in .jpg format.
License
This project is licensed under the MIT License - see the LICENSE file for details.
