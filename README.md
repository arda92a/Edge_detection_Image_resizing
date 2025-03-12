# Image Processing Toolkit

This project provides an image processing toolkit that includes **image resizing** and **edge detection**, both implemented from scratch without using OpenCV's built-in functions.

## Features
- **Image Resizing:**
  - Implements **Nearest-Neighbor** and **Bilinear** interpolation methods.
  - **Does not use OpenCV's built-in resizing functions**.
  - Supports grayscale smoothing with a Gaussian kernel.
- **Edge Detection:**
  - Implements Canny Edge Detection using **custom-built functions**.
  - Includes Gaussian smoothing, Sobel gradient computation, Non-Maximum Suppression (NMS), and edge tracking.
  - **No built-in OpenCV edge detection functions are used**.
- Command-line interface for flexible usage.

## Installation
Ensure you have Python installed, then install the required dependencies:

```sh
pip install opencv-python numpy
```

## Usage
### Image Resizing
Run the script with the following command:

```sh
python image_resizing.py <input_image> <output_image> --width <new_width> --height <new_height> --resize_method <nn/bilinear>
```

#### Example Usage
1. Resize an image using **Nearest-Neighbor Interpolation**:
```sh
python image_resizing.py input.jpg output --width 300 --height 300 --resize_method nn
```

2. Resize an image using **Bilinear Interpolation**:
```sh
python image_resizing.py input.jpg output --width 300 --height 300 --resize_method bilinear
```

### Edge Detection
Run the script with the following command:

```sh
python edge_detection.py <input_image> <output_name> --low_thresh <low_value> --high_thresh <high_value>
```

#### Example Usage
```sh
python edge_detection.py input.jpg output --low_thresh 50 --high_thresh 150
```

## Functions Overview
- **Image Resizing:**
  - `smooth(img)`: Applies Gaussian smoothing to an image.
  - `nn_resize(img, h, w, out_name)`: Resizes an image using Nearest-Neighbor interpolation.
  - `bilinear_resize(img, h, w, out_name)`: Resizes an image using Bilinear interpolation.
- **Edge Detection:**
  - `smooth(img)`: Applies Gaussian smoothing before edge detection.
  - `conv2d(image, kernel)`: Performs 2D convolution without built-in functions.
  - `compute_gradients(img, out_name)`: Computes gradient magnitude and orientation using Sobel filters.
  - `NMS(image, angles)`: Applies Non-Maximum Suppression.
  - `thresholding(image, low, high, out_name)`: Performs double thresholding.
  - `tracking(result, strong_x, strong_y, out_name)`: Tracks strong edges.
  - `edge_detection(img, low_thresh, high_thresh, out_name)`: Runs the full edge detection pipeline.

## Sample images with image resizing applied

## Requirements
- Python 3.x
- OpenCV
- NumPy

## License
This project is licensed under the MIT License.

## Contributing
Feel free to fork the repository and submit pull requests for improvements!

