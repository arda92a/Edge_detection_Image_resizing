# Image Resizing

This project provides an image resizing tool using two different interpolation methods: 

- **Nearest-Neighbor Interpolation**
- **Bilinear Interpolation**

It allows users to resize images via command-line arguments and supports both grayscale smoothing and edge detection.

## Features
- Resizes images using **Nearest-Neighbor** and **Bilinear** interpolation methods.
- Supports grayscale smoothing with a Gaussian kernel.
- Reads and writes images using OpenCV.
- Command-line interface for flexible usage.

## Installation
Ensure you have Python installed, then install the required dependencies:

```sh
pip install opencv-python numpy
```

## Usage
Run the script with the following command:

```sh
python image_resizing.py <input_image> <output_image> --width <new_width> --height <new_height> --resize_method <nn/bilinear>
```

### Example Usage
1. Resize an image using **Nearest-Neighbor Interpolation**:
```sh
python image_resizing.py input.jpg output --width 300 --height 300 --resize_method nn
```

2. Resize an image using **Bilinear Interpolation**:
```sh
python image_resizing.py input.jpg output --width 300 --height 300 --resize_method bilinear
```

## Functions Overview
- `smooth(img)`: Applies Gaussian smoothing to an image.
- `nn_resize(img, h, w, out_name)`: Resizes an image using Nearest-Neighbor interpolation.
- `bilinear_resize(img, h, w, out_name)`: Resizes an image using Bilinear interpolation.
- `__main__()`: Parses command-line arguments and executes the appropriate resizing function.

## Requirements
- Python 3.x
- OpenCV
- NumPy

## License
This project is licensed under the MIT License.

## Contributing
Feel free to fork the repository and submit pull requests for improvements!
